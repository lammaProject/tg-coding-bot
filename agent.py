import os
import sys
import json
import asyncio
import logging
from groq import AsyncGroq, APIStatusError
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")

MAX_ITERATIONS = 10       # максимум итераций агента
AGENT_TIMEOUT = 120       # таймаут всего агента в секундах

# Модели в порядке приоритета — fallback при 429/rate limit
GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # основная: умнее, 100k TPD
    "llama-3.1-8b-instant",      # fallback: быстрее, 500k TPD
]

SYSTEM_PROMPT = """Ты senior Python/TypeScript разработчик.

Когда получаешь задачу:
1. Вызови list_files с directory="" чтобы понять структуру проекта. Если получишь ошибку — пропусти этот шаг и сразу пиши код.
2. Прочитай нужные файлы через read_file перед изменением (если знаешь путь).
3. Напиши или обнови код.
4. Запуши через push_files с понятным commit_message.

Правила:
- Пиши чистый, типизированный код
- commit_message на английском в формате conventional commits (feat/fix/refactor: описание)
- Если файл не существует — создай его
- Если нужно изменить несколько файлов — пуши все за один вызов push_files
- ВАЖНО: не вызывай один и тот же инструмент дважды если он вернул ошибку
- Максимум 10 итераций — работай быстро и эффективно
"""


async def _run_agent_inner(prompt: str) -> dict:
    # Наследуем всё окружение процесса и добавляем нужные переменные поверх
    subprocess_env = {**os.environ, **{
        k: v for k, v in {
            "GITHUB_TOKEN": GITHUB_TOKEN,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": os.getenv("GITHUB_BRANCH", "master"),
        }.items() if v is not None
    }}

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_github_server.py"],
        env=subprocess_env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema
                }
                for t in tools_result.tools
            ]

            logger.info(f"MCP инструменты: {[t['name'] for t in tools]}")

            client = AsyncGroq(api_key=GROQ_API_KEY)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            groq_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"]
                    }
                }
                for t in tools
            ]

            commit_message = "chore: update code"
            files_changed = 0
            iteration = 0
            current_model_idx = 0  # начинаем с лучшей модели

            while iteration < MAX_ITERATIONS:
                iteration += 1
                model = GROQ_MODELS[current_model_idx]
                logger.info(f"Итерация {iteration}/{MAX_ITERATIONS} [{model}]")

                try:
                    response = await client.chat.completions.create(
                        model=model,
                        max_tokens=4096,
                        tools=groq_tools,
                        tool_choice="auto",
                        messages=messages,
                    )
                except APIStatusError as e:
                    logger.error(f"Groq API error {e.status_code}: {e.response.text}")
                    # При rate limit — пробуем следующую модель
                    if e.status_code == 429 and current_model_idx + 1 < len(GROQ_MODELS):
                        current_model_idx += 1
                        next_model = GROQ_MODELS[current_model_idx]
                        logger.warning(f"Rate limit, переключаюсь на {next_model}")
                        iteration -= 1  # не считать эту итерацию
                        continue
                    raise Exception(f"Groq вернул ошибку {e.status_code}: {e.message}")

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                msg = choice.message

                logger.info(f"finish_reason: {finish_reason}, tool_calls: {bool(msg.tool_calls)}")

                # Добавляем ответ в историю
                assistant_msg = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_msg)

                # Агент завершил
                if finish_reason == "stop" or not msg.tool_calls:
                    logger.info("Агент завершил работу")
                    break

                # Выполняем tool calls
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    logger.info(f"Вызов: {name}")

                    try:
                        result = await session.call_tool(name, args)
                        result_text = result.content[0].text if result.content else "ok"

                        if name == "push_files":
                            commit_message = args.get("commit_message", commit_message)
                            files_changed += len(args.get("files", []))

                        logger.info(f"Результат {name}: {result_text[:150]}")

                    except Exception as e:
                        result_text = f"Ошибка: {str(e)}"
                        logger.error(f"Ошибка {name}: {e}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })

            if iteration >= MAX_ITERATIONS:
                logger.warning("Достигнут лимит итераций")

            return {
                "commit_message": commit_message,
                "files_changed": files_changed,
                "repo_url": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
            }


async def run_agent(prompt: str) -> dict:
    """Запускает агента с таймаутом."""
    try:
        return await asyncio.wait_for(_run_agent_inner(prompt), timeout=AGENT_TIMEOUT)
    except asyncio.TimeoutError:
        raise Exception(f"Агент не успел за {AGENT_TIMEOUT} секунд. Попробуй упростить задачу.")
