import os
import sys
import logging
from groq import AsyncGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")

SYSTEM_PROMPT = """Ты senior Python/TypeScript разработчик.

Когда получаешь задачу:
1. Сначала вызови list_files("") чтобы понять структуру проекта
2. Прочитай нужные файлы через read_file перед изменением
3. Напиши или обнови код
4. Запуши через push_files с понятным commit_message в формате: feat/fix/refactor: описание

Правила:
- Пиши чистый, типизированный код
- Сохраняй существующий стиль проекта
- commit_message на английском в формате conventional commits
- Если файл не существует — создай его
- Если нужно изменить несколько файлов — пуши все за один раз
"""


async def run_agent(prompt: str) -> dict:
    """Запускает агента с MCP инструментами и возвращает результат."""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_github_server.py"],
        env={
            "GITHUB_TOKEN": GITHUB_TOKEN,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
        }
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Получаем инструменты от MCP сервера
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

            # Groq использует OpenAI-совместимый формат
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            # Конвертируем MCP tools в формат OpenAI/Groq
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

            # Агентный цикл
            while True:
                response = await client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=8096,
                    tools=groq_tools,
                    tool_choice="auto",
                    messages=messages,
                )

                choice = response.choices[0]
                finish_reason = choice.finish_reason
                logger.info(f"finish_reason: {finish_reason}")

                msg = choice.message
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in (msg.tool_calls or [])
                    ] or None
                })

                # Агент завершил работу
                if finish_reason == "stop" or not msg.tool_calls:
                    break

                # Выполняем tool calls через MCP
                import json
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    logger.info(f"Вызов инструмента: {name}({args})")

                    try:
                        result = await session.call_tool(name, args)
                        result_text = result.content[0].text if result.content else "ok"

                        # Извлекаем инфо о коммите
                        if name == "push_files":
                            commit_message = args.get("commit_message", commit_message)
                            files_changed += len(args.get("files", []))

                        logger.info(f"Результат {name}: {result_text[:100]}...")

                    except Exception as e:
                        result_text = f"Ошибка: {str(e)}"
                        logger.error(f"Ошибка вызова {name}: {e}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text
                    })

            return {
                "commit_message": commit_message,
                "files_changed": files_changed,
                "repo_url": f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
            }
