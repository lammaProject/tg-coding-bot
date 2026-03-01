import asyncio
import base64
import os
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = os.getenv("GITHUB_OWNER")
REPO = os.getenv("GITHUB_REPO")
DEFAULT_BRANCH = os.getenv("GITHUB_BRANCH", "master")

server = Server("github-coder")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="push_files",
            description="Создать или обновить файлы в GitHub репозитории и запушить коммит",
            inputSchema={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "description": "Список файлов для пуша",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            },
                            "required": ["path", "content"]
                        }
                    },
                    "commit_message": {
                        "type": "string",
                        "description": "Сообщение коммита в формате conventional commits"
                    },
                    "branch": {
                        "type": "string",
                        "description": "Ветка для пуша. Не указывай — сервер использует правильную ветку автоматически"
                    }
                },
                "required": ["files", "commit_message"]
            }
        ),
        Tool(
            name="read_file",
            description="Прочитать содержимое файла из репозитория",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь к файлу"}
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_files",
            description="Получить полное дерево всех файлов репозитория рекурсивно. Вызывай без аргументов.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="delete_file",
            description="Удалить файл из репозитория",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Путь к файлу"},
                    "commit_message": {"type": "string", "description": "Сообщение коммита"}
                },
                "required": ["path", "commit_message"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "push_files":
        result = await push_files(
            arguments["files"],
            arguments["commit_message"],
            arguments.get("branch") or DEFAULT_BRANCH
        )
        return [TextContent(type="text", text=result)]

    elif name == "read_file":
        content = await read_file(arguments["path"])
        return [TextContent(type="text", text=content)]

    elif name == "list_files":
        files = await list_files()
        return [TextContent(type="text", text=files)]

    elif name == "delete_file":
        result = await delete_file(arguments["path"], arguments["commit_message"])
        return [TextContent(type="text", text=result)]

    return [TextContent(type="text", text=f"Неизвестный инструмент: {name}")]


async def push_files(files: list, commit_message: str, branch: str = None) -> str:
    if not branch:
        branch = DEFAULT_BRANCH
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    pushed = []
    async with httpx.AsyncClient() as client:
        for file in files:
            path = file["path"]
            content = file["content"]

            # Проверяем существует ли файл (нужен sha для обновления)
            r = await client.get(
                f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}",
                headers=headers,
                params={"ref": branch}
            )
            sha = r.json().get("sha") if r.status_code == 200 else None

            body = {
                "message": commit_message,
                "content": base64.b64encode(content.encode()).decode(),
                "branch": branch
            }
            if sha:
                body["sha"] = sha

            resp = await client.put(
                f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}",
                headers=headers,
                json=body
            )

            if resp.status_code in (200, 201):
                pushed.append(path)
            else:
                return f"Ошибка при пуше {path}: {resp.text}"

    return (
        f"Успешно запушено {len(pushed)} файлов:\n"
        + "\n".join(f"  • {p}" for p in pushed)
        + f"\n\nhttps://github.com/{OWNER}/{REPO}"
    )


async def read_file(path: str) -> str:
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}",
            headers=headers
        )
        if r.status_code == 404:
            return f"Файл не найден: {path}"
        if r.status_code != 200:
            return f"Ошибка: {r.text}"

        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content


async def list_files(directory: str = "") -> str:
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    async with httpx.AsyncClient() as client:
        # Используем Git Trees API для рекурсивного списка всех файлов
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/trees/{DEFAULT_BRANCH}?recursive=1"
        r = await client.get(url, headers=headers)

        if r.status_code != 200:
            return f"Ошибка {r.status_code}: {r.text}"

        items = r.json().get("tree", [])
        result = []
        for item in items:
            if item["type"] == "blob":  # только файлы, без деревьев
                result.append(item["path"])

        return "\n".join(result) if result else "Репозиторий пуст"


async def delete_file(path: str, commit_message: str) -> str:
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    async with httpx.AsyncClient() as client:
        # Получаем sha файла
        r = await client.get(
            f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}",
            headers=headers
        )
        if r.status_code == 404:
            return f"Файл не найден: {path}"

        sha = r.json()["sha"]

        resp = await client.delete(
            f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}",
            headers=headers,
            json={"message": commit_message, "sha": sha}
        )

        if resp.status_code == 200:
            return f"Файл удалён: {path}"
        return f"Ошибка при удалении: {resp.text}"


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
