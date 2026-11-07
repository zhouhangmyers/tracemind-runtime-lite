"""Opt-in read-only tools for a trusted local workspace, with bounded text I/O."""

import asyncio
import os
import stat
from pathlib import Path, PureWindowsPath
from typing import cast

from .operations import Tool
from .protocol import Json, ToolDefinition

_SKIP = {"node_modules", "__pycache__", "build", "dist", "target"}
_PRIVATE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db"}


class ReadOnlyWorkspace:
    """Path checks assume the workspace is not being adversarially replaced.

    This is not an OS sandbox: another process racing directory replacement
    requires host-level isolation. File operations never write or launch processes.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = 256 * 1024,
        max_files: int = 500,
        max_entries: int = 5000,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace root must be a directory")
        for value in (max_file_bytes, max_files, max_entries):
            if type(value) is not int or value <= 0:
                raise ValueError("Workspace limits must be positive integers")
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_entries = max_entries

    def _resolve(self, name: str) -> Path:
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise ValueError("Use a relative path with forward slashes")
        relative = Path(name)
        windows = PureWindowsPath(name)
        if relative.is_absolute() or windows.drive or windows.is_reserved() or ":" in name:
            raise ValueError("Absolute paths and alternate streams are not allowed")
        current = self.root
        for part in relative.parts:
            if part == ".." or part.startswith(".") or part in _SKIP:
                raise ValueError("This path is outside the readable workspace scope")
            current = current / part
            if current.is_symlink() or current.is_junction():
                raise ValueError("Symlinks and junctions are not followed")
        resolved = current.resolve(strict=True)
        if not resolved.is_relative_to(self.root) or resolved.suffix.lower() in _PRIVATE_SUFFIXES:
            raise ValueError("This path is outside the readable workspace scope")
        return resolved

    def _scan(self, name: str) -> tuple[list[str], bool]:
        start = self._resolve(name)
        if not start.is_dir():
            raise ValueError("Expected a workspace directory")
        directories = [start]
        files: list[str] = []
        visited = 0
        while directories:
            directory = directories.pop()
            # Recheck entries before traversal; do not follow static links.
            directory = self._resolve(directory.relative_to(self.root).as_posix())
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    if visited > self.max_entries:
                        return sorted(files), True
                    if (
                        entry.name.startswith(".")
                        or entry.name in _SKIP
                        or entry.name.endswith(".egg-info")
                    ):
                        continue
                    relative = Path(entry.path).relative_to(self.root).as_posix()
                    try:
                        path = self._resolve(relative)
                        if path.is_dir():
                            directories.append(path)
                        elif path.is_file():
                            if len(files) == self.max_files:
                                return sorted(files), True
                            files.append(relative)
                    except (OSError, ValueError):
                        continue
        return sorted(files), False

    def list_files(self, path: str = ".") -> dict[str, Json]:
        files, truncated = self._scan(path)
        return {"files": list(files), "truncated": truncated}

    def _text(self, name: str) -> str:
        path = self._resolve(name)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(path, flags), "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > self.max_file_bytes:
                raise ValueError("Expected a regular file within the configured size limit")
            data = source.read(self.max_file_bytes + 1)
        if len(data) > self.max_file_bytes or b"\x00" in data:
            raise ValueError("Expected bounded UTF-8 text")
        return data.decode("utf-8")

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        line_count: int = 200,
    ) -> dict[str, Json]:
        if type(start_line) is not int or start_line < 1:
            raise ValueError("start_line must be a positive integer")
        if type(line_count) is not int or not 1 <= line_count <= 2000:
            raise ValueError("line_count must be between 1 and 2000")
        lines = self._text(path).splitlines()
        end = start_line - 1 + line_count
        return {
            "path": path,
            "start_line": start_line,
            "content": "\n".join(lines[start_line - 1 : end]),
            "total_lines": len(lines),
            "next_line": end + 1 if end < len(lines) else None,
        }

    def search_text(self, query: str, *, path: str = ".", limit: int = 50) -> dict[str, Json]:
        if not isinstance(query, str) or not 1 <= len(query) <= 256:
            raise ValueError("Search requires 1 to 256 characters")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100")
        files, truncated = self._scan(path)
        matches: list[Json] = []
        remaining = 4 * 1024 * 1024
        skipped = 0
        for name in files:
            try:
                text = self._text(name)
            except (OSError, ValueError):
                skipped += 1
                continue
            remaining -= len(text.encode("utf-8"))
            if remaining < 0:
                return {"matches": matches, "truncated": True, "skipped_files": skipped}
            for number, line in enumerate(text.splitlines(), 1):
                if query in line:
                    if len(matches) == limit:
                        return {"matches": matches, "truncated": True, "skipped_files": skipped}
                    matches.append({"path": name, "line": number, "text": line[:500]})
        return {"matches": matches, "truncated": truncated, "skipped_files": skipped}

    def tools(self) -> tuple[Tool, ...]:
        """The host opts in to reading this root; these tools are not registered by default."""
        path_schema: dict[str, Json] = {"type": "string", "minLength": 1, "maxLength": 512}

        async def listing(arguments: dict[str, Json]) -> Json:
            return await asyncio.to_thread(self.list_files, str(arguments.get("path", ".")))

        async def reading(arguments: dict[str, Json]) -> Json:
            return await asyncio.to_thread(
                self.read_file,
                str(arguments["path"]),
                start_line=int(cast(int, arguments.get("start_line", 1))),
                line_count=int(cast(int, arguments.get("line_count", 200))),
            )

        async def searching(arguments: dict[str, Json]) -> Json:
            return await asyncio.to_thread(
                self.search_text,
                str(arguments["query"]),
                path=str(arguments.get("path", ".")),
                limit=int(cast(int, arguments.get("limit", 50))),
            )

        return (
            Tool(
                ToolDefinition(
                    "workspace_list",
                    "List readable workspace files.",
                    {
                        "type": "object",
                        "properties": {"path": path_schema},
                        "additionalProperties": False,
                    },
                ),
                listing,
                False,
            ),
            Tool(
                ToolDefinition(
                    "workspace_read",
                    "Read a UTF-8 file by line range.",
                    {
                        "type": "object",
                        "properties": {
                            "path": path_schema,
                            "start_line": {"type": "integer", "minimum": 1},
                            "line_count": {"type": "integer", "minimum": 1, "maximum": 2000},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                reading,
                False,
            ),
            Tool(
                ToolDefinition(
                    "workspace_search",
                    "Search literal text in the workspace.",
                    {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1, "maxLength": 256},
                            "path": path_schema,
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
                searching,
                False,
            ),
        )
