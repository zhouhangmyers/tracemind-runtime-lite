import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime import ScriptedModel

from tracemind_runtime import AgentRuntime, ModelResponse, ReadOnlyWorkspace, Session, ToolCall


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        (self.root / "src").mkdir()
        (self.root / "README.md").write_text("hello\nclass Example\nlast\n", encoding="utf-8")
        (self.root / "src/a.py").write_text("class A:\n    pass\n", encoding="utf-8")
        (self.root / ".env").write_text("example private data", encoding="utf-8")
        (self.root / "key.pem").write_text("example key data", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules/hidden.txt").write_text("class Hidden", encoding="utf-8")
        self.workspace = ReadOnlyWorkspace(self.root)

    def test_lists_only_the_selected_readable_files(self):
        self.assertEqual(
            self.workspace.list_files(),
            {
                "files": ["README.md", "src/a.py"],
                "truncated": False,
            },
        )
        self.assertEqual(self.workspace.list_files("src")["files"], ["src/a.py"])

    def test_read_reports_pagination_and_line_numbers(self):
        result = self.workspace.read_file("README.md", start_line=2, line_count=1)
        self.assertEqual(result["content"], "class Example")
        self.assertEqual(result["total_lines"], 3)
        self.assertEqual(result["next_line"], 3)
        self.assertEqual(self.workspace.read_file("README.md", start_line=3)["next_line"], None)

    def test_search_is_literal_and_reports_relative_locations(self):
        result = self.workspace.search_text("class ")
        self.assertEqual(
            [(m["path"], m["line"]) for m in result["matches"]], [("README.md", 2), ("src/a.py", 1)]
        )
        self.assertEqual(self.workspace.search_text(".*")["matches"], [])
        self.assertTrue(self.workspace.search_text("class ", limit=1)["truncated"])

    def test_traversal_absolute_hidden_and_alternate_stream_paths_are_rejected(self):
        for name in (
            "../outside.txt",
            str(Path(self.temp.name) / "outside.txt"),
            ".env",
            "key.pem",
            "node_modules/hidden.txt",
            "C:/outside",
            "src\\a.py",
            "README.md:stream",
            "CON",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.workspace.read_file(name)

    def test_symlink_is_not_read_or_listed(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("do not read", encoding="utf-8")
        link = self.root / "linked.txt"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with self.assertRaises(ValueError):
            self.workspace.read_file("linked.txt")
        self.assertNotIn("linked.txt", self.workspace.list_files()["files"])

    def test_file_growth_or_oversized_content_is_not_loaded(self):
        workspace = ReadOnlyWorkspace(self.root, max_file_bytes=5)
        with self.assertRaises(ValueError):
            workspace.read_file("README.md")
        (self.root / "binary.bin").write_bytes(b"abc\x00def")
        with self.assertRaises(ValueError):
            self.workspace.read_file("binary.bin")

    @unittest.skipUnless(os.name == "nt", "Windows junctions are platform-specific")
    def test_windows_junction_is_not_traversed(self):
        outside = Path(self.temp.name) / "outside-directory"
        outside.mkdir()
        (outside / "private.txt").write_text("outside the root", encoding="utf-8")
        link = self.root / "junction"
        process = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if process.returncode:
            self.skipTest("Junction creation unavailable")
        try:
            with self.assertRaises(ValueError):
                self.workspace.read_file("junction/private.txt")
            self.assertNotIn("junction/private.txt", self.workspace.list_files()["files"])
        finally:
            link.rmdir()

    def test_scan_limits_report_incomplete_results(self):
        self.assertTrue(ReadOnlyWorkspace(self.root, max_files=1).list_files()["truncated"])
        self.assertTrue(ReadOnlyWorkspace(self.root, max_entries=1).list_files()["truncated"])

    def test_no_files_are_modified_by_the_toolkit(self):
        before = {
            p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()
        }
        self.workspace.list_files()
        self.workspace.search_text("class")
        self.workspace.read_file("README.md")
        after = {
            p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()
        }
        self.assertEqual(before, after)

    def test_special_files_are_rejected_without_reading(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO not available on this platform")
        os.mkfifo(self.root / "pipe")
        with self.assertRaises(ValueError):
            self.workspace.read_file("pipe")


class WorkspaceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_registered_tools_run_through_normal_governance(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "a.txt").write_text("hello", encoding="utf-8")
            model = ScriptedModel(
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "read",
                            "workspace_read",
                            {"path": "a.txt"},
                        ),
                    )
                ),
                ModelResponse("done"),
            )
            result = await AgentRuntime(model, tools=ReadOnlyWorkspace(directory).tools()).run(
                Session(),
                "read the selected project",
            )
            self.assertEqual(result.operations[0].status, "ok")
            self.assertEqual(json.loads(result.operations[0].content)["result"]["content"], "hello")
