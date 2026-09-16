from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pressconf.lark_export import export_markdown_to_lark, lark_status
from pressconf.web import create_app


class ImmediateThread:
    def __init__(self, *, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self) -> None:
        self.target(*self.args)


class LarkMarkdownExportTests(unittest.TestCase):
    def test_status_does_not_treat_config_error_with_zero_exit_as_authorized(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": False, "error": {"type": "config", "message": "not configured"}}),
            stderr="",
        )
        with patch(
            "pressconf.lark_export.load_lark_config",
            return_value={"transport": "lark_cli", "identity": "user"},
        ), patch("pressconf.lark_export.lark_cli_path", return_value="/usr/bin/true"), patch(
            "pressconf.lark_export.subprocess.run", return_value=completed
        ):
            status = lark_status(Path("/tmp"))
        self.assertFalse(status["configured"])

    def test_cli_export_uses_v2_markdown_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "media_feedback.md"
            source.write_text("# 媒体反馈\n\n正文\n", encoding="utf-8")
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"data": {"document": {"document_id": "docx-test", "url": "https://example.feishu.cn/docx/docx-test"}}}
                ),
                stderr="",
            )
            with patch(
                "pressconf.lark_export.load_lark_config",
                return_value={"enabled": True, "transport": "lark_cli", "identity": "user"},
            ), patch("pressconf.lark_export.lark_cli_path", return_value="/usr/bin/true"), patch(
                "pressconf.lark_export.subprocess.run", return_value=completed
            ) as run:
                meta = export_markdown_to_lark(
                    base_dir=root,
                    result_dir=root,
                    source_path=source,
                    display_name="媒体反馈",
                    metadata={"module": "feedback"},
                )

            command = run.call_args.args[0]
            self.assertIn("--api-version", command)
            self.assertIn("v2", command)
            self.assertIn("--doc-format", command)
            self.assertIn("markdown", command)
            self.assertIn("--content", command)
            self.assertNotIn("--markdown", command)
            self.assertNotIn("--title", command)
            self.assertEqual(meta["module"], "feedback")
            self.assertEqual(meta["url"], "https://example.feishu.cn/docx/docx-test")
            self.assertTrue((root / "lark_export.md").exists())
            self.assertTrue((root / "lark_export.json").exists())


class FeedbackLarkRouteTests(unittest.TestCase):
    def test_feedback_can_be_transcribed_to_lark_and_polled(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job_id = "abcdef123456"
            result_dir = root / job_id
            result_dir.mkdir(parents=True)
            (result_dir / "media_feedback.md").write_text("# 原始反馈\n", encoding="utf-8")
            (result_dir / "meta.json").write_text(json.dumps({"title": "项目媒体反馈"}), encoding="utf-8")

            app = create_app()
            app.config["TESTING"] = True
            exported = {
                "url": "https://example.feishu.cn/docx/docx-test",
                "exported_at": "2026-09-02T12:00:00",
            }
            with patch("pressconf.web.FEEDBACK_ROOT", root), patch("pressconf.web.BASE_DIR", root), patch(
                "pressconf.web.threading.Thread", ImmediateThread
            ), patch("pressconf.web.export_markdown_to_lark", return_value=exported):
                response = app.test_client().post(
                    f"/api/feedback/{job_id}/lark/export",
                    json={"title": "编辑后的媒体反馈", "markdown": "# 编辑后\n\n正文"},
                )
                status = app.test_client().get(f"/api/feedback/{job_id}/lark")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.get_json()["status"], "done")
            self.assertEqual(status.get_json()["url"], exported["url"])
            self.assertEqual((result_dir / "lark_source.md").read_text(encoding="utf-8"), "# 编辑后\n\n正文\n")

    def test_feedback_preview_exposes_lark_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job_id = "123456abcdef"
            result_dir = root / job_id
            result_dir.mkdir(parents=True)
            (result_dir / "media_feedback.md").write_text("# 媒体反馈\n", encoding="utf-8")
            app = create_app()
            with patch("pressconf.web.FEEDBACK_ROOT", root):
                response = app.test_client().get(f"/feedback/{job_id}/preview")
            self.assertEqual(response.status_code, 200)
            self.assertIn("誊写到飞书文档", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
