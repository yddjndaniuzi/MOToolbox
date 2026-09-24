import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pressconf.web as web


class RefineRouteTests(unittest.TestCase):
    def test_clear_session_removes_refine_cache_but_keeps_source_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            result_dir = raw_root / "demo"
            (result_dir / "refine_chunks" / "old-run").mkdir(parents=True)
            (result_dir / "manifest.json").write_text("{}")
            (result_dir / "brief_base.md").write_text("基础稿")
            (result_dir / "brief_refined.md").write_text("正式稿")
            (result_dir / "brief_current.md").write_text("当前稿")
            (result_dir / "brief_refined.partial.md").write_text("临时稿")
            (result_dir / "refine_chunks" / "old-run" / "batch-001.json").write_text("{}")
            web.REFINE_JOBS.clear()
            web.REFINE_JOBS["demo"] = {"status": "error", "user_instruction": "旧要求"}

            with patch.object(web, "RAW_ROOT", raw_root):
                response = web.create_app().test_client().post("/result/demo/refine/clear-session")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json["ok"])
            self.assertFalse((result_dir / "refine_chunks").exists())
            self.assertFalse((result_dir / "brief_refined.partial.md").exists())
            self.assertEqual((result_dir / "brief_base.md").read_text(), "基础稿")
            self.assertEqual((result_dir / "brief_refined.md").read_text(), "正式稿")
            self.assertEqual((result_dir / "brief_current.md").read_text(), "当前稿")
            self.assertNotIn("demo", web.REFINE_JOBS)

    def test_clear_session_refuses_running_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            result_dir = raw_root / "demo"
            cache_dir = result_dir / "refine_chunks"
            cache_dir.mkdir(parents=True)
            (result_dir / "manifest.json").write_text("{}")
            web.REFINE_JOBS.clear()
            web.REFINE_JOBS["demo"] = {"status": "running"}

            with patch.object(web, "RAW_ROOT", raw_root):
                response = web.create_app().test_client().post("/result/demo/refine/clear-session")

            self.assertEqual(response.status_code, 409)
            self.assertTrue(cache_dir.exists())

    def test_refine_can_select_one_model_from_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            raw_root = base_dir / "pressconf" / "raw"
            result_dir = raw_root / "demo"
            result_dir.mkdir(parents=True)
            (result_dir / "manifest.json").write_text(json.dumps({"event_name": "Demo"}))
            config_dir = base_dir / "pressconf" / "config"
            config_dir.mkdir()
            (config_dir / "models.json").write_text(json.dumps({"models": [
                {"id": "choice", "name": "指定模型", "provider": "anthropic", "base_url": "https://test.invalid",
                 "model": "provider/choice", "secret_ref": "choice_key", "uses": ["brief_refine"], "enabled": True},
            ]}))
            (config_dir / "secrets.json").write_text(json.dumps({"choice_key": "test-secret"}))
            web.REFINE_JOBS.clear()

            with patch.object(web, "RAW_ROOT", raw_root), patch.object(web, "BASE_DIR", base_dir), \
                 patch.object(web.threading, "Thread") as thread_class:
                response = web.create_app().test_client().post("/result/demo/refine", data={"model_id": "choice"})

            self.assertEqual(response.status_code, 302)
            self.assertEqual(web.REFINE_JOBS["demo"]["model_id"], "choice")
            self.assertEqual(web.REFINE_JOBS["demo"]["model"], "指定模型")
            args = thread_class.call_args.kwargs["args"]
            self.assertEqual(args[3]["model"], "provider/choice")
            self.assertEqual(args[3]["api_key"], "test-secret")

    def test_refine_job_reuses_live_transcript_without_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            result_dir = raw_root / "live-demo"
            transcript_dir = result_dir / "transcript"
            transcript_dir.mkdir(parents=True)
            (result_dir / "manifest.json").write_text(json.dumps({"event_name": "直播演示", "stats": {"live": True}}))
            (transcript_dir / "asr.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n产品发布\n")
            (transcript_dir / "meta.json").write_text(json.dumps({"method": "asr", "path": "transcript/asr.srt"}))
            (transcript_dir / "quality.json").write_text(json.dumps({"unresolved_intervals": [{"start": 20}]}))
            (result_dir / "brief_refined.md").write_text("完整结果")
            web.REFINE_JOBS.clear()

            with patch.object(web, "RAW_ROOT", raw_root), \
                 patch.object(web, "ensure_transcript", side_effect=AssertionError("ASR must not run")), \
                 patch.object(web, "build_fact_ledger"), \
                 patch.object(web, "write_brief_base") as write_base, \
                 patch.object(web, "refine_brief", return_value=(result_dir / "brief_refined.md", {"model": "test"})):
                web.run_refine_job("live-demo", "直播演示")

            self.assertEqual(web.REFINE_JOBS["live-demo"]["status"], "done")
            self.assertIn("产品发布", write_base.call_args.args[2])
            self.assertEqual((result_dir / "brief_current.md").read_text(), "完整结果")

    def test_auto_routed_refine_starts_without_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            result_dir = raw_root / "demo"
            result_dir.mkdir(parents=True)
            (result_dir / "manifest.json").write_text(json.dumps({"event_name": "Demo"}), encoding="utf-8")
            web.REFINE_JOBS.clear()

            with patch.object(web, "RAW_ROOT", raw_root), patch.object(web.threading, "Thread") as thread_class:
                response = web.create_app().test_client().post("/result/demo/refine")

            self.assertEqual(response.status_code, 302)
            self.assertEqual(web.REFINE_JOBS["demo"]["status"], "running")
            self.assertEqual(web.REFINE_JOBS["demo"]["model"], "按任务强度自动选模")
            self.assertEqual(web.REFINE_JOBS["demo"]["provider"], "")
            thread_class.assert_called_once()
            thread_class.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
