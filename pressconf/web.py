from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename

from pressconf.brief import write_brief_base
from pressconf.coverage import build_coverage_report
from pressconf.domains import TASK_TYPES, domain_from_manifest, list_domains, resolve_domain
from pressconf.fact_ledger import build_fact_ledger
from pressconf.keyframes import (
    CAPTURE_PROFILES,
    apply_auto_profile,
    extract_keyframes,
    extract_live_keyframes,
    format_timestamp,
    slugify,
    write_html,
    write_manifest,
    write_markdown,
)
from pressconf.transcript import ensure_transcript, read_existing_transcript, write_transcript_meta
from pressconf.transcript_reader import build_transcript_reader, read_transcript_reader, write_transcript_reader
from pressconf.refine import refine_brief
from pressconf.preview import render_markdown_preview
from pressconf.config_store import (
    MODEL_PRESETS,
    MODEL_STRENGTHS,
    delete_model_config,
    load_lark_config,
    load_models_config,
    load_knowledge_config,
    masked_secret,
    resolve_model,
    resolve_model_entry,
    save_lark_config,
    save_model_config,
    save_model_routing,
    save_knowledge_config,
)
from pressconf.knowledge import build_knowledge_index, load_knowledge_index, search_knowledge
from pressconf.lark_export import export_brief_to_lark, export_markdown_to_lark, lark_status
from pressconf.derivatives import fetch_lark_doc_markdown, generate_derivatives
from pressconf.content_review_lab import generate_content_review_lab
from pressconf.media_feedback import (
    extract_feedback_source,
    extract_price_analysis,
    generate_media_feedback,
    generate_review_video_analysis,
    sanitize_portable_markdown,
)
from pressconf.opinion_scan import (
    BROWSER_CAPTURE_BOOKMARKLET,
    SCAN_TEMPLATES,
    capture_active_browser_tab,
    format_browser_captures,
    load_browser_captures,
    read_csv_upload,
    run_scan,
    save_browser_capture,
    template_defaults,
)
from pressconf.runtime import bundled_python, data_root, resource_root, ytdlp_command, ytdlp_site_arg_variants


BASE_DIR = data_root()
RESOURCE_DIR = resource_root()
PYTHON_BIN = bundled_python()
RAW_ROOT = BASE_DIR / "pressconf" / "raw"
DERIVATIVE_ROOT = RAW_ROOT / "derivatives"
UPLOAD_ROOT = BASE_DIR / "pressconf" / "uploads"
DOWNLOAD_ROOT = BASE_DIR / "pressconf" / "downloads"
LIVE_ROOT = BASE_DIR / "pressconf" / "live"
FEEDBACK_ROOT = RAW_ROOT / "media_feedback"
REVIEW_VIDEO_ROOT = RAW_ROOT / "review_video"
CONTENT_REVIEW_LAB_ROOT = RAW_ROOT / "content_review_lab"
OPINION_SCAN_ROOT = RAW_ROOT / "opinion_scan"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
BRIEF_JOBS: dict[str, dict] = {}
BRIEF_JOBS_LOCK = threading.Lock()
REFINE_JOBS: dict[str, dict] = {}
REFINE_JOBS_LOCK = threading.Lock()
LARK_EXPORT_JOBS: dict[str, dict] = {}
LARK_EXPORT_JOBS_LOCK = threading.Lock()
DERIVATIVE_JOBS: dict[str, dict] = {}
DERIVATIVE_JOBS_LOCK = threading.Lock()
FEEDBACK_JOBS: dict[str, dict] = {}
FEEDBACK_JOBS_LOCK = threading.Lock()
FEEDBACK_LARK_EXPORT_JOBS: dict[str, dict] = {}
FEEDBACK_LARK_EXPORT_JOBS_LOCK = threading.Lock()
REVIEW_VIDEO_JOBS: dict[str, dict] = {}
REVIEW_VIDEO_JOBS_LOCK = threading.Lock()
CONTENT_REVIEW_LAB_JOBS: dict[str, dict] = {}
CONTENT_REVIEW_LAB_JOBS_LOCK = threading.Lock()


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(RESOURCE_DIR / "pressconf" / "templates"),
        static_folder=str(RESOURCE_DIR / "pressconf" / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024

    @app.get("/api/ytdlp")
    def ytdlp_status():
        from pressconf.ytdlp_update import status
        return jsonify(status())

    @app.post("/api/ytdlp/update")
    def ytdlp_update():
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify(error="仅允许本机更新"), 403
        if request.headers.get("Origin") not in {None, request.host_url.rstrip("/")}:
            return jsonify(error="不允许跨站请求"), 403
        if not request.is_json:
            return jsonify(error="需要 JSON 请求"), 415
        if sys.platform != "darwin":
            return jsonify(error="一键更新目前支持 macOS"), 400
        from pressconf.ytdlp_update import start_update
        started = start_update()
        return jsonify(running=True), 202 if started else 409

    def storage_locations():
        return {
            "data": ("应用数据", BASE_DIR),
            "raw": ("图包与生成结果", RAW_ROOT),
            "downloads": ("下载缓存", DOWNLOAD_ROOT),
            "uploads": ("上传文件", UPLOAD_ROOT),
            "live": ("直播录制", LIVE_ROOT),
            "models": ("语音模型缓存", Path.home() / ".cache" / "huggingface" / "hub"),
        }

    @app.context_processor
    def folder_context():
        return {"finder_available": sys.platform == "darwin"}

    @app.post("/api/storage/open")
    def open_storage():
        # Only local, same-origin UI requests may launch Finder.
        if request.remote_addr not in {"127.0.0.1", "::1"}:
            return jsonify(error="仅支持在本机打开访达"), 403
        if request.headers.get("Origin") not in {None, request.host_url.rstrip("/")}:
            return jsonify(error="不允许跨站请求"), 403
        if not request.is_json:
            return jsonify(error="需要 JSON 请求"), 415
        if sys.platform != "darwin":
            return jsonify(error="此功能仅支持 macOS"), 400
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="请求格式错误"), 400
        location = payload.get("location")
        if not isinstance(location, str) or location not in storage_locations():
            return jsonify(error="未知目录"), 400
        path = storage_locations()[location][1].resolve()
        slug = payload.get("slug")
        if slug is not None:
            if location != "raw" or not isinstance(slug, str) or not slug or Path(slug).name != slug or slug in {".", ".."}:
                return jsonify(error="无效图包"), 400
            target = (path / slug).resolve()
            if path not in target.parents or not target.is_dir():
                return jsonify(error="图包不存在"), 404
            path = target
        try:
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["/usr/bin/open", str(path)], check=True, timeout=10, capture_output=True)
        except (OSError, subprocess.SubprocessError):
            return jsonify(error="无法打开访达，请稍后重试"), 500
        return jsonify(ok=True)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            profiles=CAPTURE_PROFILES,
            domains=list_domains(),
            task_types=TASK_TYPES,
            jobs=list_jobs(),
        )

    @app.get("/admin")
    def admin():
        models_config = load_models_config(BASE_DIR)
        knowledge_config = load_knowledge_config(BASE_DIR)
        knowledge_index = load_knowledge_index(BASE_DIR)
        lark_config = load_lark_config(BASE_DIR)
        models = models_config.get("models", [])
        selected_model_id = request.args.get("model", "").strip()
        new_model = request.args.get("new") == "1"
        model = {} if new_model else next(
            (item for item in models if item.get("id") == selected_model_id),
            models[0] if models else {},
        )
        search_query = request.args.get("q", "").strip()
        return render_template(
            "admin.html",
            storage_locations=storage_locations(),
            models=models,
            model=model,
            model_routes=models_config.get("routing_by_strength") or {},
            model_strengths=MODEL_STRENGTHS,
            model_presets=MODEL_PRESETS,
            secret_mask=masked_secret(BASE_DIR, model.get("secret_ref", "")) if model else "未配置",
            knowledge_config=knowledge_config,
            knowledge_index=knowledge_index,
            lark_config=lark_config,
            lark_status=lark_status(BASE_DIR),
            search_query=search_query,
            search_results=search_knowledge(BASE_DIR, search_query) if search_query else [],
        )

    @app.get("/derivatives")
    def derivatives():
        return render_template("derivatives.html")

    @app.get("/history/<module>")
    def module_history(module: str):
        module_meta = history_module_meta(module)
        if not module_meta:
            return jsonify({"ok": False, "error": "Unsupported history module"}), 404
        return render_template(
            "history.html",
            module=module,
            module_title=module_meta["title"],
            module_home=module_meta["home"],
            entries=list_history_entries(module),
        )

    @app.get("/feedback")
    def feedback():
        return render_template("media_feedback.html")

    @app.get("/review-video")
    def review_video():
        return render_template("review_video.html", domains=list_domains())

    @app.get("/review-content-lab")
    def review_content_lab():
        return render_template("review_content_lab.html")

    @app.get("/calculator")
    def calculator():
        return render_template("calculator.html")

    @app.get("/opinion-scan")
    def opinion_scan():
        template_key = request.args.get("template", "product_launch")
        defaults = template_defaults(template_key)
        return render_template(
            "opinion_scan.html",
            templates=SCAN_TEMPLATES,
            defaults=defaults,
            bookmarklet=BROWSER_CAPTURE_BOOKMARKLET,
            result=None,
            error="",
        )

    @app.post("/opinion-scan")
    def opinion_scan_generate():
        template_key = request.form.get("template_key", "product_launch")
        selected_sources = request.form.getlist("sources")
        defaults = template_defaults(template_key)
        job_id = uuid.uuid4().hex[:12]
        payload = {
            "base_dir": str(BASE_DIR),
            "template_key": template_key,
            "target": request.form.get("target", ""),
            "keywords": request.form.get("keywords", ""),
            "aliases": request.form.get("aliases", ""),
            "excludes": request.form.get("excludes", ""),
            "risk_words": request.form.get("risk_words", ""),
            "topics": request.form.get("topics", ""),
            "time_range": request.form.get("time_range", ""),
            "sources": selected_sources or defaults["sources"],
            "pasted_content": request.form.get("pasted_content", ""),
            "urls": request.form.get("urls", ""),
            "fetch_urls": bool(request.form.get("fetch_urls")),
            "use_ai": bool(request.form.get("use_ai")),
        }
        try:
            payload["csv_rows"] = read_csv_upload(request.files.get("csv_file"))
            result = run_scan(payload, OPINION_SCAN_ROOT / job_id)
            return render_template(
                "opinion_scan.html",
                templates=SCAN_TEMPLATES,
                defaults=template_defaults(template_key),
                bookmarklet=BROWSER_CAPTURE_BOOKMARKLET,
                result={**result, "job_id": job_id},
                error="",
                form=payload,
            )
        except Exception as exc:
            return render_template(
                "opinion_scan.html",
                templates=SCAN_TEMPLATES,
                defaults=defaults,
                bookmarklet=BROWSER_CAPTURE_BOOKMARKLET,
                result=None,
                error=str(exc),
                form=payload,
            ), 400

    @app.route("/api/opinion-scan/browser-capture", methods=["POST", "OPTIONS"])
    def opinion_scan_browser_capture():
        if request.method == "OPTIONS":
            return cors_json({})
        raw = request.get_data(as_text=True) or "{}"
        try:
            payload = json.loads(raw)
            capture = save_browser_capture(OPINION_SCAN_ROOT, payload)
            return cors_json({"ok": True, "capture": capture})
        except Exception as exc:
            return cors_json({"ok": False, "error": str(exc)}, status=400)

    @app.get("/api/opinion-scan/browser-captures")
    def opinion_scan_browser_captures():
        captures = load_browser_captures(OPINION_SCAN_ROOT)
        return jsonify(
            {
                "ok": True,
                "captures": captures,
                "pasted_content": format_browser_captures(captures),
            }
        )

    @app.post("/api/opinion-scan/capture-active-browser")
    def opinion_scan_capture_active_browser():
        payload = request.get_json(silent=True) or {}
        browser = str(payload.get("browser") or "chrome")
        try:
            capture = save_browser_capture(OPINION_SCAN_ROOT, capture_active_browser_tab(browser))
            return jsonify({"ok": True, "capture": capture})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.post("/api/derivatives/generate")
    def derivative_generate():
        job_id = uuid.uuid4().hex[:12]
        payload = request.get_json(silent=True)
        if payload is None:
            payload = request.form.to_dict(flat=True)
            payload["requestedOutputs"] = request.form.getlist("requestedOutputs")
            upload_dir = DERIVATIVE_ROOT / job_id / "uploads"
            try:
                for form_key, payload_key in (
                    ("whitepaperUpload", "whitepaperUploadPath"),
                    ("introUpload", "introUploadPath"),
                ):
                    upload = request.files.get(form_key)
                    if upload and upload.filename:
                        payload[payload_key] = str(save_derivative_upload(upload, upload_dir, payload_key))
            except ValueError as exc:
                return jsonify({"ok": False, "error": str(exc)}), 400
        with DERIVATIVE_JOBS_LOCK:
            DERIVATIVE_JOBS[job_id] = {
                "id": job_id,
                "status": "running",
                "message": "准备读取材料",
                "percent": 5,
                "preview": "",
                "result": "",
            }
        thread = threading.Thread(target=run_derivative_job, args=(job_id, payload), daemon=True)
        thread.start()
        return jsonify({"ok": True, "job_id": job_id})

    @app.get("/api/derivatives/job/<job_id>")
    def derivative_job_status(job_id: str):
        with DERIVATIVE_JOBS_LOCK:
            job_data = dict(DERIVATIVE_JOBS.get(job_id, {"status": "missing", "message": "任务不存在"}))
        return jsonify(job_data)

    @app.get("/derivatives/<job_id>/preview")
    def derivative_preview(job_id: str):
        result_dir = DERIVATIVE_ROOT / job_id
        result_path = result_dir / "derivatives.md"
        content = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
        meta = read_manifest(result_dir / "meta.json")
        return render_template(
            "feedback_preview.html",
            job_id=job_id,
            title=history_title_from_meta(meta, "副产物预览"),
            content=content,
            blocks=render_markdown_preview(content, "") if content else [],
            back_url=url_for("derivatives"),
            back_label="返回副产物生成",
            markdown_url=url_for("raw_file", filename=f"derivatives/{job_id}/derivatives.md"),
            empty_text="还没有可预览的副产物内容。",
            obsidian_payload={"module": "derivatives", "id": job_id},
        )

    @app.post("/api/feedback/generate")
    def feedback_generate():
        job_id = uuid.uuid4().hex[:12]
        job_dir = FEEDBACK_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        questionnaire = request.files.get("questionnaire")
        if not questionnaire or not questionnaire.filename:
            return jsonify({"ok": False, "error": "请上传媒体问卷原始表单。"}), 400
        try:
            questionnaire_path = save_feedback_upload(questionnaire, job_dir, "questionnaire")
            reference_paths = [
                save_feedback_upload(item, job_dir, f"reference-{index + 1}")
                for index, item in enumerate(request.files.getlist("references"))
                if item and item.filename
            ]
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        payload = {
            "product_name": request.form.get("product_name", ""),
            "meeting_context": request.form.get("meeting_context", ""),
            "additional_instruction": request.form.get("additional_instruction", ""),
            "questionnaire_path": str(questionnaire_path),
            "reference_paths": [str(path) for path in reference_paths],
        }
        with FEEDBACK_JOBS_LOCK:
            FEEDBACK_JOBS[job_id] = {
                "id": job_id,
                "status": "running",
                "message": "准备读取问卷",
                "percent": 5,
                "preview": "",
            }
        thread = threading.Thread(target=run_feedback_job, args=(job_id, payload), daemon=True)
        thread.start()
        return jsonify({"ok": True, "job_id": job_id})

    @app.get("/api/feedback/job/<job_id>")
    def feedback_job_status(job_id: str):
        with FEEDBACK_JOBS_LOCK:
            job_data = dict(FEEDBACK_JOBS.get(job_id, {"status": "missing", "message": "任务不存在"}))
        return jsonify(job_data)

    @app.post("/api/feedback/<job_id>/lark/export")
    def feedback_lark_export_generate(job_id: str):
        if not re.fullmatch(r"[0-9a-f]{12}", job_id):
            return jsonify({"ok": False, "error": "无效的媒体反馈任务 ID。"}), 400
        result_dir = FEEDBACK_ROOT / job_id
        generated_path = result_dir / "media_feedback.md"
        if not generated_path.exists():
            return jsonify({"ok": False, "error": "还没有可誊写的媒体反馈 Markdown。"}), 404

        body = request.get_json(silent=True) or {}
        markdown = str(body.get("markdown") or "").strip()
        if not markdown:
            markdown = generated_path.read_text(encoding="utf-8").strip()
        if not markdown:
            return jsonify({"ok": False, "error": "媒体反馈内容为空，无法誊写。"}), 400

        manifest = read_manifest(result_dir / "meta.json")
        title = str(body.get("title") or manifest.get("title") or f"媒体反馈-{job_id}").strip()
        source_path = result_dir / "lark_source.md"
        source_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")

        with FEEDBACK_LARK_EXPORT_JOBS_LOCK:
            current = FEEDBACK_LARK_EXPORT_JOBS.get(job_id, {})
            if current.get("status") == "running":
                return jsonify({"ok": True, "job_id": job_id, "status": "running"})
            FEEDBACK_LARK_EXPORT_JOBS[job_id] = {
                "id": job_id,
                "status": "running",
                "message": "准备誊写到飞书文档",
                "percent": 8,
            }
        thread = threading.Thread(
            target=run_feedback_lark_export_job,
            args=(job_id, title, source_path),
            daemon=True,
        )
        thread.start()
        return jsonify({"ok": True, "job_id": job_id, "status": "running"})

    @app.get("/api/feedback/<job_id>/lark")
    def feedback_lark_export_status(job_id: str):
        if not re.fullmatch(r"[0-9a-f]{12}", job_id):
            return jsonify({"status": "missing", "message": "无效的媒体反馈任务 ID。"}), 400
        with FEEDBACK_LARK_EXPORT_JOBS_LOCK:
            job_data = dict(
                FEEDBACK_LARK_EXPORT_JOBS.get(job_id, {"status": "idle", "message": "等待誊写"})
            )
        meta = read_manifest(FEEDBACK_ROOT / job_id / "lark_export.json")
        if meta:
            job_data["has_export"] = True
            job_data["url"] = meta.get("url", "")
            job_data["exported_at"] = meta.get("exported_at", "")
        return jsonify(job_data)

    @app.post("/api/review-video/generate")
    def review_video_generate():
        job_id = uuid.uuid4().hex[:12]
        job_dir = REVIEW_VIDEO_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            source = resolve_review_video_source(job_dir)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        payload = {
            "product_name": request.form.get("product_name", ""),
            "media_name": request.form.get("media_name", ""),
            "video_title": request.form.get("video_title", ""),
            "additional_instruction": request.form.get("additional_instruction", ""),
            "domain_id": request.form.get("domain_id", "auto"),
            "source": serialize_review_source(source),
        }
        with REVIEW_VIDEO_JOBS_LOCK:
            REVIEW_VIDEO_JOBS[job_id] = {
                "id": job_id,
                "status": "running",
                "message": "准备获取逐字稿",
                "percent": 5,
                "preview": "",
            }
        thread = threading.Thread(target=run_review_video_job, args=(job_id, payload), daemon=True)
        thread.start()
        return jsonify({"ok": True, "job_id": job_id})

    @app.get("/api/review-video/job/<job_id>")
    def review_video_job_status(job_id: str):
        with REVIEW_VIDEO_JOBS_LOCK:
            job_data = dict(REVIEW_VIDEO_JOBS.get(job_id, {"status": "missing", "message": "任务不存在"}))
        return jsonify(job_data)

    @app.post("/api/review-content-lab/generate")
    def review_content_lab_generate():
        job_id = uuid.uuid4().hex[:12]
        job_dir = CONTENT_REVIEW_LAB_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            source = resolve_content_review_lab_source(job_dir)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        payload = {
            "product_name": request.form.get("product_name", ""),
            "media_name": request.form.get("media_name", ""),
            "content_title": request.form.get("content_title", ""),
            "review_reference": request.form.get("review_reference", ""),
            "additional_instruction": request.form.get("additional_instruction", ""),
            "source": serialize_review_source(source),
        }
        with CONTENT_REVIEW_LAB_JOBS_LOCK:
            CONTENT_REVIEW_LAB_JOBS[job_id] = {
                "id": job_id,
                "status": "running",
                "message": "准备读取待审内容",
                "percent": 5,
                "preview": "",
            }
        thread = threading.Thread(target=run_content_review_lab_job, args=(job_id, payload), daemon=True)
        thread.start()
        return jsonify({"ok": True, "job_id": job_id})

    @app.get("/api/review-content-lab/job/<job_id>")
    def review_content_lab_job_status(job_id: str):
        with CONTENT_REVIEW_LAB_JOBS_LOCK:
            job_data = dict(CONTENT_REVIEW_LAB_JOBS.get(job_id, {"status": "missing", "message": "任务不存在"}))
        return jsonify(job_data)

    @app.get("/feedback/<job_id>/preview")
    def feedback_preview(job_id: str):
        result_dir = FEEDBACK_ROOT / job_id
        result_path = result_dir / "media_feedback.md"
        manifest = read_manifest(result_dir / "meta.json")
        preview_title = history_title_from_meta(manifest, "媒体反馈预览")
        obsidian_payload = {"module": "feedback", "id": job_id}
        lark_payload = {"job_id": job_id, "title": preview_title}
        if not result_path.exists():
            return render_template(
                "feedback_preview.html",
                job_id=job_id,
                title=preview_title,
                content="",
                blocks=[],
                back_url=url_for("feedback"),
                back_label="返回媒体反馈",
                markdown_url=url_for("raw_file", filename=f"media_feedback/{job_id}/media_feedback.md"),
                obsidian_payload=obsidian_payload,
                lark_payload=lark_payload,
                lark_export=read_manifest(result_dir / "lark_export.json"),
            )
        content = result_path.read_text(encoding="utf-8")
        return render_template(
            "feedback_preview.html",
            job_id=job_id,
            title=preview_title,
            content=content,
            blocks=render_markdown_preview(content, ""),
            back_url=url_for("feedback"),
            back_label="返回媒体反馈",
            markdown_url=url_for("raw_file", filename=f"media_feedback/{job_id}/media_feedback.md"),
            obsidian_payload=obsidian_payload,
            lark_payload=lark_payload,
            lark_export=read_manifest(result_dir / "lark_export.json"),
        )

    @app.get("/review-video/<job_id>/preview")
    def review_video_preview(job_id: str):
        result_dir = REVIEW_VIDEO_ROOT / job_id
        result_path = result_dir / "review_video_analysis.md"
        obsidian_payload = {"module": "review_video", "id": job_id}
        if not result_path.exists():
            return render_template(
                "feedback_preview.html",
                job_id=job_id,
                title="评测视频分析预览",
                content="",
                blocks=[],
                back_url=url_for("review_video"),
                back_label="返回评测视频分析",
                markdown_url=url_for("raw_file", filename=f"review_video/{job_id}/review_video_analysis.md"),
                empty_text="还没有可预览的评测视频分析内容。",
                transcript_url=url_for("review_video_transcript", job_id=job_id),
                obsidian_payload=obsidian_payload,
            )
        content = result_path.read_text(encoding="utf-8")
        return render_template(
            "feedback_preview.html",
            job_id=job_id,
            title="评测视频分析预览",
            content=content,
            blocks=render_markdown_preview(content, ""),
            back_url=url_for("review_video"),
            back_label="返回评测视频分析",
            markdown_url=url_for("raw_file", filename=f"review_video/{job_id}/review_video_analysis.md"),
            empty_text="还没有可预览的评测视频分析内容。",
            transcript_url=url_for("review_video_transcript", job_id=job_id),
            obsidian_payload=obsidian_payload,
        )

    @app.get("/review-video/<job_id>/transcript")
    def review_video_transcript(job_id: str):
        result_dir = REVIEW_VIDEO_ROOT / job_id
        reader = read_transcript_reader(result_dir)
        meta = read_manifest(result_dir / "meta.json")
        manifest = read_manifest(result_dir / "manifest.json")
        video_path = review_video_playback_path(manifest)
        if not reader or not video_path:
            return render_template(
                "transcript_reader.html",
                title=history_title_from_meta(meta, "逐字稿查阅"),
                job_id=job_id,
                reader={},
                video_url="",
                back_url=url_for("review_video_preview", job_id=job_id),
            )
        return render_template(
            "transcript_reader.html",
            title=history_title_from_meta(meta, "逐字稿查阅"),
            job_id=job_id,
            reader=reader,
            video_url=url_for("review_video_source_video", job_id=job_id),
            back_url=url_for("review_video_preview", job_id=job_id),
        )

    @app.get("/review-video/<job_id>/source-video")
    def review_video_source_video(job_id: str):
        video_path = review_video_playback_path(read_manifest(REVIEW_VIDEO_ROOT / job_id / "manifest.json"))
        if not video_path:
            return jsonify({"ok": False, "error": "没有找到可播放的视频。"}), 404
        return send_file(video_path, conditional=True)

    @app.get("/review-content-lab/<job_id>/preview")
    def review_content_lab_preview(job_id: str):
        result_dir = CONTENT_REVIEW_LAB_ROOT / job_id
        result_path = result_dir / "content_review_lab.md"
        content = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
        meta = read_manifest(result_dir / "meta.json")
        return render_template(
            "feedback_preview.html",
            job_id=job_id,
            title=history_title_from_meta(meta, "媒体待审内容审核预览"),
            content=content,
            blocks=render_markdown_preview(content, "") if content else [],
            back_url=url_for("review_content_lab"),
            back_label="返回审核实验室",
            markdown_url=url_for("raw_file", filename=f"content_review_lab/{job_id}/content_review_lab.md"),
            empty_text="还没有可预览的媒体审核报告。",
            obsidian_payload=None,
        )

    @app.post("/admin/model")
    def admin_model_save():
        saved_model_id = save_model_config(BASE_DIR, request.form)
        return redirect(url_for("admin", model=saved_model_id) + "#models")

    @app.post("/admin/model/routing")
    def admin_model_routing_save():
        save_model_routing(BASE_DIR, request.form)
        return redirect(url_for("admin") + "#models")

    @app.post("/admin/model/delete")
    def admin_model_delete():
        delete_model_config(BASE_DIR, request.form.get("model_id", "").strip())
        return redirect(url_for("admin") + "#models")

    @app.post("/admin/knowledge")
    def admin_knowledge_save():
        save_knowledge_config(BASE_DIR, request.form)
        return redirect(url_for("admin"))

    @app.post("/admin/knowledge/reindex")
    def admin_knowledge_reindex():
        build_knowledge_index(BASE_DIR)
        return redirect(url_for("admin"))

    @app.post("/admin/lark")
    def admin_lark_save():
        save_lark_config(BASE_DIR, request.form)
        return redirect(url_for("admin") + "#lark")

    @app.get("/api/knowledge/search")
    def api_knowledge_search():
        query = request.args.get("q", "")
        return jsonify({"results": search_knowledge(BASE_DIR, query)})

    @app.post("/capture")
    def capture():
        source = resolve_video_source()
        event_name = request.form.get("event_name", "").strip() or source["default_event_name"]
        profile = request.form.get("profile", "auto")
        if profile not in CAPTURE_PROFILES:
            profile = "auto"

        args = build_capture_args(event_name=event_name, profile=profile)
        output_dir = RAW_ROOT / slugify(event_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "status": "queued",
                "event_name": event_name,
                "slug": output_dir.name,
                "source_type": source["type"],
                "message": "等待开始",
                "percent": 0,
                "keyframes": 0,
                "sampled_frames": 0,
                "control": "running",
            }

        thread = threading.Thread(
            target=run_capture_job,
            args=(job_id, source, output_dir, event_name, args),
            daemon=True,
        )
        thread.start()

        return redirect(url_for("job", job_id=job_id))

    @app.get("/job/<job_id>")
    def job(job_id: str):
        return render_template("job.html", job_id=job_id)

    @app.get("/api/job/<job_id>")
    def job_status(job_id: str):
        with JOBS_LOCK:
            job_data = dict(JOBS.get(job_id, {"status": "missing", "message": "任务不存在"}))
        return jsonify(job_data)

    @app.get("/api/result/<slug>/brief")
    def brief_status(slug: str):
        with BRIEF_JOBS_LOCK:
            job_data = dict(BRIEF_JOBS.get(slug, {"status": "idle", "message": "等待生成"}))
        result_dir = RAW_ROOT / slug
        job_data["has_brief"] = (result_dir / "brief_base.md").exists()
        job_data["has_transcript"] = first_transcript_file(result_dir) is not None
        return jsonify(job_data)

    @app.get("/api/result/<slug>/refine")
    def refine_status(slug: str):
        with REFINE_JOBS_LOCK:
            job_data = dict(REFINE_JOBS.get(slug, {"status": "idle", "message": "等待精加工"}))
        result_dir = RAW_ROOT / slug
        partial_path = result_dir / "brief_refined.partial.md"
        job_data["has_refined"] = (result_dir / "brief_refined.md").exists()
        if partial_path.exists():
            partial = partial_path.read_text(encoding="utf-8", errors="ignore")
            job_data["generated_chars"] = len(partial)
            job_data["preview"] = tail_preview(partial)
        return jsonify(job_data)

    @app.get("/api/result/<slug>/lark")
    def lark_export_status(slug: str):
        with LARK_EXPORT_JOBS_LOCK:
            job_data = dict(LARK_EXPORT_JOBS.get(slug, {"status": "idle", "message": "等待写入"}))
        meta = read_manifest(RAW_ROOT / slug / "lark_export.json")
        if meta:
            job_data["has_export"] = True
            job_data["url"] = meta.get("url", "")
            job_data["exported_at"] = meta.get("exported_at", "")
        return jsonify(job_data)

    @app.post("/api/job/<job_id>/control")
    def job_control(job_id: str):
        action = request.form.get("action", "")
        if action not in {"pause", "resume", "stop"}:
            return jsonify({"ok": False, "error": "Unsupported action"}), 400
        control = {"pause": "pause", "resume": "running", "stop": "stop"}[action]
        message = {"pause": "已暂停", "resume": "继续采集中", "stop": "正在停止"}[action]
        update_job(job_id, control=control, message=message)
        return jsonify({"ok": True, "control": control})

    @app.get("/api/job/<job_id>/frames")
    def job_frames(job_id: str):
        with JOBS_LOCK:
            job_data = dict(JOBS.get(job_id, {}))
        slug = job_data.get("slug")
        if not slug:
            return jsonify({"frames": []})
        frames_dir = RAW_ROOT / slug / "frames"
        if not frames_dir.exists():
            return jsonify({"frames": []})
        frames = [
            {"name": frame_path.name, "url": f"/raw/{slug}/frames/{frame_path.name}"}
            for frame_path in sorted(frames_dir.glob("*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)[:24]
        ]
        return jsonify({"frames": frames})

    @app.post("/api/title")
    def title_lookup():
        source_type = request.form.get("source_type", "local")
        try:
            title = resolve_source_title(source_type)
            return jsonify({"ok": True, "title": title})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/result/<slug>")
    def result(slug: str):
        manifest = read_manifest(RAW_ROOT / slug / "manifest.json")
        display_name = event_display_name(manifest, slug)
        return render_template(
            "result.html",
            slug=slug,
            manifest=manifest,
            display_name=display_name,
            keyframes=manifest.get("keyframes", []),
            jobs=list_jobs(),
        )

    @app.get("/raw/<path:filename>")
    def raw_file(filename: str):
        return send_from_directory(RAW_ROOT, filename)

    @app.get("/tmp-images/<token>/<slug>/frames/<filename>")
    def temp_image_file(token: str, slug: str, filename: str):
        config = load_lark_config(BASE_DIR)
        if token != config.get("image_host_token"):
            return jsonify({"ok": False, "error": "Forbidden"}), 403
        if not re.match(r"^[^/]+\.jpg$", filename):
            return jsonify({"ok": False, "error": "Invalid image"}), 400
        frame_dir = (RAW_ROOT / slug / "frames").resolve()
        raw_root = RAW_ROOT.resolve()
        if raw_root not in frame_dir.parents:
            return jsonify({"ok": False, "error": "Invalid path"}), 400
        return send_from_directory(frame_dir, filename)

    @app.get("/result/<slug>/brief")
    def brief_form(slug: str):
        result_dir = RAW_ROOT / slug
        manifest = read_manifest(result_dir / "manifest.json")
        display_name = event_display_name(manifest, slug)
        brief_path = result_dir / "brief_base.md"
        refined_path = result_dir / "brief_refined.md"
        brief_content = brief_path.read_text(encoding="utf-8") if brief_path.exists() else ""
        refined_content = refined_path.read_text(encoding="utf-8") if refined_path.exists() else ""
        transcript_meta = read_manifest(result_dir / "transcript" / "meta.json")
        refine_meta = read_manifest(result_dir / "refine_meta.json")
        refine_models_by_strength = {
            strength: resolve_model(BASE_DIR, "brief_refine", strength)
            for strength in MODEL_STRENGTHS
        }
        refine_pool_models = [
            model for model in load_models_config(BASE_DIR).get("models", [])
            if model.get("enabled", True) and "brief_refine" in model.get("uses", [])
        ]
        active_refine_model = refine_models_by_strength["heavy"]
        with BRIEF_JOBS_LOCK:
            brief_job = dict(BRIEF_JOBS.get(slug, {}))
        with REFINE_JOBS_LOCK:
            refine_job = dict(REFINE_JOBS.get(slug, {}))
        return render_template(
            "brief.html",
            slug=slug,
            manifest=manifest,
            display_name=display_name,
            brief_content=brief_content,
            refined_content=refined_content,
            transcript_meta=transcript_meta,
            refine_meta=refine_meta,
            active_refine_model=active_refine_model,
            refine_models_by_strength=refine_models_by_strength,
            refine_pool_models=refine_pool_models,
            brief_job=brief_job,
            refine_job=refine_job,
            has_coverage_report=(result_dir / "coverage_report.json").exists(),
            has_fact_ledger=(result_dir / "fact_ledger.json").exists(),
            error=request.args.get("error", ""),
        )

    @app.post("/result/<slug>/brief")
    def brief_generate(slug: str):
        result_dir = RAW_ROOT / slug
        manifest = read_manifest(result_dir / "manifest.json")
        display_name = event_display_name(manifest, slug)
        with BRIEF_JOBS_LOCK:
            current = BRIEF_JOBS.get(slug, {})
            if current.get("status") == "running":
                return redirect(url_for("brief_form", slug=slug))
            BRIEF_JOBS[slug] = {
                "status": "running",
                "message": "准备获取转写",
                "percent": 3,
            }
        thread = threading.Thread(target=run_brief_job, args=(slug, display_name), daemon=True)
        thread.start()
        return redirect(url_for("brief_form", slug=slug))

    @app.post("/result/<slug>/refine")
    def refine_generate(slug: str):
        manifest = read_manifest(RAW_ROOT / slug / "manifest.json")
        display_name = event_display_name(manifest, slug)
        next_page = request.args.get("next", "brief")
        user_instruction = request.form.get("user_instruction", "").strip()
        tier_override = request.form.get("brief_tier", "").strip().lower()
        if tier_override not in {"full", "lite", "nano"}:
            tier_override = ""
        format_override = request.form.get("event_format", "").strip().lower()
        if format_override not in {"hardware", "software"}:
            format_override = ""
        selected_model_id = request.form.get("model_id", "").strip()
        model_config = None
        model_label = "按任务强度自动选模"
        if selected_model_id:
            selected_model = next((model for model in load_models_config(BASE_DIR).get("models", [])
                                   if model.get("id") == selected_model_id and model.get("enabled", True)
                                   and "brief_refine" in model.get("uses", [])), None)
            if selected_model is None:
                return redirect(url_for("brief_form", slug=slug, error="所选模型不可用，请重新选择。"))
            model_config = resolve_model_entry(BASE_DIR, selected_model)
            model_label = model_config["name"]
        with REFINE_JOBS_LOCK:
            current = REFINE_JOBS.get(slug, {})
            if current.get("status") == "running":
                if next_page == "preview":
                    return redirect(url_for("brief_preview", slug=slug, variant="refined"))
                return redirect(url_for("brief_form", slug=slug))
            REFINE_JOBS[slug] = {
                "status": "running",
                "message": f"{model_label} 精加工中",
                "percent": 8,
                "user_instruction": user_instruction,
                "brief_tier": tier_override,
                "event_format": format_override,
                "model_id": selected_model_id,
                "model": model_label,
                "provider": (model_config or {}).get("provider", ""),
        }
        thread = threading.Thread(target=run_refine_job, args=(slug, display_name, user_instruction, model_config, tier_override, format_override), daemon=True)
        thread.start()
        if next_page == "preview":
            return redirect(url_for("brief_preview", slug=slug, variant="refined"))
        return redirect(url_for("brief_form", slug=slug))

    @app.post("/result/<slug>/refine/clear-session")
    def refine_clear_session(slug: str):
        result_dir = (RAW_ROOT / slug).resolve()
        if RAW_ROOT.resolve() not in result_dir.parents or not (result_dir / "manifest.json").is_file():
            return jsonify({"ok": False, "error": "简报不存在。"}), 404
        with REFINE_JOBS_LOCK:
            if REFINE_JOBS.get(slug, {}).get("status") == "running":
                return jsonify({"ok": False, "error": "精加工正在运行，请完成后再清除 Session。"}), 409
            cache_dir = result_dir / "refine_chunks"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            for filename in ("brief_refined.partial.md", "brief_refined.complete.tmp"):
                (result_dir / filename).unlink(missing_ok=True)
            REFINE_JOBS.pop(slug, None)
        return jsonify({"ok": True})

    @app.post("/result/<slug>/lark/export")
    def lark_export_generate(slug: str):
        manifest = read_manifest(RAW_ROOT / slug / "manifest.json")
        display_name = event_display_name(manifest, slug)
        variant = request.args.get("variant", "refined")
        with LARK_EXPORT_JOBS_LOCK:
            current = LARK_EXPORT_JOBS.get(slug, {})
            if current.get("status") == "running":
                return redirect(url_for("brief_preview", slug=slug, variant=variant))
            LARK_EXPORT_JOBS[slug] = {
                "status": "running",
                "message": "准备写入飞书文档",
                "percent": 8,
            }
        thread = threading.Thread(target=run_lark_export_job, args=(slug, display_name, variant), daemon=True)
        thread.start()
        return redirect(url_for("brief_preview", slug=slug, variant=variant))

    @app.post("/api/obsidian/export")
    def obsidian_export():
        payload = request.get_json(silent=True) or {}
        try:
            meta = export_generated_markdown_to_obsidian(payload)
            return jsonify({"ok": True, **meta})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/result/<slug>/preview")
    def brief_preview(slug: str):
        variant = request.args.get("variant", "refined")
        result_dir = RAW_ROOT / slug
        manifest = read_manifest(result_dir / "manifest.json")
        display_name = event_display_name(manifest, slug)
        filename = "brief_refined.md" if variant == "refined" and (result_dir / "brief_refined.md").exists() else "brief_base.md"
        if variant == "refined" and (result_dir / "brief_current.md").exists():
            filename = "brief_current.md"
        path = result_dir / filename
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        with LARK_EXPORT_JOBS_LOCK:
            lark_job = dict(LARK_EXPORT_JOBS.get(slug, {}))
        return render_template(
            "preview.html",
            slug=slug,
            display_name=display_name,
            variant="refined" if filename in {"brief_refined.md", "brief_current.md"} else "base",
            filename=filename,
            content=content,
            blocks=render_markdown_preview(content, slug) if content else [],
            keyframes=manifest.get("keyframes", []),
            lark_job=lark_job,
            lark_export=read_manifest(result_dir / "lark_export.json"),
            obsidian_payload={"module": "brief", "id": slug, "variant": variant},
        )

    @app.post("/api/result/<slug>/preview/edit")
    def preview_edit(slug: str):
        payload = request.get_json(silent=True) or {}
        variant = str(payload.get("variant") or "refined")
        action = str(payload.get("action") or "")
        block_index = int(payload.get("block_index") or 0)
        source_block_index_raw = payload.get("source_block_index")
        source_block_index = int(source_block_index_raw) if source_block_index_raw not in {None, ""} else None
        image_path = str(payload.get("image_path") or "").strip()
        try:
            filename = edit_preview_markdown(slug, variant, action, block_index, image_path, source_block_index)
            return jsonify({"ok": True, "filename": filename})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/result/<slug>/delete")
    def delete_result(slug: str):
        ok = delete_result_dir(slug)
        if not ok:
            return jsonify({"ok": False, "error": "Invalid result path"}), 400
        return redirect(url_for("index"))

    @app.post("/results/delete")
    def delete_results():
        slugs = request.form.getlist("slugs")
        deleted = [slug for slug in slugs if delete_result_dir(slug)]
        return jsonify({"ok": True, "deleted": deleted})

    return app


def delete_result_dir(slug: str) -> bool:
    result_dir = (RAW_ROOT / slug).resolve()
    raw_root = RAW_ROOT.resolve()
    if raw_root not in result_dir.parents or result_dir == raw_root:
        return False
    if result_dir.exists():
        shutil.rmtree(result_dir)
    return True


def history_module_meta(module: str) -> dict[str, str] | None:
    return {
        "brief": {"title": "发布会简报历史", "home": "/"},
        "derivatives": {"title": "副产物历史", "home": "/derivatives"},
        "feedback": {"title": "媒体反馈历史", "home": "/feedback"},
        "review_video": {"title": "评测视频分析历史", "home": "/review-video"},
    }.get(module)


def list_history_entries(module: str) -> list[dict]:
    if module == "brief":
        return list_brief_history_entries()
    roots = {
        "derivatives": (DERIVATIVE_ROOT, "derivatives.md", "derivatives"),
        "feedback": (FEEDBACK_ROOT, "media_feedback.md", "feedback"),
        "review_video": (REVIEW_VIDEO_ROOT, "review_video_analysis.md", "review-video"),
    }
    if module not in roots:
        return []
    root, markdown_name, preview_prefix = roots[module]
    entries = []
    for markdown_path in sorted(root.glob(f"*/{markdown_name}"), key=lambda item: item.stat().st_mtime, reverse=True):
        job_dir = markdown_path.parent
        meta = read_manifest(job_dir / "meta.json")
        entries.append(
            {
                "id": job_dir.name,
                "title": history_title_from_meta(meta, job_dir.name),
                "updated": format_file_time(markdown_path),
                "preview_url": f"/{preview_prefix}/{job_dir.name}/preview",
                "markdown_url": f"/raw/{root.name}/{job_dir.name}/{markdown_name}",
                "obsidian_payload": {"module": module, "id": job_dir.name},
            }
        )
    return entries


def list_brief_history_entries() -> list[dict]:
    entries = []
    if not RAW_ROOT.exists():
        return entries
    for result_dir in sorted(RAW_ROOT.glob("*/"), key=lambda item: item.stat().st_mtime, reverse=True):
        if not (result_dir / "manifest.json").exists():
            continue
        path, filename = markdown_file_for_history(result_dir)
        if not path:
            continue
        manifest = read_manifest(result_dir / "manifest.json")
        slug = result_dir.name
        variant = "refined" if filename in {"brief_refined.md", "brief_current.md"} else "base"
        entries.append(
            {
                "id": slug,
                "title": event_display_name(manifest, slug),
                "updated": format_file_time(path),
                "preview_url": f"/result/{slug}/preview?variant={variant}",
                "markdown_url": f"/raw/{slug}/{filename}",
                "obsidian_payload": {"module": "brief", "id": slug, "variant": variant},
            }
        )
    return entries


def markdown_file_for_history(result_dir: Path) -> tuple[Path | None, str]:
    for filename in ("brief_current.md", "brief_refined.md", "brief_base.md", "keyframes.md"):
        path = result_dir / filename
        if path.exists():
            return path, filename
    return None, ""


def history_title_from_meta(meta: dict, fallback: str) -> str:
    for key in ("title", "product_name", "productName", "video_title", "videoTitle", "name"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return fallback


def format_file_time(path: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))


def export_generated_markdown_to_obsidian(payload: dict) -> dict:
    module = str(payload.get("module") or "").strip()
    item_id = str(payload.get("id") or "").strip()
    variant = str(payload.get("variant") or "refined").strip()
    source_path, title, folder_name = generated_markdown_source(module, item_id, variant)
    if not source_path.exists():
        raise ValueError("没有找到可写入 Obsidian 的 Markdown。")

    config = load_knowledge_config(BASE_DIR)
    vault_path = Path(str(config.get("vault_path") or "")).expanduser()
    if not vault_path.exists() or not vault_path.is_dir():
        raise ValueError("后台绑定的 Obsidian vault 不存在，请先在后台控制台配置知识库路径。")
    include_subdir = str(config.get("include_subdir") or "").strip().strip("/")
    target_dir = vault_path / include_subdir if include_subdir else vault_path
    if vault_path.resolve() not in target_dir.resolve().parents and target_dir.resolve() != vault_path.resolve():
        raise ValueError("Obsidian 子目录配置不合法。")
    target_dir = target_dir / "MOtoolbox 回写" / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = unique_markdown_path(target_dir, title)
    content = source_path.read_text(encoding="utf-8")
    if module in {"feedback", "review_video"}:
        content = sanitize_portable_markdown(content)
    target_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return {
        "path": str(target_path),
        "title": target_path.stem,
    }


def generated_markdown_source(module: str, item_id: str, variant: str) -> tuple[Path, str, str]:
    if not re.match(r"^[^/]+$", item_id):
        raise ValueError("Invalid item id.")
    if module == "brief":
        result_dir = RAW_ROOT / item_id
        manifest = read_manifest(result_dir / "manifest.json")
        path, filename = markdown_file_for_variant(result_dir, "refined" if variant == "refined" else "base")
        return path, event_display_name(manifest, item_id), "发布会简报"
    if module == "derivatives":
        result_dir = DERIVATIVE_ROOT / item_id
        meta = read_manifest(result_dir / "meta.json")
        return result_dir / "derivatives.md", history_title_from_meta(meta, f"副产物-{item_id}"), "副产物"
    if module == "feedback":
        result_dir = FEEDBACK_ROOT / item_id
        meta = read_manifest(result_dir / "meta.json")
        return result_dir / "media_feedback.md", history_title_from_meta(meta, f"媒体反馈-{item_id}"), "媒体反馈"
    if module == "review_video":
        result_dir = REVIEW_VIDEO_ROOT / item_id
        meta = read_manifest(result_dir / "meta.json")
        return result_dir / "review_video_analysis.md", history_title_from_meta(meta, f"评测视频分析-{item_id}"), "评测视频分析"
    raise ValueError("Unsupported module.")


def unique_markdown_path(target_dir: Path, title: str) -> Path:
    safe_title = sanitize_markdown_filename(title) or "MOtoolbox 输出"
    target_path = target_dir / f"{safe_title}.md"
    if not target_path.exists():
        return target_path
    suffix = time.strftime("%Y%m%d-%H%M%S")
    return target_dir / f"{safe_title}-{suffix}.md"


def sanitize_markdown_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", " ", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value[:96].strip()


def run_capture_job(job_id: str, source: dict, output_dir: Path, event_name: str, args: SimpleNamespace) -> None:
    try:
        update_job(job_id, status="running", message="准备视频", percent=1)
        video_path = prepare_video_source(job_id, source, event_name)
        args.video = video_path
        args.source = serialize_source(source, video_path)

        is_live = source["type"] == "live"
        if is_live and args.profile == "auto":
            args.effective_profile = "program"
            args.profile_detection = {"note": "live streams default to program mode"}
        elif args.profile == "auto":
            update_job(job_id, message="自动判别发布会范式", percent=2)
            apply_auto_profile(args, Path(video_path))
        else:
            args.effective_profile = args.profile

        update_job(
            job_id,
            message=f"使用 {args.effective_profile} 策略抽帧",
            percent=4,
            effective_profile=args.effective_profile,
            profile_detection=args.profile_detection,
        )

        def on_progress(progress: dict) -> None:
            stage = progress.get("stage")
            if stage == "paused":
                message = "已暂停截图采集，音视频继续录制"
            elif stage == "writing" and is_live:
                message = "正在整理截图并保存录制文件"
            elif is_live:
                message = "正在采集截图，同时录制音视频"
            else:
                message = "采集中"
            capture_percent = progress.get("percent", 0)
            if source["type"] == "online":
                capture_percent = 25 + float(capture_percent or 0) * 0.74
            update_job(
                job_id,
                status="running",
                message=message,
                percent=capture_percent,
                keyframes=progress.get("keyframes", 0),
                sampled_frames=progress.get("sampled_frames", 0),
                last_timestamp_sec=progress.get("last_timestamp_sec", 0),
                duration_sec=progress.get("duration_sec", 0),
                latest_frame=progress.get("latest_frame"),
                capture_stage=progress.get("stage"),
                recording_status="recording" if is_live else None,
            )

        def live_control() -> str:
            with JOBS_LOCK:
                return str(JOBS.get(job_id, {}).get("control") or "running")

        if is_live:
            record_seconds = source.get("record_seconds")
            record_seconds = int(record_seconds) if record_seconds else None
            live_recording = output_dir / "live_recording.mp4"
            recorder = LiveRecorder(
                page_url=str(source.get("url") or video_path),
                stream_url=str(video_path),
                output_path=live_recording,
                duration_seconds=record_seconds,
            )
            try:
                keyframes, stats = extract_live_keyframes(
                    stream_url=str(video_path),
                    output_dir=output_dir,
                    profile=args.effective_profile,
                    sample_every=args.sample_every,
                    diff_threshold=args.diff_threshold,
                    min_gap=args.min_gap,
                    strong_threshold=args.strong_threshold,
                    strong_min_gap=args.strong_min_gap,
                    duration_seconds=record_seconds,
                    max_frames=args.max_frames,
                    max_width=args.max_width,
                    avoid_speaker_only=args.avoid_speaker_only,
                    progress_callback=on_progress,
                    control_callback=live_control,
                )
            finally:
                update_job(job_id, recording_status="stopping")
                saved_recording = recorder.stop()
            if saved_recording is None or not saved_recording.exists() or saved_recording.stat().st_size == 0:
                raise RuntimeError("直播没有生成可用的完整录制文件，已停止后续 ASR，避免输出残缺简报。")
            if saved_recording is not None and saved_recording.exists() and saved_recording.stat().st_size > 0:
                args.video = saved_recording.resolve()
                args.source["recording"] = str(saved_recording.resolve())
                update_job(job_id, recording_status="saved")
                try:
                    update_job(job_id, message="正在从录制文件复采截图", percent=95, recording_status="saved")
                    recorded_keyframes, recorded_stats = extract_keyframes(
                        video_path=saved_recording.resolve(),
                        output_dir=output_dir,
                        profile=args.effective_profile,
                        sample_every=args.sample_every,
                        diff_threshold=args.diff_threshold,
                        min_gap=args.min_gap,
                        strong_threshold=args.strong_threshold,
                        strong_min_gap=args.strong_min_gap,
                        max_frames=args.max_frames,
                        max_width=args.max_width,
                        avoid_speaker_only=args.avoid_speaker_only,
                    )
                    recorded_stats["live"] = True
                    recorded_stats["source_for_keyframes"] = "recording"
                    recorded_stats["live_sampling_keyframes"] = len(keyframes)
                    keyframes, stats = recorded_keyframes, recorded_stats
                except Exception as exc:
                    stats["source_for_keyframes"] = "live-stream"
                    stats["recording_resample_error"] = str(exc)
        else:
            keyframes, stats = extract_keyframes(
                video_path=Path(video_path),
                output_dir=output_dir,
                profile=args.effective_profile,
                sample_every=args.sample_every,
                diff_threshold=args.diff_threshold,
                min_gap=args.min_gap,
                strong_threshold=args.strong_threshold,
                strong_min_gap=args.strong_min_gap,
                max_frames=args.max_frames,
                max_width=args.max_width,
                avoid_speaker_only=args.avoid_speaker_only,
                progress_callback=on_progress,
            )
        write_manifest(output_dir, Path(args.video), keyframes, args, stats)
        write_markdown(output_dir, event_name, keyframes)
        write_html(output_dir, event_name, keyframes)
        update_job(
            job_id,
            status="done",
            message="完成",
            percent=100,
            keyframes=len(keyframes),
            recording_status="saved" if is_live else None,
            result_url=f"/result/{output_dir.name}",
        )
    except Exception as exc:
        update_job(job_id, status="error", message=str(exc))


def update_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        current = JOBS.setdefault(job_id, {"id": job_id})
        current.update(updates)


def update_brief_job(slug: str, **updates) -> None:
    with BRIEF_JOBS_LOCK:
        current = BRIEF_JOBS.setdefault(slug, {"slug": slug})
        current.update(updates)


def update_refine_job(slug: str, **updates) -> None:
    with REFINE_JOBS_LOCK:
        current = REFINE_JOBS.setdefault(slug, {"slug": slug})
        current.update(updates)


def update_lark_export_job(slug: str, **updates) -> None:
    with LARK_EXPORT_JOBS_LOCK:
        current = LARK_EXPORT_JOBS.setdefault(slug, {"slug": slug})
        current.update(updates)


def update_derivative_job(job_id: str, **updates) -> None:
    with DERIVATIVE_JOBS_LOCK:
        current = DERIVATIVE_JOBS.setdefault(job_id, {"id": job_id})
        current.update(updates)


def update_feedback_job(job_id: str, **updates) -> None:
    with FEEDBACK_JOBS_LOCK:
        current = FEEDBACK_JOBS.setdefault(job_id, {"id": job_id})
        current.update(updates)


def update_feedback_lark_export_job(job_id: str, **updates) -> None:
    with FEEDBACK_LARK_EXPORT_JOBS_LOCK:
        current = FEEDBACK_LARK_EXPORT_JOBS.setdefault(job_id, {"id": job_id})
        current.update(updates)


def update_review_video_job(job_id: str, **updates) -> None:
    with REVIEW_VIDEO_JOBS_LOCK:
        current = REVIEW_VIDEO_JOBS.setdefault(job_id, {"id": job_id})
        current.update(updates)


def update_content_review_lab_job(job_id: str, **updates) -> None:
    with CONTENT_REVIEW_LAB_JOBS_LOCK:
        current = CONTENT_REVIEW_LAB_JOBS.setdefault(job_id, {"id": job_id})
        current.update(updates)


def run_with_review_video_progress(
    job_id: str,
    *,
    message: str,
    percent: int,
    action,
    tick_seconds: int = 6,
):
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["result"] = action()
        except BaseException as exc:
            box["error"] = exc

    started_at = time.monotonic()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        elapsed = int(time.monotonic() - started_at)
        update_review_video_job(
            job_id,
            status="running",
            message=f"{message}，已用 {elapsed} 秒",
            percent=percent,
            elapsed_seconds=elapsed,
        )
        thread.join(tick_seconds)
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("result")


def run_with_content_review_lab_progress(
    job_id: str,
    *,
    message: str,
    percent: int,
    action,
    tick_seconds: int = 6,
):
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["result"] = action()
        except BaseException as exc:
            box["error"] = exc

    started_at = time.monotonic()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        elapsed = int(time.monotonic() - started_at)
        update_content_review_lab_job(
            job_id,
            status="running",
            message=f"{message}，已用 {elapsed} 秒",
            percent=percent,
            elapsed_seconds=elapsed,
        )
        thread.join(tick_seconds)
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("result")


def save_feedback_upload(upload, job_dir: Path, prefix: str) -> Path:
    original = Path(upload.filename or "upload").name
    suffix = Path(original).suffix.lower()
    allowed = {".xlsx", ".xlsm", ".docx", ".md", ".markdown", ".txt"}
    if suffix not in allowed:
        raise ValueError(f"暂不支持的文件类型：{suffix or original}")
    safe_name = f"{secure_filename(Path(original).stem) or prefix}{suffix}"
    path = job_dir / f"{prefix}-{safe_name}"
    upload.save(path)
    return path


def save_content_review_lab_upload(upload, job_dir: Path) -> Path:
    original = Path(upload.filename or "upload").name
    suffix = Path(original).suffix.lower()
    allowed = {".docx", ".md", ".markdown", ".txt"}
    if suffix not in allowed:
        raise ValueError(f"实验版暂不支持的稿件文件类型：{suffix or original}")
    safe_name = f"{secure_filename(Path(original).stem) or 'media-content'}{suffix}"
    path = job_dir / f"source-{safe_name}"
    upload.save(path)
    return path


def save_derivative_upload(upload, upload_dir: Path, prefix: str) -> Path:
    original = Path(upload.filename or "upload").name
    suffix = Path(original).suffix.lower()
    allowed = {".docx", ".md", ".markdown", ".pdf", ".txt"}
    if suffix not in allowed:
        raise ValueError(f"副产物暂不支持的材料文件类型：{suffix or original}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{secure_filename(Path(original).stem) or prefix}{suffix}"
    path = upload_dir / f"{prefix}-{safe_name}"
    upload.save(path)
    return path


def source_material_block(source_label: str, content: str) -> str:
    content = str(content or "").strip()
    if not content:
        return ""
    return f"## 来源：{source_label}\n\n{content}"


def resolve_content_review_lab_source(job_dir: Path) -> dict:
    source_type = request.form.get("source_type", "paste")
    if source_type == "paste":
        content = request.form.get("content_text", "").strip()
        if not content:
            raise ValueError("请粘贴待审核媒体正文。")
        return {"type": "paste", "content": content}
    if source_type == "file":
        upload = request.files.get("content_file")
        if not upload or not upload.filename:
            raise ValueError("请上传 MD、TXT 或 DOCX 稿件。")
        return {"type": "file", "path": save_content_review_lab_upload(upload, job_dir), "filename": upload.filename}
    if source_type == "lark":
        url = request.form.get("lark_url", "").strip()
        if not url:
            raise ValueError("请填写飞书文档链接。")
        return {"type": "lark", "url": url}
    if source_type == "online_video":
        url = request.form.get("video_url", "").strip()
        if not url:
            raise ValueError("请填写在线视频或字幕可解析链接。")
        return {"type": "online_video", "url": url}
    raise ValueError("暂不支持的实验输入来源。")


def resolve_review_video_source(job_dir: Path) -> dict:
    source_type = request.form.get("source_type", "local")
    if source_type == "online":
        url = request.form.get("online_url", "").strip()
        if not url:
            raise ValueError("请填写在线视频 URL。")
        return {"type": "online", "url": url}

    uploaded = request.files.get("video_file")
    if uploaded and uploaded.filename:
        original = Path(uploaded.filename).name
        safe_name = secure_filename(original) or "review-video.mp4"
        target = job_dir / f"upload-{safe_name}"
        uploaded.save(target)
        return {"type": "upload", "path": target.resolve(), "filename": original}

    video_path_value = request.form.get("video_path", "").strip()
    if not video_path_value:
        raise ValueError("请填写本机视频路径，或上传视频文件。")
    video_path = Path(video_path_value).expanduser().resolve()
    if not video_path.exists():
        raise ValueError(f"视频不存在：{video_path}")
    return {"type": "local", "path": video_path}


def serialize_review_source(source: dict) -> dict:
    result = dict(source)
    if "path" in result:
        result["path"] = str(result["path"])
    return result


def review_video_manifest(source: dict, video_title: str) -> dict:
    if source.get("type") == "online":
        return {
            "video": "",
            "event_name": video_title,
            "source": {
                "type": "online",
                "url": source.get("url", ""),
            },
        }
    return {
        "video": str(source.get("path") or ""),
        "event_name": video_title,
        "source": {
            "type": source.get("type") or "local",
            "path": str(source.get("path") or ""),
        },
    }


def review_video_playback_path(manifest: dict) -> Path | None:
    candidates = [
        str((manifest.get("source") or {}).get("prepared_video") or ""),
        str(manifest.get("video") or ""),
        str((manifest.get("source") or {}).get("path") or ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists() and path.is_file():
            return path
    return None


def content_review_lab_video_manifest(source: dict, content_title: str) -> dict:
    return {
        "video": "",
        "event_name": content_title,
        "source": {
            "type": "online",
            "url": str(source.get("url") or ""),
        },
    }


def run_content_review_lab_job(job_id: str, payload: dict) -> None:
    result_dir = CONTENT_REVIEW_LAB_ROOT / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = payload.get("source") or {}
        product_name = str(payload.get("product_name") or "").strip()
        media_name = str(payload.get("media_name") or "").strip()
        content_title = str(payload.get("content_title") or "").strip() or content_review_lab_default_title(source)
        review_reference_input = str(payload.get("review_reference") or "").strip()
        additional_instruction = str(payload.get("additional_instruction") or "").strip()

        source_text, source_meta = extract_content_review_lab_text(job_id, result_dir, source, content_title)
        if not source_text.strip():
            raise RuntimeError("没有读取到可审核的媒体正文。")
        (result_dir / "source_extracted.md").write_text(source_text.strip() + "\n", encoding="utf-8")
        review_reference, reference_meta = resolve_content_review_lab_reference(job_id, review_reference_input)
        if review_reference_input:
            (result_dir / "review_reference_input.md").write_text(review_reference_input.strip() + "\n", encoding="utf-8")
        if review_reference:
            (result_dir / "review_reference_resolved.md").write_text(review_reference.strip() + "\n", encoding="utf-8")

        update_content_review_lab_job(job_id, status="running", message="模型初筛风险表达", percent=48)

        def on_progress(progress: dict) -> None:
            generated_chars = int(progress.get("generated_chars") or 0)
            stage = str(progress.get("stage") or "媒体内容审核报告")
            base_percent = 50 if stage.startswith("风险初筛") else 76
            cap = 75 if stage.startswith("风险初筛") else 95
            update_content_review_lab_job(
                job_id,
                status="running",
                message=f"模型正在生成{stage}，已输出 {generated_chars} 字",
                percent=min(cap, base_percent + generated_chars // 280),
                generated_chars=generated_chars,
                preview=progress.get("preview", ""),
            )

        result, meta = generate_content_review_lab(
            base_dir=BASE_DIR,
            product_name=product_name,
            media_name=media_name,
            content_title=content_title,
            source_label=str(source_meta.get("label") or ""),
            source_text=source_text,
            review_reference=review_reference,
            additional_instruction=additional_instruction,
            progress_callback=on_progress,
        )
        meta.update(
            {
                "content_title": content_title,
                "source_type": source.get("type", ""),
                "source_meta": source_meta,
                "reference_meta": reference_meta,
            }
        )
        (result_dir / "content_review_lab.md").write_text(result.strip() + "\n", encoding="utf-8")
        (result_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        update_content_review_lab_job(
            job_id,
            status="done",
            message=f"完成：{meta.get('model', '模型')} / {source_meta.get('label', '正文')}",
            percent=100,
            result=result,
            preview=tail_preview(result, 1200),
            preview_url=f"/review-content-lab/{job_id}/preview",
            warnings=meta.get("warnings", []),
            source_meta=source_meta,
            reference_meta=reference_meta,
        )
    except Exception as exc:
        update_content_review_lab_job(job_id, status="error", message=str(exc), percent=100)


def extract_content_review_lab_text(job_id: str, result_dir: Path, source: dict, content_title: str) -> tuple[str, dict]:
    source_type = str(source.get("type") or "")
    if source_type == "paste":
        update_content_review_lab_job(job_id, status="running", message="读取粘贴正文", percent=16)
        return str(source.get("content") or ""), {"label": "粘贴正文"}
    if source_type == "file":
        update_content_review_lab_job(job_id, status="running", message="解析上传稿件", percent=18)
        path = Path(str(source.get("path") or ""))
        text = extract_feedback_source(path)
        return text, {"label": f"上传稿件：{source.get('filename') or path.name}", "path": str(path)}
    if source_type == "lark":
        update_content_review_lab_job(job_id, status="running", message="读取飞书文档", percent=20)
        text = run_with_content_review_lab_progress(
            job_id,
            message="读取飞书文档中",
            percent=24,
            action=lambda: fetch_lark_doc_markdown(BASE_DIR, str(source.get("url") or "")),
        )
        return str(text or ""), {"label": "飞书文档", "url": str(source.get("url") or "")}
    if source_type == "online_video":
        return extract_content_review_lab_video_text(job_id, result_dir, source, content_title)
    raise RuntimeError("无法识别待审内容来源。")


def resolve_content_review_lab_reference(job_id: str, raw_reference: str) -> tuple[str, dict]:
    raw_reference = raw_reference.strip()
    urls = extract_lark_reference_urls(raw_reference)
    instruction_text = strip_reference_urls(raw_reference, urls)
    if not urls:
        if raw_reference:
            update_content_review_lab_job(job_id, status="running", message="读取审核参考文本", percent=28)
        return raw_reference, {
            "input_kind": "text" if raw_reference else "empty",
            "document_count": 0,
            "instruction_chars": len(raw_reference),
        }

    update_content_review_lab_job(job_id, status="running", message=f"识别到 {len(urls)} 条飞书审核参考，准备读取", percent=30)
    document_blocks: list[str] = []
    for index, url in enumerate(urls, start=1):
        try:
            markdown = run_with_content_review_lab_progress(
                job_id,
                message=f"飞书 MCP 读取审核参考 {index}/{len(urls)}",
                percent=min(45, 30 + index * 4),
                action=lambda url=url: fetch_lark_doc_markdown(BASE_DIR, url),
            )
        except Exception as exc:
            raise RuntimeError(f"没有读到审核参考飞书文档 {index}/{len(urls)}：{url}。请检查分享权限或后台飞书 MCP 配置。") from exc
        markdown = str(markdown or "").strip()
        if not markdown:
            raise RuntimeError(f"审核参考飞书文档 {index}/{len(urls)} 没有返回可用正文：{url}")
        document_blocks.append(f"## 飞书审核参考 {index}\n- 链接：{url}\n\n{markdown}")

    blocks = ["# 审核参考输入"]
    if instruction_text:
        blocks.extend(["", "## 用户审核要求", "", instruction_text])
    blocks.extend(["", "## 飞书文档参考正文", "", "\n\n".join(document_blocks)])
    return "\n".join(blocks).strip(), {
        "input_kind": "mixed" if instruction_text else "lark_docs",
        "document_count": len(urls),
        "lark_urls": urls,
        "instruction_chars": len(instruction_text),
    }


def extract_lark_reference_urls(text: str) -> list[str]:
    pattern = re.compile(r"https?://[^\s<>()\[\]{}\"']+", flags=re.IGNORECASE)
    urls: list[str] = []
    for match in pattern.finditer(text or ""):
        url = match.group(0).rstrip(".,;:!?，。；：！？、")
        host_part = url.split("/", 3)[2].lower() if url.count("/") >= 2 else ""
        if any(domain in host_part for domain in ("feishu.cn", "larksuite.com", "feishu.com")) and url not in urls:
            urls.append(url)
    return urls


def strip_reference_urls(text: str, urls: list[str]) -> str:
    result = text
    for url in urls:
        result = re.sub(rf"\[([^\]]+)\]\({re.escape(url)}\)", r"\1", result)
        result = result.replace(url, "")
    result = re.sub(r"^[ \t]*(?:[-*]\s*)?$", "", result, flags=re.MULTILINE)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_content_review_lab_video_text(job_id: str, result_dir: Path, source: dict, content_title: str) -> tuple[str, dict]:
    manifest = content_review_lab_video_manifest(source, content_title)
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_content_review_lab_job(job_id, status="running", message="准备获取视频字幕或转写", percent=18)
    try:
        transcript, transcript_meta = run_with_content_review_lab_progress(
            job_id,
            message="抓取在线视频字幕中",
            percent=26,
            action=lambda: ensure_transcript(
                result_dir=result_dir,
                manifest=manifest,
                python_bin=PYTHON_BIN,
                base_dir=BASE_DIR,
            ),
        )
    except RuntimeError:
        update_content_review_lab_job(job_id, status="running", message="未抓到字幕，尝试下载视频用于 ASR", percent=28)
        try:
            video_path = run_with_content_review_lab_progress(
                job_id,
                message="yt-dlp 下载视频中",
                percent=34,
                action=lambda: download_with_ytdlp(
                    url=str(source.get("url") or ""),
                    output_dir=DOWNLOAD_ROOT / "content_review_lab" / job_id,
                    output_template="source.%(ext)s",
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                "实验版未能从该视频链接获取字幕或视频。请改用可解析视频链接，或粘贴/上传媒体导出的稿件、字幕或逐字稿。"
            ) from exc
        manifest["video"] = str(video_path)
        manifest["source"]["prepared_video"] = str(video_path)
        (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        transcript, transcript_meta = run_with_content_review_lab_progress(
            job_id,
            message="视频下载完成，ASR 转写中",
            percent=42,
            action=lambda: ensure_transcript(
                result_dir=result_dir,
                manifest=manifest,
                python_bin=PYTHON_BIN,
                base_dir=BASE_DIR,
            ),
        )
    write_transcript_meta(result_dir, transcript_meta)
    return str(transcript or ""), {
        "label": f"在线视频逐字稿：{transcript_meta.get('method', 'transcript')}",
        "url": str(source.get("url") or ""),
        "transcript_method": transcript_meta.get("method", ""),
    }


def content_review_lab_default_title(source: dict) -> str:
    source_type = str(source.get("type") or "")
    if source_type == "file":
        return Path(str(source.get("filename") or source.get("path") or "待审稿件")).stem
    if source_type == "lark":
        return "飞书待审媒体内容"
    if source_type == "online_video":
        return "视频待审媒体内容"
    return "待审媒体内容"


def run_review_video_job(job_id: str, payload: dict) -> None:
    result_dir = REVIEW_VIDEO_ROOT / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        source = payload.get("source") or {}
        product_name = str(payload.get("product_name") or "").strip()
        media_name = str(payload.get("media_name") or "").strip()
        video_title = str(payload.get("video_title") or "").strip() or review_video_default_title(source)
        additional_instruction = str(payload.get("additional_instruction") or "").strip()
        manifest = review_video_manifest(source, video_title)
        manifest["domain"] = {"requested": str(payload.get("domain_id") or "auto")}
        (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        update_review_video_job(job_id, status="running", message="准备抓取字幕或执行 ASR", percent=16)
        try:
            transcript, transcript_meta = run_with_review_video_progress(
                job_id,
                message="抓取在线视频字幕 / 本地视频 ASR 转写中",
                percent=24,
                action=lambda: ensure_transcript(
                    result_dir=result_dir,
                    manifest=manifest,
                    python_bin=PYTHON_BIN,
                    base_dir=BASE_DIR,
                ),
            )
        except RuntimeError as exc:
            if source.get("type") != "online":
                raise
            update_review_video_job(job_id, status="running", message="未抓到字幕，下载视频用于 ASR", percent=24)
            video_path = run_with_review_video_progress(
                job_id,
                message="yt-dlp 下载视频中",
                percent=32,
                action=lambda: download_with_ytdlp(
                    url=str(source.get("url") or ""),
                    output_dir=DOWNLOAD_ROOT / "review_video" / job_id,
                    output_template="source.%(ext)s",
                ),
            )
            manifest["video"] = str(video_path)
            manifest["source"]["prepared_video"] = str(video_path)
            (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            transcript, transcript_meta = run_with_review_video_progress(
                job_id,
                message="下载完成，ASR 转写中",
                percent=40,
                action=lambda: ensure_transcript(
                    result_dir=result_dir,
                    manifest=manifest,
                    python_bin=PYTHON_BIN,
                    base_dir=BASE_DIR,
                ),
            )
        if not transcript.strip():
            raise RuntimeError("没有获取到有效逐字稿。")
        manifest["domain"] = resolve_domain(manifest, transcript)
        (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        domain, _ = domain_from_manifest(manifest)
        write_transcript_meta(result_dir, transcript_meta)
        build_fact_ledger(
            result_dir=result_dir,
            manifest=manifest,
            transcript=transcript,
            transcript_meta=transcript_meta,
        )
        (result_dir / "transcript.md").write_text(transcript, encoding="utf-8")

        manifest = prepare_review_video_player(job_id, source, result_dir, manifest)
        update_review_video_job(job_id, status="running", message="模型精校查阅逐字稿", percent=42)

        def on_progress(progress: dict) -> None:
            generated_chars = int(progress.get("generated_chars") or 0)
            stage = str(progress.get("stage") or "评测视频分析")
            if stage.startswith("逐字稿精校"):
                base_percent, cap = 42, 60
            elif stage.startswith("逐字稿理解"):
                base_percent, cap = 62, 78
            else:
                base_percent, cap = 78, 95
            percent = min(cap, base_percent + generated_chars // 260)
            update_review_video_job(
                job_id,
                status="running",
                message=f"模型正在生成{stage}，已输出 {generated_chars} 字",
                percent=percent,
                generated_chars=generated_chars,
                preview=progress.get("preview", ""),
            )

        reader_segments, reader_meta = build_transcript_reader(
            base_dir=BASE_DIR,
            transcript_text=transcript,
            transcript_meta=transcript_meta,
            progress_callback=on_progress,
        )
        write_transcript_reader(result_dir, reader_segments, reader_meta)
        update_review_video_job(job_id, status="running", message="模型理解逐字稿", percent=62)

        result, meta = generate_review_video_analysis(
            base_dir=BASE_DIR,
            product_name=product_name,
            media_name=media_name,
            video_title=video_title,
            video_url=str(source.get("url") or ""),
            transcript_text=transcript,
            additional_instruction=additional_instruction,
            progress_callback=on_progress,
            domain=domain,
            work_dir=result_dir,
        )
        meta["product_name"] = product_name
        meta["media_name"] = media_name
        meta["video_title"] = video_title
        meta["title"] = video_title or product_name or f"评测视频分析-{job_id}"
        meta["transcript_reader"] = reader_meta
        meta["warnings"] = sorted(set([*(meta.get("warnings") or []), *(reader_meta.get("warnings") or [])]))
        (result_dir / "review_video_analysis.md").write_text(result.strip() + "\n", encoding="utf-8")
        (result_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        update_review_video_job(
            job_id,
            status="done",
            message=f"完成：{meta.get('model', '模型')} / {transcript_meta.get('method', 'transcript')}",
            percent=100,
            result=result,
            preview=tail_preview(result, 1200),
            preview_url=f"/review-video/{job_id}/preview",
            transcript_url=f"/review-video/{job_id}/transcript",
            transcript_method=transcript_meta.get("method", ""),
            warnings=meta.get("warnings", []),
        )
    except Exception as exc:
        update_review_video_job(job_id, status="error", message=str(exc), percent=100)


def review_video_default_title(source: dict) -> str:
    if source.get("type") == "online":
        url = str(source.get("url") or "")
        return slugify(url).strip("-")[:64] or "评测视频"
    if source.get("filename"):
        return Path(str(source["filename"])).stem
    path = source.get("path")
    return Path(str(path)).stem if path else "评测视频"


def prepare_review_video_player(job_id: str, source: dict, result_dir: Path, manifest: dict) -> dict:
    if review_video_playback_path(manifest) or source.get("type") != "online":
        return manifest
    update_review_video_job(job_id, status="running", message="下载查阅播放器视频", percent=38)
    video_path = run_with_review_video_progress(
        job_id,
        message="yt-dlp 下载查阅播放器视频中",
        percent=40,
        action=lambda: download_with_ytdlp(
            url=str(source.get("url") or ""),
            output_dir=DOWNLOAD_ROOT / "review_video" / job_id,
            output_template="source.%(ext)s",
        ),
    )
    manifest["video"] = str(video_path)
    manifest.setdefault("source", {})["prepared_video"] = str(video_path)
    (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_feedback_job(job_id: str, payload: dict) -> None:
    result_dir = FEEDBACK_ROOT / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        questionnaire_path = Path(str(payload.get("questionnaire_path") or ""))
        reference_paths = [Path(item) for item in payload.get("reference_paths", [])]
        product_name = str(payload.get("product_name") or "").strip()
        meeting_context = str(payload.get("meeting_context") or "").strip()
        additional_instruction = str(payload.get("additional_instruction") or "").strip()

        update_feedback_job(job_id, status="running", message="解析媒体问卷原始表单", percent=16)
        questionnaire_text = extract_feedback_source(questionnaire_path)
        if not questionnaire_text.strip():
            raise RuntimeError("没有从问卷文件中读取到有效文本。")
        (result_dir / "questionnaire_extracted.md").write_text(questionnaire_text, encoding="utf-8")

        update_feedback_job(job_id, status="running", message="计算媒体价格预期", percent=24)
        price_analysis = extract_price_analysis(questionnaire_path, questionnaire_text)
        price_analysis_markdown = price_analysis.to_markdown()
        (result_dir / "price_analysis.md").write_text(price_analysis_markdown + "\n", encoding="utf-8")

        reference_parts: list[str] = []
        if reference_paths:
            update_feedback_job(job_id, status="running", message="读取参考反馈文档", percent=28)
        for reference_path in reference_paths:
            text = extract_feedback_source(reference_path)
            if text.strip():
                reference_parts.append(f"## {reference_path.name}\n{text.strip()}")
        reference_feedback = "\n\n".join(reference_parts)
        if reference_feedback:
            (result_dir / "reference_feedback.md").write_text(reference_feedback, encoding="utf-8")

        update_feedback_job(job_id, status="running", message="模型生成媒体反馈", percent=36)

        def on_progress(progress: dict) -> None:
            generated_chars = int(progress.get("generated_chars") or 0)
            percent = min(94, 36 + generated_chars // 240)
            update_feedback_job(
                job_id,
                status="running",
                message=f"模型正在生成媒体反馈，已输出 {generated_chars} 字",
                percent=percent,
                generated_chars=generated_chars,
                preview=progress.get("preview", ""),
            )

        result, meta = generate_media_feedback(
            base_dir=BASE_DIR,
            product_name=product_name,
            meeting_context=meeting_context,
            questionnaire_text=questionnaire_text,
            price_analysis_markdown=price_analysis_markdown,
            reference_feedback=reference_feedback,
            additional_instruction=additional_instruction,
            progress_callback=on_progress,
        )
        meta["product_name"] = product_name
        meta["title"] = product_name or f"媒体反馈-{job_id}"
        meta["price_records"] = len(price_analysis.records)
        meta["price_computable_records"] = sum(1 for record in price_analysis.records if record.parsed_value is not None)
        (result_dir / "media_feedback.md").write_text(result.strip() + "\n", encoding="utf-8")
        (result_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        update_feedback_job(
            job_id,
            status="done",
            message=f"完成：{meta.get('model', '模型')}",
            percent=100,
            result=result,
            preview=tail_preview(result, 1200),
            preview_url=f"/feedback/{job_id}/preview",
            warnings=meta.get("warnings", []),
        )
    except Exception as exc:
        update_feedback_job(job_id, status="error", message=str(exc), percent=100)


def run_feedback_lark_export_job(job_id: str, title: str, source_path: Path) -> None:
    try:
        result_dir = FEEDBACK_ROOT / job_id
        update_feedback_lark_export_job(job_id, status="running", message="整理飞书 Markdown", percent=28)
        meta = export_markdown_to_lark(
            base_dir=BASE_DIR,
            result_dir=result_dir,
            source_path=source_path,
            display_name=title,
            metadata={"module": "feedback"},
        )
        update_feedback_lark_export_job(
            job_id,
            status="done",
            message="已誊写到飞书文档",
            percent=100,
            url=meta.get("url", ""),
            exported_at=meta.get("exported_at", ""),
        )
    except Exception as exc:
        update_feedback_lark_export_job(job_id, status="error", message=str(exc), percent=100)


def run_derivative_job(job_id: str, payload: dict) -> None:
    result_dir = DERIVATIVE_ROOT / job_id
    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        product_name = str(payload.get("productName") or "").strip()
        product_position = str(payload.get("productPosition") or "").strip()
        launch_info = str(payload.get("launchInfo") or "").strip()
        price_info = str(payload.get("priceInfo") or "").strip()
        asset_table = str(payload.get("assetTable") or "").strip()
        additional_instruction = str(payload.get("additionalInstruction") or "").strip()
        existing_asset = str(payload.get("existingAsset") or "").strip()
        existing_review = str(payload.get("existingReview") or "").strip()
        existing_press = str(payload.get("existingPress") or "").strip()
        existing_qa = str(payload.get("existingQa") or "").strip()
        requested_outputs = payload.get("requestedOutputs") or []
        if isinstance(requested_outputs, str):
            requested_outputs = [requested_outputs]
        requested_outputs = [str(item).strip() for item in requested_outputs if str(item).strip()]
        if not requested_outputs:
            raise RuntimeError("请至少选择一份本次要生成的材料。")

        update_derivative_job(job_id, status="running", message="读取原始材料", percent=12)
        whitepaper_sources: list[str] = []
        intro_sources: list[str] = []
        whitepaper_url = str(payload.get("whitepaperUrl") or "").strip()
        intro_url = str(payload.get("introUrl") or "").strip()
        whitepaper_path = str(payload.get("whitepaperUploadPath") or "").strip()
        intro_path = str(payload.get("introUploadPath") or "").strip()

        if whitepaper_url:
            update_derivative_job(job_id, status="running", message="飞书 MCP：读取市场白皮书", percent=18)
            if block := source_material_block("飞书文档", fetch_lark_doc_markdown(BASE_DIR, whitepaper_url)):
                whitepaper_sources.append(block)
        if intro_url:
            update_derivative_job(job_id, status="running", message="飞书 MCP：读取产品简介", percent=24)
            if block := source_material_block("飞书文档", fetch_lark_doc_markdown(BASE_DIR, intro_url)):
                intro_sources.append(block)
        if whitepaper_path:
            update_derivative_job(job_id, status="running", message="解析本地市场白皮书", percent=28)
            if block := source_material_block("本地文件", extract_feedback_source(Path(whitepaper_path))):
                whitepaper_sources.append(block)
        if intro_path:
            update_derivative_job(job_id, status="running", message="解析本地产品简介", percent=32)
            if block := source_material_block("本地文件", extract_feedback_source(Path(intro_path))):
                intro_sources.append(block)

        # Keep older browser drafts usable after the input UI moved to mixed sources.
        whitepaper_input = str(payload.get("whitepaperInput") or "").strip()
        intro_input = str(payload.get("introInput") or "").strip()
        if whitepaper_input:
            whitepaper_sources.append(source_material_block("本地文本", whitepaper_input))
        if intro_input:
            intro_sources.append(source_material_block("本地文本", intro_input))
        if not whitepaper_sources and not intro_sources:
            raise RuntimeError("请至少提供一份飞书文档或本地文件。")

        whitepaper = "\n\n".join(whitepaper_sources)
        intro = "\n\n".join(intro_sources)

        if whitepaper:
            (result_dir / "whitepaper.md").write_text(whitepaper, encoding="utf-8")
        if intro:
            (result_dir / "intro.md").write_text(intro, encoding="utf-8")
        update_derivative_job(job_id, status="running", message="模型生成副产物", percent=36)

        def on_progress(progress: dict) -> None:
            generated_chars = int(progress.get("generated_chars") or 0)
            stage = str(progress.get("stage") or "副产物")
            stage_base = {
                "卖点资产表": 36,
                "评测指南": 48,
                "新闻稿": 74,
                "自检与待确认": 88,
            }.get(stage, 36)
            stage_cap = {
                "卖点资产表": 47,
                "评测指南": 73,
                "新闻稿": 87,
                "自检与待确认": 95,
            }.get(stage, 92)
            percent = min(92, 36 + generated_chars // 260)
            percent = min(stage_cap, stage_base + generated_chars // 360)
            update_derivative_job(
                job_id,
                status="running",
                message=f"模型正在生成{stage}，已输出 {generated_chars} 字",
                percent=percent,
                generated_chars=generated_chars,
                preview=progress.get("preview", ""),
            )

        result, meta = generate_derivatives(
            base_dir=BASE_DIR,
            product_name=product_name,
            product_position=product_position,
            launch_info=launch_info,
            price_info=price_info,
            whitepaper=whitepaper,
            intro=intro,
            asset_table=asset_table,
            additional_instruction=additional_instruction,
            existing_asset=existing_asset,
            existing_review=existing_review,
            existing_press=existing_press,
            existing_qa=existing_qa,
            requested_outputs=requested_outputs,
            progress_callback=on_progress,
        )
        meta["product_name"] = product_name
        meta["title"] = product_name or f"副产物-{job_id}"
        meta["requested_outputs"] = requested_outputs
        (result_dir / "derivatives.md").write_text(result.strip() + "\n", encoding="utf-8")
        (result_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        update_derivative_job(
            job_id,
            status="done",
            message=f"完成：{meta.get('model', '模型')}",
            percent=100,
            result=result,
            sections=meta.get("sections", {}),
            preview=result[-1200:],
            preview_url=f"/derivatives/{job_id}/preview",
        )
    except Exception as exc:
        update_derivative_job(job_id, status="error", message=str(exc), percent=100)


def run_brief_job(slug: str, display_name: str) -> None:
    result_dir = RAW_ROOT / slug
    try:
        manifest = read_manifest(result_dir / "manifest.json")
        brief_manifest = dict(manifest)
        brief_manifest["event_name"] = display_name
        update_brief_job(slug, status="running", message="获取字幕或 ASR 转写中", percent=20)
        def on_asr_progress(progress: dict[str, Any]) -> None:
            asr_percent = int(progress.get("percent") or 0)
            update_brief_job(
                slug,
                status="running",
                message=str(progress.get("message") or "ASR 转写中"),
                percent=min(80, 20 + int(asr_percent * 0.6)),
                asr_stage=progress.get("stage"),
            )

        transcript, transcript_meta = ensure_transcript(
            result_dir=result_dir,
            manifest=manifest,
            python_bin=PYTHON_BIN,
            base_dir=BASE_DIR,
            progress_callback=on_asr_progress,
        )
        domain_resolution = resolve_domain(manifest, transcript)
        manifest["domain"] = domain_resolution
        (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        brief_manifest["domain"] = domain_resolution
        brief_manifest["transcript_meta"] = transcript_meta
        update_brief_job(slug, status="running", message="生成简报基础稿", percent=82)
        write_transcript_meta(result_dir, transcript_meta)
        build_fact_ledger(
            result_dir=result_dir,
            manifest=brief_manifest,
            transcript=transcript,
            transcript_meta=transcript_meta,
        )
        brief_path = write_brief_base(result_dir, brief_manifest, transcript, slug)
        build_coverage_report(
            result_dir=result_dir,
            transcript=transcript,
            base_text=brief_path.read_text(encoding="utf-8"),
        )
        update_brief_job(
            slug,
            status="done",
            message=f"完成：{transcript_meta.get('method', 'transcript')}",
            percent=100,
        )
    except Exception as exc:
        update_brief_job(slug, status="error", message=str(exc), percent=100)


def run_refine_job(slug: str, display_name: str, user_instruction: str = "", model_config: dict[str, object] | None = None, tier_override: str = "", format_override: str = "") -> None:
    try:
        model_label = str((model_config or {}).get("model") or (model_config or {}).get("name") or "自动路由模型")
        result_dir = RAW_ROOT / slug
        manifest = read_manifest(result_dir / "manifest.json")
        brief_manifest = dict(manifest)
        brief_manifest["event_name"] = display_name

        update_refine_job(slug, status="running", message="重新选图：读取已有转写", percent=10, preview="")
        transcript, transcript_meta = read_existing_transcript(result_dir)
        domain_resolution = resolve_domain(manifest, transcript)
        manifest["domain"] = domain_resolution
        (result_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        brief_manifest["domain"] = domain_resolution
        update_refine_job(slug, status="running", message="重新选图：刷新基础稿图文关系", percent=24)
        brief_manifest["transcript_meta"] = transcript_meta
        write_transcript_meta(result_dir, transcript_meta)
        build_fact_ledger(
            result_dir=result_dir,
            manifest=brief_manifest,
            transcript=transcript,
            transcript_meta=transcript_meta,
        )
        write_brief_base(result_dir, brief_manifest, transcript, slug)

        if user_instruction:
            update_refine_job(slug, status="running", message=f"{model_label} 精加工中：已注入本次补充要求", percent=35)
        else:
            update_refine_job(slug, status="running", message=f"{model_label} 精加工中", percent=35)
        def on_progress(progress: dict) -> None:
            generated_chars = int(progress.get("generated_chars") or 0)
            percent = min(88, 35 + generated_chars // 180)
            update_refine_job(
                slug,
                status="running",
                message=progress.get("message") or (f"{model_label} 生成中，已输出 {generated_chars} 字" + ("（含补充要求）" if user_instruction else "")),
                percent=percent,
                generated_chars=generated_chars,
                preview=progress.get("preview", ""),
            )

        _, meta = refine_brief(
            result_dir=result_dir,
            display_name=display_name,
            base_dir=BASE_DIR,
            user_instruction=user_instruction,
            tier_override=tier_override,
            format_override=format_override,
            progress_callback=on_progress,
            model_config=model_config,
        )
        refined_path = RAW_ROOT / slug / "brief_refined.md"
        current_path = RAW_ROOT / slug / "brief_current.md"
        if refined_path.exists():
            shutil.copyfile(refined_path, current_path)
        update_refine_job(
            slug,
            status="done",
            message=f"完成：{meta.get('model', '模型')}",
            percent=100,
        )
    except Exception as exc:
        update_refine_job(slug, status="error", message=str(exc), percent=100)


def run_lark_export_job(slug: str, display_name: str, variant: str) -> None:
    try:
        result_dir = RAW_ROOT / slug
        update_lark_export_job(slug, status="running", message="整理飞书 Markdown", percent=25)
        meta = export_brief_to_lark(
            base_dir=BASE_DIR,
            result_dir=result_dir,
            display_name=display_name,
            variant=variant,
        )
        update_lark_export_job(
            slug,
            status="done",
            message="已写入飞书文档",
            percent=100,
            url=meta.get("url", ""),
            exported_at=meta.get("exported_at", ""),
        )
    except Exception as exc:
        update_lark_export_job(slug, status="error", message=str(exc), percent=100)


def tail_preview(text: str, limit: int = 900) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def resolve_video_source() -> dict:
    source_type = request.form.get("source_type", "local")

    if source_type == "online":
        url = request.form.get("online_url", "").strip()
        if not url:
            raise ValueError("Please provide an online video URL.")
        title = request.form.get("source_title", "").strip()
        return {
            "type": "online",
            "url": url,
            "default_event_name": title or slugify(url).strip("-")[:64] or "online-video",
        }

    if source_type == "live":
        url = request.form.get("live_url", "").strip()
        if not url:
            raise ValueError("Please provide a live stream URL.")
        title = request.form.get("source_title", "").strip()
        return {
            "type": "live",
            "url": url,
            "record_seconds": parse_optional_int("record_seconds"),
            "default_event_name": title or "live-capture",
        }

    uploaded = request.files.get("video_file")
    if uploaded and uploaded.filename:
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        filename = secure_filename(uploaded.filename) or "upload.mp4"
        video_path = UPLOAD_ROOT / filename
        uploaded.save(video_path)
        title = request.form.get("source_title", "").strip()
        return {
            "type": "upload",
            "path": video_path.resolve(),
            "default_event_name": title or video_path.stem,
        }

    video_path_value = request.form.get("video_path", "").strip()
    if not video_path_value:
        raise ValueError("Please provide a local video path or upload a video file.")
    video_path = Path(video_path_value).expanduser().resolve()
    if not video_path.exists():
        raise ValueError(f"Video does not exist: {video_path}")
    return {
        "type": "local",
        "path": video_path,
        "default_event_name": request.form.get("source_title", "").strip() or video_path.stem,
    }


def resolve_source_title(source_type: str) -> str:
    if source_type == "local":
        uploaded = request.files.get("video_file")
        if uploaded and uploaded.filename:
            return Path(uploaded.filename).stem
        video_path_value = request.form.get("video_path", "").strip()
        if not video_path_value:
            raise ValueError("请先填写本机视频路径或选择上传文件。")
        return Path(video_path_value).expanduser().stem

    if source_type == "online":
        url = request.form.get("online_url", "").strip()
        if not url:
            raise ValueError("请先填写在线视频 URL。")
        return get_ytdlp_title(url)

    if source_type == "live":
        url = request.form.get("live_url", "").strip()
        if not url:
            raise ValueError("请先填写直播流 URL。")
        return get_ytdlp_title(url)

    raise ValueError(f"Unsupported source type: {source_type}")


def prepare_video_source(job_id: str, source: dict, event_name: str) -> Path:
    source_type = source["type"]
    if source_type in {"local", "upload"}:
        return Path(source["path"]).resolve()

    if source_type == "online":
        update_job(job_id, message="下载在线视频", percent=1)

        def on_download_progress(progress: dict) -> None:
            download_percent = float(progress.get("percent") or 0)
            detail = " · ".join(
                item
                for item in (
                    str(progress.get("downloaded") or ""),
                    str(progress.get("speed") or ""),
                    f"剩余 {progress.get('eta')}" if progress.get("eta") else "",
                )
                if item
            )
            update_job(
                job_id,
                message=f"下载在线视频 {download_percent:.1f}%{f' · {detail}' if detail else ''}",
                percent=1 + download_percent * 0.23,
                download_percent=download_percent,
                download_speed=progress.get("speed"),
                download_eta=progress.get("eta"),
                downloaded=progress.get("downloaded"),
            )

        return download_with_ytdlp(
            url=source["url"],
            output_dir=DOWNLOAD_ROOT / slugify(event_name),
            output_template="source.%(ext)s",
            progress_callback=on_download_progress,
        )

    if source_type == "live":
        record_seconds = source.get("record_seconds")
        record_seconds = max(10, int(record_seconds)) if record_seconds else None
        update_job(job_id, message="解析直播流", percent=1, record_seconds=record_seconds)
        return resolve_stream_url(source["url"])

    raise ValueError(f"Unsupported source type: {source_type}")


def serialize_source(source: dict, prepared_video: Path | str) -> dict:
    result = {
        "type": source.get("type"),
        "prepared_video": str(prepared_video),
    }
    for key in ("url", "record_seconds"):
        if key in source and source[key] is not None:
            result[key] = source[key]
    if "path" in source:
        result["path"] = str(source["path"])
    return result


class LiveRecorder:
    """看护式直播录制：ffmpeg 掉线后重新解析流地址、续录新分段，结束时拼接。

    直播流地址普遍带过期 token（30-60 分钟），单个 ffmpeg 进程会在 token
    失效或网络抖动时静默退出，导致录制中途截断而采图仍在继续。
    分段用 .ts 容器，进程被强杀也不会损坏已写入内容。
    """

    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, page_url: str, stream_url: str, output_path: Path, duration_seconds: int | None):
        self.page_url = page_url
        self.stream_url = stream_url
        self.output_path = output_path
        self.duration_seconds = duration_seconds
        self.segments_dir = output_path.parent / "live_segments"
        self.segments: list[Path] = []
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._started_at = time.monotonic()
        if self.segments_dir.exists():
            shutil.rmtree(self.segments_dir)
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        self._thread = threading.Thread(target=self._supervise, daemon=True)
        self._thread.start()

    def _remaining_seconds(self) -> float | None:
        if not self.duration_seconds:
            return None
        return self.duration_seconds - (time.monotonic() - self._started_at)

    def _spawn(self, segment_path: Path) -> subprocess.Popen:
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y"]
        if self.stream_url.startswith(("http://", "https://")):
            command += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "30"]
        command += [
            "-fflags", "+genpts+discardcorrupt",
            "-err_detect", "ignore_err",
            "-i", self.stream_url,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "mpegts",
            str(segment_path),
        ]
        remaining = self._remaining_seconds()
        if remaining is not None:
            command[-3:-3] = ["-t", str(max(10, int(remaining)))]
        return subprocess.Popen(command, cwd=BASE_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _supervise(self) -> None:
        index = 0
        failures = 0
        while not self._stop.is_set():
            remaining = self._remaining_seconds()
            if remaining is not None and remaining <= 5:
                break
            index += 1
            segment = self.segments_dir / f"part_{index:03d}.ts"
            with self._proc_lock:
                if self._stop.is_set():
                    break
                self._proc = self._spawn(segment)
            self._proc.wait()
            if segment.exists() and segment.stat().st_size > 64_000:
                self.segments.append(segment)
                failures = 0
            else:
                segment.unlink(missing_ok=True)
                failures += 1
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    break
            if self._stop.is_set():
                break
            # 进程退出但任务未结束：流地址可能已过期，重新解析后续录
            self._stop.wait(3)
            try:
                self.stream_url = resolve_stream_url(self.page_url)
            except Exception:
                failures += 1
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    break
                self._stop.wait(15)

    def stop(self) -> Path | None:
        self._stop.set()
        with self._proc_lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            self._write_recording_meta(self.segments, "stop-timeout", False, "recorder thread did not stop within 30 seconds")
            raise RuntimeError(f"直播录制进程未能安全停止；分段已保留在 {self.segments_dir}。")
        return self._merge_segments()

    def _merge_segments(self) -> Path | None:
        segments = [item for item in self.segments if item.exists() and item.stat().st_size > 0]
        if not segments:
            self._write_recording_meta([], "no-segments", False, "no usable recording segments")
            return None
        list_path = self.segments_dir / "segments.txt"
        list_path.write_text(
            "".join(f"file '{item.as_posix()}'\n" for item in segments),
            encoding="utf-8",
        )
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(self.output_path),
        ]
        result = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode == 0 and self.output_path.exists() and self.output_path.stat().st_size > 0:
            self._write_recording_meta(segments, "concat-copy", True)
            shutil.rmtree(self.segments_dir, ignore_errors=True)
            return self.output_path

        # Timestamp/codec discontinuities can make stream-copy concat fail. A second
        # pass normalizes every segment instead of silently discarding all but one.
        transcode = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(self.output_path),
        ]
        retry = subprocess.run(transcode, cwd=BASE_DIR, capture_output=True, text=True)
        if retry.returncode == 0 and self.output_path.exists() and self.output_path.stat().st_size > 0:
            self._write_recording_meta(segments, "concat-transcode", True)
            shutil.rmtree(self.segments_dir, ignore_errors=True)
            return self.output_path

        self._write_recording_meta(
            segments,
            "failed",
            False,
            error=(retry.stderr or result.stderr or "ffmpeg concat failed")[-4000:],
        )
        raise RuntimeError(
            f"直播录制的 {len(segments)} 个分段均已保留在 {self.segments_dir}，但无法合并；"
            "为避免用不完整视频继续 ASR，任务已停止。"
        )

    def _write_recording_meta(
        self,
        segments: list[Path],
        strategy: str,
        complete: bool,
        error: str = "",
    ) -> None:
        payload = {
            "complete": complete,
            "merge_strategy": strategy,
            "wall_clock_seconds": round(time.monotonic() - self._started_at, 3),
            "requested_seconds": self.duration_seconds,
            "segment_count": len(segments),
            "segments": [
                {"name": item.name, "bytes": item.stat().st_size}
                for item in segments if item.exists()
            ],
            "error": error,
        }
        (self.output_path.parent / "recording_meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def download_with_ytdlp(
    url: str,
    output_dir: Path,
    output_template: str,
    extra_args: list[str] | None = None,
    progress_callback=None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*"):
        if existing.is_file() and existing.suffix != ".part":
            existing.unlink()

    selectors = [
        "bv*[protocol^=http][vcodec*=avc1]+ba[protocol^=http]/b[protocol^=http][vcodec*=avc1]",
        "bv*[protocol^=http]+ba[protocol^=http]/b[protocol^=http]/best[protocol^=http]/best",
        "best",
    ]
    errors: list[str] = []

    downloaded = False
    for site_args in ytdlp_site_arg_variants(url):
        for selector in selectors:
            command = ytdlp_command(
                *site_args,
                "--continue",
                "--no-playlist",
                "--check-formats",
                "--newline",
                "--no-colors",
                "--progress-template",
                "download:__MOTOOLBOX_PROGRESS__%(progress._percent_str)s|%(progress._speed_str)s|%(progress._eta_str)s|%(progress._downloaded_bytes_str)s",
                "-f",
                selector,
                "--merge-output-format",
                "mp4",
                "--ffmpeg-location",
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-o",
                str(output_dir / output_template),
                url,
            )
            if extra_args:
                command[-1:-1] = extra_args

            returncode, output = run_ytdlp_download(command, progress_callback)
            if returncode == 0:
                downloaded = True
                break
            errors.append((output or "yt-dlp failed").strip())
            if "HTTP Error 412" in output or "cookies" in output.lower():
                break
        if downloaded:
            break
    if not downloaded:
        raise RuntimeError("\n\n".join(errors[-2:]))

    candidates = sorted(
        [item for item in output_dir.iterdir() if item.is_file() and item.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("yt-dlp finished, but no video file was found.")
    return candidates[0].resolve()


def run_ytdlp_download(command: list[str], progress_callback=None) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        output_lines.append(line)
        if len(output_lines) > 200:
            output_lines.pop(0)
        if progress_callback and line.startswith("__MOTOOLBOX_PROGRESS__"):
            fields = line.removeprefix("__MOTOOLBOX_PROGRESS__").split("|", 3)
            while len(fields) < 4:
                fields.append("")
            percent_text, speed, eta, downloaded = (field.strip() for field in fields)
            match = re.search(r"[\d.]+", percent_text)
            if match:
                progress_callback(
                    {
                        "percent": min(100.0, float(match.group(0))),
                        "speed": "" if speed == "N/A" else speed,
                        "eta": "" if eta == "N/A" else eta,
                        "downloaded": "" if downloaded == "N/A" else downloaded,
                    }
                )
    return process.wait(), "\n".join(output_lines)


def resolve_stream_url(url: str) -> str:
    selectors = [
        "b[protocol^=http][vcodec*=avc1]/b[protocol^=http][vcodec*=h264]",
        "b[protocol^=http]/best[protocol^=http]/best",
    ]
    errors: list[str] = []
    for site_args in ytdlp_site_arg_variants(url):
        for selector in selectors:
            command = ytdlp_command(
                *site_args,
                "-g",
                "--no-playlist",
                "-f",
                selector,
                url,
            )
            result = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True)
            if result.returncode == 0:
                urls = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith(("http://", "https://"))]
                if urls:
                    return urls[0]
            errors.append((result.stderr or result.stdout or "yt-dlp could not resolve the live stream").strip())
    raise RuntimeError(errors[-1] if errors else "yt-dlp did not return a playable stream URL.")


def get_ytdlp_title(url: str) -> str:
    errors: list[str] = []
    for site_args in ytdlp_site_arg_variants(url):
        command = ytdlp_command(
            *site_args,
            "--dump-json",
            "--skip-download",
            "--no-playlist",
            url,
        )
        result = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode == 0:
            first_line = next((line for line in result.stdout.splitlines() if line.strip().startswith("{")), "")
            if first_line:
                metadata = json.loads(first_line)
                title = str(metadata.get("title") or metadata.get("fulltitle") or "").strip()
                if title:
                    return title
        errors.append((result.stderr or result.stdout or "yt-dlp title lookup failed").strip())
    raise RuntimeError(errors[-1] if errors else "yt-dlp did not return metadata.")


def build_capture_args(event_name: str, profile: str) -> SimpleNamespace:
    defaults = CAPTURE_PROFILES[profile]
    args = SimpleNamespace(
        video=None,
        event_name=event_name,
        output_root=RAW_ROOT,
        profile=profile,
        effective_profile=profile,
        profile_detection={},
        sample_every=parse_float("sample_every", defaults.sample_every),
        diff_threshold=parse_float("diff_threshold", defaults.diff_threshold),
        min_gap=parse_float("min_gap", defaults.min_gap),
        strong_threshold=parse_float("strong_threshold", defaults.strong_threshold),
        strong_min_gap=parse_float("strong_min_gap", defaults.strong_min_gap),
        avoid_speaker_only=request.form.get("avoid_speaker_only") == "on",
        max_frames=parse_int("max_frames", 0),
        max_width=parse_int("max_width", 1600),
        domain_requested=request.form.get("domain_id", "auto").strip(),
        task_type=request.form.get("task_type", "business_review").strip(),
    )
    return args


def parse_float(name: str, default: float) -> float:
    value = request.form.get(name, "").strip()
    if not value:
        return default
    return float(value)


def parse_int(name: str, default: int) -> int:
    value = request.form.get(name, "").strip()
    if not value:
        return default
    return int(value)


def parse_optional_int(name: str) -> int | None:
    value = request.form.get(name, "").strip()
    if not value:
        return None
    return int(value)


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_transcript_file(result_dir: Path) -> Path | None:
    transcript_dir = result_dir / "transcript"
    for name in ("source.srt", "source.vtt", "asr.srt", "asr.txt"):
        path = transcript_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def markdown_file_for_variant(result_dir: Path, variant: str) -> tuple[Path, str]:
    if variant == "refined":
        current_path = result_dir / "brief_current.md"
        if current_path.exists():
            return current_path, "brief_current.md"
        refined_path = result_dir / "brief_refined.md"
        if refined_path.exists():
            shutil.copyfile(refined_path, current_path)
            return current_path, "brief_current.md"
    return result_dir / "brief_base.md", "brief_base.md"


def edit_preview_markdown(
    slug: str,
    variant: str,
    action: str,
    block_index: int,
    image_path: str,
    source_block_index: int | None = None,
) -> str:
    if not re.match(r"^frames/[^/]+\.jpg$", image_path):
        raise ValueError("Invalid image path.")
    result_dir = RAW_ROOT / slug
    path, filename = markdown_file_for_variant(result_dir, variant)
    if not path.exists():
        raise ValueError("Markdown file does not exist.")
    content = path.read_text(encoding="utf-8")
    blocks = render_markdown_preview(content, slug)
    if block_index < 0 or block_index >= len(blocks):
        raise ValueError("Invalid target block.")
    lines = content.splitlines()
    block = blocks[block_index]
    image_line = f"  ![]({image_path})"

    if action == "add_image":
        insert_at = image_insert_index(lines, block.end_line + 1)
        if image_line.strip() not in [line.strip() for line in lines[max(0, block.start_line - 1) : min(len(lines), insert_at + 1)]]:
            lines.insert(insert_at, image_line)
    elif action == "remove_image":
        remove_image_line(lines, f"![]({image_path})", blocks, source_block_index)
    elif action == "move_image":
        target = f"![]({image_path})"
        insert_at = image_insert_index(lines, block.end_line + 1)
        removed_at = remove_image_line(lines, target, blocks, source_block_index)
        if removed_at is None:
            raise ValueError("Image is not in the document.")
        if removed_at < insert_at:
            insert_at -= 1
        insert_at = image_insert_index(lines, min(insert_at, len(lines)))
        lines.insert(insert_at, image_line)
    else:
        raise ValueError("Unsupported edit action.")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return filename


def image_insert_index(lines: list[str], index: int) -> int:
    index = min(max(index, 0), len(lines))
    while index < len(lines) and lines[index].strip().startswith("![]("):
        index += 1
    return index


def remove_image_line(lines: list[str], target: str, blocks: list, source_block_index: int | None = None) -> int | None:
    if source_block_index is not None and 0 <= source_block_index < len(blocks):
        block = blocks[source_block_index]
        start = max(0, block.start_line)
        end = min(len(lines) - 1, block.end_line)
        for index in range(start, end + 1):
            if lines[index].strip() == target:
                del lines[index]
                return index
    for index, line in enumerate(lines):
        if line.strip() == target:
            del lines[index]
            return index
    return None


def event_display_name(manifest: dict, slug: str) -> str:
    event_name = str(manifest.get("event_name") or "").strip()
    if event_name and not looks_like_auto_slug(event_name, slug):
        return event_name

    source = manifest.get("source") or {}
    for value in (source.get("title"), source.get("url"), manifest.get("video")):
        title = title_from_source_value(str(value or ""))
        if title:
            return title
    return event_name or slug


def looks_like_auto_slug(value: str, slug: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized == slug.lower():
        return True
    return bool(re.search(r"-(auto|lecture|program)$", normalized))


def title_from_source_value(value: str) -> str:
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return ""
    title = Path(value).expanduser().stem if "/" in value or "." in Path(value).name else value
    title = re.sub(r"\s+-\s+\d{2,4}\s+-\s+", " - ", title).strip()
    parts = [part.strip() for part in title.split(" - ") if part.strip()]
    if len(parts) >= 2 and normalize_title(parts[0]) == normalize_title(parts[-1]):
        title = parts[0]
    return title


def normalize_title(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).lower()


def cors_json(payload: dict, status: int = 200):
    response = jsonify(payload)
    response.status_code = status
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def list_jobs() -> list[dict]:
    jobs = []
    if not RAW_ROOT.exists():
        return jobs

    for manifest_path in sorted(RAW_ROOT.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        manifest = read_manifest(manifest_path)
        stats = manifest.get("stats", {})
        keyframes = manifest.get("keyframes", [])
        jobs.append(
            {
                "slug": manifest_path.parent.name,
                "event_name": event_display_name(manifest, manifest_path.parent.name),
                "profile": manifest.get("effective_profile") or manifest.get("profile") or "-",
                "count": len(keyframes),
                "duration": format_timestamp(float(stats.get("duration_sec") or 0)),
            }
        )
    return jobs


app = create_app()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--motoolbox-ytdlp":
        import yt_dlp

        yt_dlp.main(sys.argv[2:])
    else:
        app.run(host="127.0.0.1", port=int(os.environ.get("MOTOOLBOX_PORT", "5058")), debug=False, use_reloader=False)
