from __future__ import annotations

import json
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import imageio_ffmpeg
import numpy as np

from pressconf.asr_hotword_loader import asr_hotword_prompt, build_asr_hotwords
from pressconf.runtime import ytdlp_command, ytdlp_site_arg_variants
from pressconf.asr_segments import VERSION as ASR_PIPELINE_VERSION, issues as asr_issues, transcribe_chunks


FASTER_WHISPER_MODEL_ALIASES = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
LOCAL_ASR_RESCUE_MODELS = ("small", "base")
HF_RATE_LIMIT_COOLDOWN_SEC = 15 * 60
FASTER_WHISPER_MAX_CONTEXT_TERMS = 24
MAX_ASR_REPETITION_RATIO = 0.08
DEFAULT_FAST_ASR_MODEL = "large-v3-turbo"
DEFAULT_MLX_ASR_MODEL = "mlx-community/whisper-large-v3-turbo-q4"
MLX_MODEL_ALIASES = {
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": DEFAULT_MLX_ASR_MODEL,
    "turbo": DEFAULT_MLX_ASR_MODEL,
    "large-v3-turbo-fp16": "mlx-community/whisper-large-v3-turbo",
}


def ensure_transcript(
    *,
    result_dir: Path,
    manifest: dict[str, Any],
    python_bin: Path,
    base_dir: Path,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    transcript_dir = result_dir / "transcript"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    existing = first_existing(transcript_dir, ["source.srt", "source.vtt", "asr.srt", "asr.txt"])
    if existing and asr_cache_usable(existing, manifest):
        content = existing.read_text(encoding="utf-8", errors="ignore")
        meta = transcript_meta("cached", str(existing.relative_to(result_dir)), content)
        return content, attach_quality_meta(transcript_dir, meta)

    source = manifest.get("source") or {}
    source_type = source.get("type") or infer_source_type(manifest)

    if source_type == "online" and source.get("url"):
        subtitle = download_subtitle(
            url=str(source["url"]),
            output_dir=transcript_dir,
            python_bin=python_bin,
            base_dir=base_dir,
        )
        if subtitle:
            content = subtitle.read_text(encoding="utf-8", errors="ignore")
            meta = transcript_meta("yt-dlp-subtitle", str(subtitle.relative_to(result_dir)), content)
            return content, attach_quality_meta(transcript_dir, meta)

    video_path = local_video_path(manifest)
    if not video_path:
        raise RuntimeError("没有找到可用于 ASR 的本地视频文件。请重新运行一次采图任务，或确认 manifest 中的视频路径仍然存在。")

    hotword_context = build_asr_hotwords(manifest)
    asr_path = run_asr(
        video_path=video_path,
        output_dir=transcript_dir,
        python_bin=python_bin,
        base_dir=base_dir,
        hotword_context=hotword_context,
        progress_callback=progress_callback,
    )
    content = asr_path.read_text(encoding="utf-8", errors="ignore")
    meta = transcript_meta("asr", str(asr_path.relative_to(result_dir)), content)
    if hotword_context.get("terms"):
        meta["hotwords"] = {
            "term_count": hotword_context.get("term_count", 0),
            "available_term_count": hotword_context.get("available_term_count", 0),
            "brands": hotword_context.get("brands", []),
            "categories": hotword_context.get("categories", []),
            "files": hotword_context.get("files", []),
            "terms": hotword_context.get("terms", [])[:120],
        }
    return content, attach_quality_meta(transcript_dir, meta)


def asr_cache_usable(path: Path, manifest: dict[str, Any]) -> bool:
    if path.name.startswith("source."):
        return True
    try:
        quality = json.loads((path.parent / "quality.json").read_text())
    except (OSError, ValueError):
        quality = {}
    duration = float(quality.get("duration") or (manifest.get("stats") or {}).get("duration_sec") or 0)
    if quality.get("unresolved_intervals"):
        return False
    return duration <= 180 or quality.get("asr_pipeline_version") == ASR_PIPELINE_VERSION


def transcript_meta(method: str, path: str, content: str) -> dict[str, Any]:
    language = detect_transcript_language(content)
    result = {
        "method": method,
        "path": path,
        "language": language["language"],
        "language_label": language["label"],
        "language_confidence": language["confidence"],
        "output_language": "zh-CN",
        "language_note": language["note"],
    }
    return result


def attach_quality_meta(transcript_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    quality_path = transcript_dir / "quality.json"
    if not quality_path.exists():
        return meta
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return meta
    result = dict(meta)
    result["quality"] = quality
    return result


def detect_transcript_language(content: str) -> dict[str, Any]:
    text = strip_timing_noise(content)
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", text))
    latin_chars = sum(len(item) for item in re.findall(r"[A-Za-z]+", text))
    signal = cjk_chars + latin_chars
    if signal <= 20:
        return {
            "language": "unknown",
            "label": "语言未判定",
            "confidence": 0.0,
            "note": "转写文本过短，无法可靠判断语言。",
        }

    cjk_ratio = cjk_chars / max(signal, 1)
    if cjk_chars >= 8 and latin_words >= 3:
        language = "mixed"
        label = "中英混合"
        confidence = 0.78
        note = "转写为中英混合：后续生成会用中文输出，保留英文产品名和技术名。"
    elif cjk_ratio >= 0.25:
        language = "zh"
        label = "中文"
        confidence = min(0.98, 0.55 + cjk_ratio)
        note = "转写主要为中文，按中文发布会处理。"
    elif latin_words >= 12 and cjk_chars <= max(10, latin_words * 0.2):
        language = "en"
        label = "英文"
        confidence = min(0.96, 0.55 + latin_words / max(latin_words + cjk_chars, 1))
        note = "转写主要为英文：后续生成会理解英文内容，输出中文简报，并保留产品/技术专有名词原文。"
    else:
        language = "mixed"
        label = "中英混合"
        confidence = 0.68
        note = "转写为中英混合：后续生成会用中文输出，保留英文产品名和技术名。"
    return {"language": language, "label": label, "confidence": round(confidence, 2), "note": note}


def strip_timing_noise(content: str) -> str:
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or "-->" in stripped:
            continue
        lines.append(re.sub(r"<[^>]+>", " ", stripped))
    return " ".join(lines)


def first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def infer_source_type(manifest: dict[str, Any]) -> str:
    stats = manifest.get("stats") or {}
    if stats.get("live"):
        return "live"
    video = str(manifest.get("video") or "")
    if "/pressconf/downloads/" in video:
        return "online"
    return "local"


def local_video_path(manifest: dict[str, Any]) -> Path | None:
    video = manifest.get("video")
    if not video:
        return None
    path = Path(str(video)).expanduser()
    return path if path.exists() else None


def download_subtitle(url: str, output_dir: Path, python_bin: Path, base_dir: Path) -> Path | None:
    for existing in output_dir.glob("subtitle.*"):
        if existing.is_file():
            existing.unlink()

    subtitles: list[Path] = []
    # Prefer human/official subtitles. Automatic captions are a fallback, not a
    # peer candidate whose newer mtime can accidentally win.
    for subtitle_flag in ("--write-subs", "--write-auto-subs"):
        for site_args in ytdlp_site_arg_variants(url):
            command = ytdlp_command(
                *site_args,
                "--no-playlist",
                "--skip-download",
                subtitle_flag,
                "--sub-langs",
                "zh-Hans,zh-Hant,zh-CN,zh-TW,zh,en",
                "--sub-format",
                "srt/vtt/best",
                "-o",
                str(output_dir / "subtitle.%(ext)s"),
                url,
            )
            subprocess.run(command, cwd=base_dir, capture_output=True, text=True)
            subtitles = sorted(
                [
                    item
                    for item in output_dir.glob("subtitle.*")
                    if item.suffix.lower() in {".srt", ".vtt"} and item.stat().st_size > 0
                ],
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if subtitles:
                break
        if subtitles:
            break
    if not subtitles:
        return None

    subtitle = subtitles[0]
    target = output_dir / f"source{subtitle.suffix.lower()}"
    subtitle.replace(target)
    (output_dir / "quality.json").write_text(json.dumps({
        "engine": "yt-dlp-subtitle",
        "subtitle_kind": "automatic" if subtitle_flag == "--write-auto-subs" else "human-or-official",
        "confidence_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def run_asr(
    video_path: Path,
    output_dir: Path,
    python_bin: Path,
    base_dir: Path,
    hotword_context: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    pipeline_started = time.time()
    audio_path = output_dir / "audio.wav"
    emit_asr_progress(progress_callback, stage="extract-audio", message="正在提取 16kHz 单声道音频", percent=5)
    extract_audio(video_path, audio_path, base_dir)
    emit_asr_progress(progress_callback, stage="audio-ready", message="音频准备完成，正在启动 ASR", percent=12)

    requested_engine = os.environ.get("MOTOOLBOX_ASR_ENGINE", "mlx").strip().lower() or "mlx"
    failures: list[dict[str, str]] = []
    target: Path | None = None
    engine_used = ""
    if requested_engine in {"mlx", "mlx-only", "auto"} and mlx_asr_available():
        try:
            emit_asr_progress(
                progress_callback,
                stage="mlx",
                message="MLX Whisper 正在使用 Apple GPU 转写（首次运行会下载约 464 MB 模型）",
                percent=18,
            )
            target = run_mlx_whisper(audio_path, output_dir, hotword_context or {}, progress_callback)
            engine_used = "mlx-whisper"
        except Exception as exc:
            failures.append({"engine": "mlx-whisper", "error": str(exc)})
            write_asr_attempts(output_dir, requested_engine, failures)

    if target is None and requested_engine in {"mlx", "auto", "faster-whisper", "faster_whisper"} and module_available("faster_whisper"):
        emit_asr_progress(progress_callback, stage="fallback", message="MLX 不可用，改用 faster-whisper 本地兜底", percent=20)
        try:
            target = run_faster_whisper(audio_path, output_dir, hotword_context or {})
            engine_used = "faster-whisper"
        except Exception as exc:
            failures.append({"engine": "faster-whisper", "error": str(exc)})
            write_asr_attempts(output_dir, requested_engine, failures)

    if target is None and requested_engine in {"openai-whisper", "whisper"} and shutil.which("whisper"):
        target = run_openai_whisper_cli(audio_path, output_dir, base_dir, hotword_context or {})
        engine_used = "openai-whisper-cli"

    if target is None:
        raise RuntimeError("ASR 引擎均不可用：" + json.dumps(failures, ensure_ascii=False))

    elapsed = round(time.time() - pipeline_started, 3)
    augment_asr_quality(
        output_dir,
        requested_engine=requested_engine,
        engine_used=engine_used,
        fallback_failures=failures,
        elapsed_sec=elapsed,
        audio_path=audio_path,
    )
    emit_asr_progress(progress_callback, stage="complete", message=f"ASR 完成，用时 {format_elapsed(elapsed)}", percent=100)
    return target


def write_asr_attempts(output_dir: Path, requested_engine: str, attempts: list[dict[str, str]]) -> None:
    """Persist failures immediately so a slow fallback never hides its trigger."""
    (output_dir / "asr_attempts.json").write_text(json.dumps({
        "requested_engine": requested_engine,
        "attempts": attempts,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def mlx_asr_available() -> bool:
    return sys.platform == "darwin" and module_available("mlx_whisper") and module_available("mlx")


def configured_mlx_model() -> str:
    explicit = os.environ.get("MOTOOLBOX_MLX_ASR_MODEL", "").strip()
    generic = os.environ.get("MOTOOLBOX_ASR_MODEL", "").strip()
    model = explicit or generic or DEFAULT_FAST_ASR_MODEL
    return MLX_MODEL_ALIASES.get(model, model)


def load_pcm16_audio(audio_path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != 16000:
        raise RuntimeError(
            f"MLX 需要 16kHz/mono/PCM16 音频，实际为 {sample_rate}Hz/{channels}ch/{sample_width * 8}bit"
        )
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, len(audio) / sample_rate


def run_mlx_whisper(audio_path: Path, output_dir: Path, hotword_context: dict[str, Any], progress_callback=None) -> Path:
    import mlx_whisper

    started = time.time()
    target = output_dir / "asr.srt"
    audio, duration = load_pcm16_audio(audio_path)
    model_name = configured_mlx_model()
    language_hint = os.environ.get("MOTOOLBOX_ASR_LANGUAGE", "").strip()
    word_timestamps = env_flag("MOTOOLBOX_ASR_WORD_TIMESTAMPS", default=False)
    kwargs: dict[str, Any] = {
        "path_or_hf_repo": model_name,
        "verbose": False,
        "condition_on_previous_text": False,
        "word_timestamps": word_timestamps,
    }
    if language_hint:
        kwargs["language"] = language_hint
    prompt = asr_hotword_prompt(hotword_context)
    if prompt:
        kwargs["initial_prompt"] = prompt
    reliable_language = None
    def decode(samples, retry):
        nonlocal reliable_language
        options = dict(kwargs)
        if retry:
            options.pop("initial_prompt", None)
            if not language_hint and reliable_language:
                options["language"] = reliable_language
        decoded = mlx_whisper.transcribe(samples, **options)
        if not asr_issues(decoded.get("segments", []), len(samples) / 16000):
            reliable_language = decoded.get("language")
        return decoded

    result = transcribe_chunks(audio, decode, lambda done, total: emit_asr_progress(
        progress_callback, stage="mlx", message=f"音频分段转写 {done}/{total}（含异常段重试）",
        percent=18 + int(75 * done / total),
    )) if duration > 180 else decode(audio, False)
    segments = list(result.get("segments") or [])
    repetition_candidates = [
        item for item in segments
        if float(item.get("start") or 0) < max(duration - 1.0, 0)
    ]
    repetition_count = sum(
        1 for item in repetition_candidates if looks_like_asr_repetition(str(item.get("text") or ""))
    )
    repetition_ratio = repetition_count / max(len(repetition_candidates), 1)
    if repetition_ratio > MAX_ASR_REPETITION_RATIO:
        raise RuntimeError(
            f"MLX ASR 重复幻觉比例过高（{repetition_count}/{len(segments)}，约 {repetition_ratio:.0%}）"
        )
    lines: list[str] = []
    quality_segments: list[dict[str, Any]] = []
    replacement_chars = 0
    cue_index = 0
    discarded_tail_segments = 0
    for segment in segments:
        raw_text = str(segment.get("text") or "").strip()
        replacement_chars += raw_text.count("\ufffd")
        text = sanitize_asr_replacement_chars(raw_text)
        if not text:
            continue
        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        # Whisper may continue decoding past the final audio frame and invent a
        # short closing phrase. Never persist cues that start outside the source,
        # and clamp a partially overlapping final cue to the real duration.
        if start >= duration:
            discarded_tail_segments += 1
            continue
        if start >= duration - 1.0 and (looks_like_asr_repetition(raw_text) or end > duration + 0.25):
            discarded_tail_segments += 1
            continue
        end = min(end, duration)
        if end <= start:
            discarded_tail_segments += 1
            continue
        cue_index += 1
        lines.extend([str(cue_index), f"{srt_timestamp(start)} --> {srt_timestamp(end)}", text, ""])
        quality_segments.append({
            "index": cue_index,
            "start": round(start, 3),
            "end": round(end, 3),
            "avg_logprob": round(float(segment.get("avg_logprob") or 0), 4),
            "no_speech_prob": round(float(segment.get("no_speech_prob") or 0), 4),
            "compression_ratio": round(float(segment.get("compression_ratio") or 0), 4),
        })
    target.write_text("\n".join(lines), encoding="utf-8")
    if target.stat().st_size == 0:
        raise RuntimeError("MLX ASR finished, but no SRT file was generated.")
    suspicious = suspicious_asr_segments(quality_segments)
    elapsed = round(time.time() - started, 3)
    (output_dir / "quality.json").write_text(json.dumps({
        "engine": "mlx-whisper",
        "asr_pipeline_version": ASR_PIPELINE_VERSION,
        "audio_chunks": result.get("audio_chunks", []),
        "unresolved_intervals": result.get("unresolved_intervals", []),
        "device": "metal",
        "model": model_name,
        "language": result.get("language") or language_hint or "unknown",
        "duration": round(duration, 3),
        "segment_count": len(quality_segments),
        "suspicious_segment_count": len(suspicious),
        "suspicious_segments": suspicious,
        "replacement_chars_repaired": replacement_chars,
        "discarded_tail_segments": discarded_tail_segments,
        "elapsed_sec": elapsed,
        "realtime_factor": round(elapsed / max(duration, 0.001), 4),
        "settings": {
            "decoder": "greedy",
            "word_timestamps": word_timestamps,
            "language_hint": language_hint or "auto",
            "condition_on_previous_text": False,
            "chunk_seconds": 180,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def extract_audio(video_path: Path, audio_path: Path, base_dir: Path) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]
    result = subprocess.run(command, cwd=base_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg audio extraction failed").strip())


def module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def run_openai_whisper_cli(audio_path: Path, output_dir: Path, base_dir: Path, hotword_context: dict[str, Any]) -> Path:
    command = [
        "whisper",
        str(audio_path),
        "--task",
        "transcribe",
        "--model",
        os.environ.get("MOTOOLBOX_ASR_MODEL", DEFAULT_FAST_ASR_MODEL),
        "--output_format",
        "srt",
        "--output_dir",
        str(output_dir),
    ]
    prompt = asr_hotword_prompt(hotword_context)
    if prompt:
        command.extend(["--initial_prompt", prompt])
    language_hint = os.environ.get("MOTOOLBOX_ASR_LANGUAGE", "").strip()
    if language_hint:
        command.extend(["--language", language_hint])
    result = subprocess.run(command, cwd=base_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "whisper ASR failed").strip())

    generated = output_dir / f"{audio_path.stem}.srt"
    target = output_dir / "asr.srt"
    if generated.exists():
        generated.replace(target)
    if not target.exists():
        raise RuntimeError("ASR finished, but no SRT file was generated.")
    (output_dir / "quality.json").write_text(json.dumps({
        "engine": "openai-whisper-cli",
        "model": os.environ.get("MOTOOLBOX_ASR_MODEL", DEFAULT_FAST_ASR_MODEL),
        "language_hint": language_hint or "auto",
        "confidence_available": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def run_faster_whisper(audio_path: Path, output_dir: Path, hotword_context: dict[str, Any]) -> Path:
    from faster_whisper import WhisperModel

    target = output_dir / "asr.srt"
    started = time.time()
    requested_model = os.environ.get("MOTOOLBOX_ASR_MODEL", DEFAULT_FAST_ASR_MODEL).strip() or DEFAULT_FAST_ASR_MODEL
    model_names = asr_model_candidates(requested_model)
    terms = [
        str(term) for term in hotword_context.get("terms", []) if str(term).strip()
    ][:FASTER_WHISPER_MAX_CONTEXT_TERMS]
    beam_size = max(1, int(os.environ.get("MOTOOLBOX_ASR_BEAM_SIZE", "1") or 1))
    word_timestamps = env_flag("MOTOOLBOX_ASR_WORD_TIMESTAMPS", default=False)
    transcribe_args: dict[str, Any] = {
        "vad_filter": True,
        "beam_size": beam_size,
        "condition_on_previous_text": False,
        "word_timestamps": word_timestamps,
    }
    language_hint = os.environ.get("MOTOOLBOX_ASR_LANGUAGE", "").strip()
    if language_hint:
        transcribe_args["language"] = language_hint
    prompt = asr_hotword_prompt(hotword_context)
    if prompt:
        transcribe_args["initial_prompt"] = prompt
    # initial_prompt is the safer default. Strong hotword bias can turn unrelated
    # English speech into repeated domain vocabulary. Keep it opt-in only.
    enable_strong_hotwords = os.environ.get("MOTOOLBOX_ASR_STRONG_HOTWORDS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if terms and enable_strong_hotwords:
        transcribe_args["hotwords"] = " ".join(terms)
    failures: list[dict[str, str]] = []
    segments = None
    info = None
    model_name = requested_model
    audio, audio_duration = load_pcm16_audio(audio_path)
    chunk_result = {}
    reliable_language = None
    for candidate in model_names:
        try:
            model = WhisperModel(
                candidate,
                device=os.environ.get("MOTOOLBOX_ASR_DEVICE", "auto").strip() or "auto",
                compute_type=os.environ.get("MOTOOLBOX_ASR_COMPUTE_TYPE", "int8").strip() or "int8",
            )
            def decode(samples, retry):
                nonlocal reliable_language
                options = dict(transcribe_args)
                if retry:
                    options.pop("initial_prompt", None)
                    options.pop("hotwords", None)
                    if not language_hint and reliable_language:
                        options["language"] = reliable_language
                raw, detected = model.transcribe(samples, **options)
                decoded = {"segments": [asdict(s) for s in raw], "language": detected.language}
                if not asr_issues(decoded["segments"], len(samples) / 16000):
                    reliable_language = detected.language
                return decoded

            if audio_duration > 180:
                from types import SimpleNamespace
                chunk_result = transcribe_chunks(audio, decode)
                candidate_segments = [SimpleNamespace(**s) for s in chunk_result["segments"]]
                candidate_info = SimpleNamespace(language=chunk_result["language"], duration=audio_duration)
            else:
                raw_segments, candidate_info = model.transcribe(str(audio_path), **transcribe_args)
                candidate_segments = list(raw_segments)
            repetition_count = sum(
                1 for segment in candidate_segments if looks_like_asr_repetition(str(segment.text or ""))
            )
            repetition_ratio = repetition_count / max(len(candidate_segments), 1)
            if repetition_ratio > MAX_ASR_REPETITION_RATIO:
                raise RuntimeError(
                    f"ASR 重复幻觉比例过高（{repetition_count}/{len(candidate_segments)}，"
                    f"约 {repetition_ratio:.0%}）"
                )
            segments, info = candidate_segments, candidate_info
            model_name = candidate
            break
        except Exception as exc:
            failures.append({"model": candidate, "error": str(exc), "cache": faster_whisper_cache_state(candidate)})
            if is_huggingface_rate_limit(exc):
                mark_huggingface_rate_limit(exc)
            # Never delete the complete model repository here. A failed download
            # followed by immediate deletion/retry amplifies Hub 429 errors.
            continue
    if segments is None or info is None:
        raise RuntimeError("本地 ASR 模型均不可用：" + json.dumps(failures, ensure_ascii=False))

    lines = []
    quality_segments: list[dict[str, Any]] = []
    replacement_chars = 0
    for index, segment in enumerate(segments, start=1):
        raw_text = segment.text.strip()
        replacement_chars += raw_text.count("\ufffd")
        text = sanitize_asr_replacement_chars(raw_text)
        if not text:
            continue
        lines.extend([str(index), f"{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}", text, ""])
        quality_segments.append({
            "index": index,
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "avg_logprob": round(float(getattr(segment, "avg_logprob", 0.0)), 4),
            "no_speech_prob": round(float(getattr(segment, "no_speech_prob", 0.0)), 4),
            "compression_ratio": round(float(getattr(segment, "compression_ratio", 0.0)), 4),
        })
    target.write_text("\n".join(lines), encoding="utf-8")
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError("ASR finished, but no SRT file was generated.")
    suspicious = suspicious_asr_segments(quality_segments)
    elapsed = round(time.time() - started, 3)
    duration = round(float(getattr(info, "duration", 0.0)), 3)
    (output_dir / "quality.json").write_text(json.dumps({
        "engine": "faster-whisper",
        "asr_pipeline_version": ASR_PIPELINE_VERSION,
        "audio_chunks": chunk_result.get("audio_chunks", []),
        "unresolved_intervals": chunk_result.get("unresolved_intervals", []),
        "model": model_name,
        "language": getattr(info, "language", "unknown"),
        "language_probability": round(float(getattr(info, "language_probability", 0.0)), 4),
        "duration": duration,
        "duration_after_vad": round(float(getattr(info, "duration_after_vad", 0.0)), 3),
        "segment_count": len(quality_segments),
        "suspicious_segment_count": len(suspicious),
        "suspicious_segments": suspicious,
        "requested_model": requested_model,
        "fallback_used": model_name != requested_model,
        "model_attempts": failures,
        "replacement_chars_repaired": replacement_chars,
        "elapsed_sec": elapsed,
        "realtime_factor": round(elapsed / max(duration, 0.001), 4),
        "settings": {
            "beam_size": beam_size,
            "vad_filter": True,
            "word_timestamps": word_timestamps,
            "language_hint": language_hint or "auto",
            "compute_type": os.environ.get("MOTOOLBOX_ASR_COMPUTE_TYPE", "int8") or "int8",
        },
        "strong_hotwords_enabled": enable_strong_hotwords,
        "context_term_count": len(terms),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def srt_timestamp(value: float) -> str:
    millis = int(round(float(value) * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def suspicious_asr_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in segments
        if item["avg_logprob"] < -1.0 or item["no_speech_prob"] > 0.6 or item["compression_ratio"] > 2.4
    ]


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def emit_asr_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    message: str,
    percent: int,
) -> None:
    if callback:
        callback({"stage": stage, "message": message, "percent": percent})


def format_elapsed(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分 {seconds} 秒"
    if minutes:
        return f"{minutes} 分 {seconds} 秒"
    return f"{seconds} 秒"


def augment_asr_quality(
    output_dir: Path,
    *,
    requested_engine: str,
    engine_used: str,
    fallback_failures: list[dict[str, str]],
    elapsed_sec: float,
    audio_path: Path,
) -> None:
    path = output_dir / "quality.json"
    try:
        quality = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        quality = {}
    finished_at = datetime.now(timezone.utc)
    started_at = datetime.fromtimestamp(finished_at.timestamp() - elapsed_sec, timezone.utc)
    quality.update({
        "requested_engine": requested_engine,
        "engine_used": engine_used,
        "engine_fallbacks": fallback_failures,
        "pipeline_started_at": started_at.isoformat(),
        "pipeline_finished_at": finished_at.isoformat(),
        "pipeline_elapsed_sec": elapsed_sec,
        "audio_bytes": audio_path.stat().st_size if audio_path.exists() else 0,
    })
    path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")


def faster_whisper_model_id(model_name: str) -> str:
    return FASTER_WHISPER_MODEL_ALIASES.get(model_name, model_name)


def faster_whisper_cache_dir(model_name: str) -> Path:
    model_id = faster_whisper_model_id(model_name)
    return Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))


def faster_whisper_cache_state(model_name: str) -> str:
    cache_dir = faster_whisper_cache_dir(model_name)
    if not cache_dir.exists():
        return "missing"
    model_files = list(cache_dir.glob("snapshots/*/model.bin"))
    if any(path.exists() and path.resolve().exists() for path in model_files):
        return "complete"
    return "partial"


def huggingface_cooldown_path() -> Path:
    return Path.home() / ".cache" / "motoolbox" / "hf-rate-limit.json"


def huggingface_rate_limited() -> bool:
    path = huggingface_cooldown_path()
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("retry_after_epoch") or 0) > time.time()
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def mark_huggingface_rate_limit(exc: Exception) -> None:
    path = huggingface_cooldown_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "retry_after_epoch": time.time() + HF_RATE_LIMIT_COOLDOWN_SEC,
        "error": str(exc)[-2000:],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def is_huggingface_rate_limit(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "too many requests" in message


def asr_model_candidates(requested_model: str) -> list[str]:
    candidates: list[str] = []
    requested_state = faster_whisper_cache_state(requested_model)
    if requested_state == "complete" or not huggingface_rate_limited():
        candidates.append(requested_model)
    for model_name in LOCAL_ASR_RESCUE_MODELS:
        if faster_whisper_cache_state(model_name) == "complete" and model_name not in candidates:
            candidates.append(model_name)
    if not candidates:
        raise RuntimeError(
            "Hugging Face 当前处于限流冷却期，且没有完整的本地 ASR 模型。"
            "请稍后重试，或预先下载模型后再生成。"
        )
    return candidates


def sanitize_asr_replacement_chars(text: str) -> str:
    """Keep uncertain speech visible without rejecting an otherwise complete transcript."""
    text = str(text or "")
    if looks_like_asr_repetition(text):
        return "[ASR 重复失真，待核实]"
    text = re.sub(r"\ufffd+", "[听不清]", text)
    return re.sub(r"(?:\[听不清\]\s*){2,}", "[听不清]", text).strip()


def looks_like_asr_repetition(text: str) -> bool:
    compact = re.sub(r"[\s，。,.!?！？]", "", str(text or ""))
    if len(compact) < 40:
        return False
    if re.search(r"(.{1,12})\1{5,}", compact):
        return True
    # Long low-diversity output is another common Whisper runaway pattern.
    return len(compact) >= 100 and len(set(compact)) / len(compact) < 0.12


def write_transcript_meta(result_dir: Path, meta: dict[str, Any]) -> None:
    (result_dir / "transcript" / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
