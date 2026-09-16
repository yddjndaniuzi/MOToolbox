from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass
class Keyframe:
    index: int
    timestamp_sec: float
    timestamp: str
    path: str
    diff_score: float
    reason: str


@dataclass(frozen=True)
class CaptureProfile:
    sample_every: float
    diff_threshold: float
    min_gap: float
    strong_threshold: float
    strong_min_gap: float


CAPTURE_PROFILES: dict[str, CaptureProfile] = {
    # Stage keynote / presenter + deck. The main signal is PPT page changes.
    "lecture": CaptureProfile(
        sample_every=1.5,
        diff_threshold=0.012,
        min_gap=0.0,
        strong_threshold=0.08,
        strong_min_gap=0.0,
    ),
    # Edited video program. Keep steady time coverage, plus major visual cuts.
    "program": CaptureProfile(
        sample_every=1.0,
        diff_threshold=0.012,
        min_gap=6.0,
        strong_threshold=0.04,
        strong_min_gap=2.0,
    ),
    # Balanced default for unknown recordings.
    "auto": CaptureProfile(
        sample_every=2.0,
        diff_threshold=0.015,
        min_gap=0.0,
        strong_threshold=0.10,
        strong_min_gap=0.0,
    ),
}


def detect_capture_profile(video_path: Path, probe_seconds: float = 600, sample_every: float = 2.0) -> tuple[str, dict[str, float | int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else probe_seconds
    scan_until = min(duration, probe_seconds)
    step = max(1, int(round(fps * sample_every)))
    max_frame = int(scan_until * fps)

    previous_fingerprint: np.ndarray | None = None
    diffs: list[float] = []
    frame_number = 0

    while frame_number <= max_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            break
        fingerprint = frame_fingerprint(frame)
        if previous_fingerprint is not None:
            diffs.append(fingerprint_diff(previous_fingerprint, fingerprint))
        previous_fingerprint = fingerprint
        frame_number += step

    capture.release()

    if not diffs:
        return "lecture", {
            "probe_seconds": round(scan_until, 3),
            "probe_samples": 0,
            "probe_active_ratio": 0.0,
            "probe_strong_ratio": 0.0,
            "probe_median_diff": 0.0,
            "probe_p75_diff": 0.0,
        }

    values = np.array(diffs, dtype=np.float32)
    active_ratio = float(np.mean(values >= 0.015))
    strong_ratio = float(np.mean(values >= 0.08))
    median_diff = float(np.median(values))
    p75_diff = float(np.percentile(values, 75))

    # Edited launch films tend to keep moving: more frequent medium changes,
    # plus occasional hard cuts. Keynotes with static decks spend more time still.
    detected = "program" if active_ratio >= 0.32 or median_diff >= 0.018 or p75_diff >= 0.035 else "lecture"
    confidence = max(
        abs(active_ratio - 0.32) / 0.32,
        abs(median_diff - 0.018) / 0.018,
        abs(p75_diff - 0.035) / 0.035,
    )

    return detected, {
        "probe_seconds": round(scan_until, 3),
        "probe_samples": len(diffs) + 1,
        "probe_active_ratio": round(active_ratio, 4),
        "probe_strong_ratio": round(strong_ratio, 4),
        "probe_median_diff": round(median_diff, 4),
        "probe_p75_diff": round(p75_diff, 4),
        "probe_confidence": round(min(confidence, 1.0), 4),
    }


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "pressconf-event"


def format_timestamp(seconds: float) -> str:
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    return str(timedelta(seconds=whole)) + f".{millis:03d}"


def frame_fingerprint(frame: np.ndarray, size: int = 16) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 80, 160)
    small_edges = cv2.resize(edges, (size, size), interpolation=cv2.INTER_AREA)
    return np.concatenate(
        [
            small.astype(np.float32).reshape(-1) / 255.0,
            small_edges.astype(np.float32).reshape(-1) / 255.0,
        ]
    )


def fingerprint_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def should_keep_frame(
    *,
    profile: str,
    last_fingerprint: np.ndarray | None,
    diff_score: float,
    timestamp_sec: float,
    last_kept_timestamp_sec: float | None,
    diff_threshold: float,
    min_gap: float,
    strong_threshold: float,
    strong_min_gap: float,
) -> tuple[bool, str]:
    if last_fingerprint is None or last_kept_timestamp_sec is None:
        return True, "first"

    elapsed = timestamp_sec - last_kept_timestamp_sec
    if profile == "program":
        if elapsed >= strong_min_gap and diff_score >= strong_threshold:
            return True, "strong-change"
        if elapsed >= min_gap and diff_score >= diff_threshold:
            return True, "time-coverage"
        return False, ""

    if diff_score >= diff_threshold:
        return True, "visual-change"
    return False, ""


@lru_cache(maxsize=1)
def frontal_face_cascade() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


def is_speaker_only_frame(frame: np.ndarray) -> bool:
    """Conservatively catch presenter close-ups with little brief value."""
    cascade = frontal_face_cascade()
    if cascade is None:
        return False

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_width = min(width, 640)
    if detect_width != width:
        scale = detect_width / width
        detection_gray = cv2.resize(
            gray,
            (detect_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        detection_gray = gray

    # A keynote slide keeps a tiny presenter in frame often. Only filter close
    # frontal shots where a face-sized subject dominates the candidate image.
    min_face = max(24, int(round(detect_width * 0.07)))
    faces = cascade.detectMultiScale(
        detection_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_face, min_face),
    )
    if len(faces) != 1 or has_dark_stage_border(gray):
        return False

    detection_height, detection_width = detection_gray.shape[:2]
    for x, y, face_width, face_height in faces:
        face_area_ratio = (face_width * face_height) / (detection_width * detection_height)
        face_center_x = (x + face_width / 2) / detection_width
        face_center_y = (y + face_height / 2) / detection_height
        if face_area_ratio >= 0.025 and 0.24 <= face_center_x <= 0.76 and face_center_y <= 0.52:
            return True
    return False


def has_dark_stage_border(gray: np.ndarray) -> bool:
    height, width = gray.shape[:2]
    top_height = max(1, height // 12)
    side_width = max(1, width // 16)
    edge_pixels = np.concatenate(
        [
            gray[:top_height].reshape(-1),
            gray[-top_height:].reshape(-1),
            gray[:, :side_width].reshape(-1),
            gray[:, -side_width:].reshape(-1),
        ]
    )
    return float(np.mean(edge_pixels < 32)) >= 0.45


def save_frame(frame: np.ndarray, path: Path, max_width: int) -> None:
    height, width = frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        frame = cv2.resize(
            frame,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    profile: str,
    sample_every: float,
    diff_threshold: float,
    min_gap: float,
    strong_threshold: float,
    strong_min_gap: float,
    max_frames: int,
    max_width: int,
    avoid_speaker_only: bool = True,
    progress_callback: Callable[[dict[str, float | int | bool | str]], None] | None = None,
) -> tuple[list[Keyframe], dict[str, float | int | bool]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0
    step = max(1, int(round(fps * sample_every)))

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    keyframes: list[Keyframe] = []
    last_fingerprint: np.ndarray | None = None
    last_kept_timestamp_sec: float | None = None
    frame_number = 0
    sampled_frames = 0
    speaker_only_skipped = 0
    stopped_by_max_frames = False
    last_timestamp_sec = 0.0
    progress_stride = 10

    while True:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            break

        timestamp_sec = frame_number / fps
        last_timestamp_sec = timestamp_sec
        sampled_frames += 1
        fingerprint = frame_fingerprint(frame)
        diff_score = 1.0 if last_fingerprint is None else fingerprint_diff(last_fingerprint, fingerprint)
        keep_frame, keep_reason = should_keep_frame(
            profile=profile,
            last_fingerprint=last_fingerprint,
            diff_score=diff_score,
            timestamp_sec=timestamp_sec,
            last_kept_timestamp_sec=last_kept_timestamp_sec,
            diff_threshold=diff_threshold,
            min_gap=min_gap,
            strong_threshold=strong_threshold,
            strong_min_gap=strong_min_gap,
        )

        if keep_frame and avoid_speaker_only and is_speaker_only_frame(frame):
            speaker_only_skipped += 1
        elif keep_frame:
            index = len(keyframes) + 1
            filename = f"{index:04d}_{int(timestamp_sec):06d}s.jpg"
            frame_path = frames_dir / filename
            save_frame(frame, frame_path, max_width=max_width)
            keyframes.append(
                Keyframe(
                    index=index,
                    timestamp_sec=round(timestamp_sec, 3),
                    timestamp=format_timestamp(timestamp_sec),
                    path=str(frame_path.relative_to(output_dir)),
                    diff_score=round(diff_score, 4),
                    reason=keep_reason,
                )
            )
            last_fingerprint = fingerprint
            last_kept_timestamp_sec = timestamp_sec

        if progress_callback and (sampled_frames == 1 or sampled_frames % progress_stride == 0):
            progress_callback(
                {
                    "duration_sec": round(float(duration), 3),
                    "last_timestamp_sec": round(float(last_timestamp_sec), 3),
                    "sampled_frames": sampled_frames,
                    "keyframes": len(keyframes),
                    "percent": round((last_timestamp_sec / duration * 100), 2) if duration else 0.0,
                    "stage": "extracting",
                }
            )

        if max_frames > 0 and len(keyframes) >= max_frames:
            stopped_by_max_frames = True
            break

        frame_number += step
        if duration and timestamp_sec >= duration:
            break

    capture.release()
    if progress_callback:
        progress_callback(
            {
                "duration_sec": round(float(duration), 3),
                "last_timestamp_sec": round(float(last_timestamp_sec), 3),
                "sampled_frames": sampled_frames,
                "keyframes": len(keyframes),
                "percent": 100.0,
                "stage": "writing",
            }
        )
    stats: dict[str, float | int | bool] = {
        "fps": round(float(fps), 3),
        "total_frames": total_frames,
        "duration_sec": round(float(duration), 3),
        "sampled_frames": sampled_frames,
        "speaker_only_skipped": speaker_only_skipped,
        "last_timestamp_sec": round(float(last_timestamp_sec), 3),
        "stopped_by_max_frames": stopped_by_max_frames,
        "effective_profile": profile,
    }
    return keyframes, stats


def extract_live_keyframes(
    stream_url: str,
    output_dir: Path,
    profile: str,
    sample_every: float,
    diff_threshold: float,
    min_gap: float,
    strong_threshold: float,
    strong_min_gap: float,
    duration_seconds: int | None,
    max_frames: int,
    max_width: int,
    avoid_speaker_only: bool = True,
    progress_callback: Callable[[dict[str, float | int | bool | str]], None] | None = None,
    control_callback: Callable[[], str] | None = None,
) -> tuple[list[Keyframe], dict[str, float | int | bool]]:
    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        raise RuntimeError("Cannot open live stream.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_step = max(1, int(round(fps * sample_every)))

    frames_dir = output_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    keyframes: list[Keyframe] = []
    last_fingerprint: np.ndarray | None = None
    last_kept_timestamp_sec: float | None = None
    frame_number = 0
    sampled_frames = 0
    speaker_only_skipped = 0
    stopped_by_max_frames = False
    stopped_by_user = False
    last_timestamp_sec = 0.0
    max_stream_frames = int(max(1, duration_seconds) * fps) if duration_seconds else None

    while max_stream_frames is None or frame_number <= max_stream_frames:
        control = control_callback() if control_callback else "running"
        if control == "stop":
            stopped_by_user = True
            break
        saving_paused = control == "pause"

        ok, frame = capture.read()
        if not ok:
            break

        if frame_number % frame_step != 0:
            frame_number += 1
            continue

        timestamp_sec = frame_number / fps
        last_timestamp_sec = timestamp_sec
        sampled_frames += 1
        if saving_paused:
            if progress_callback and (sampled_frames == 1 or sampled_frames % 10 == 0):
                progress_callback(
                    {
                        "duration_sec": round(float(duration_seconds or 0), 3),
                        "last_timestamp_sec": round(float(last_timestamp_sec), 3),
                        "sampled_frames": sampled_frames,
                        "keyframes": len(keyframes),
                        "percent": round((last_timestamp_sec / duration_seconds * 100), 2) if duration_seconds else 0.0,
                        "stage": "paused",
                    }
                )
            frame_number += 1
            continue

        fingerprint = frame_fingerprint(frame)
        diff_score = 1.0 if last_fingerprint is None else fingerprint_diff(last_fingerprint, fingerprint)
        keep_frame, keep_reason = should_keep_frame(
            profile=profile,
            last_fingerprint=last_fingerprint,
            diff_score=diff_score,
            timestamp_sec=timestamp_sec,
            last_kept_timestamp_sec=last_kept_timestamp_sec,
            diff_threshold=diff_threshold,
            min_gap=min_gap,
            strong_threshold=strong_threshold,
            strong_min_gap=strong_min_gap,
        )

        if keep_frame and avoid_speaker_only and is_speaker_only_frame(frame):
            speaker_only_skipped += 1
        elif keep_frame:
            index = len(keyframes) + 1
            filename = f"{index:04d}_{int(timestamp_sec):06d}s.jpg"
            frame_path = frames_dir / filename
            save_frame(frame, frame_path, max_width=max_width)
            keyframes.append(
                Keyframe(
                    index=index,
                    timestamp_sec=round(timestamp_sec, 3),
                    timestamp=format_timestamp(timestamp_sec),
                    path=str(frame_path.relative_to(output_dir)),
                    diff_score=round(diff_score, 4),
                    reason=keep_reason,
                )
            )
            last_fingerprint = fingerprint
            last_kept_timestamp_sec = timestamp_sec
            if progress_callback:
                progress_callback(
                    {
                        "duration_sec": round(float(duration_seconds or 0), 3),
                        "last_timestamp_sec": round(float(last_timestamp_sec), 3),
                        "sampled_frames": sampled_frames,
                        "keyframes": len(keyframes),
                        "percent": round((last_timestamp_sec / duration_seconds * 100), 2) if duration_seconds else 0.0,
                        "stage": "live",
                        "latest_frame": keyframes[-1].path,
                    }
                )

        if progress_callback and (sampled_frames == 1 or sampled_frames % 10 == 0):
            progress_callback(
                {
                    "duration_sec": round(float(duration_seconds or 0), 3),
                    "last_timestamp_sec": round(float(last_timestamp_sec), 3),
                    "sampled_frames": sampled_frames,
                    "keyframes": len(keyframes),
                    "percent": round((last_timestamp_sec / duration_seconds * 100), 2) if duration_seconds else 0.0,
                    "stage": "live",
                }
            )

        if max_frames > 0 and len(keyframes) >= max_frames:
            stopped_by_max_frames = True
            break

        frame_number += 1

    capture.release()
    if progress_callback:
        progress_callback(
            {
                "duration_sec": round(float(duration_seconds or 0), 3),
                "last_timestamp_sec": round(float(last_timestamp_sec), 3),
                "sampled_frames": sampled_frames,
                "keyframes": len(keyframes),
                "percent": 100.0,
                "stage": "writing",
            }
        )

    stats: dict[str, float | int | bool] = {
        "fps": round(float(fps), 3),
        "total_frames": 0,
        "duration_sec": round(float(duration_seconds or last_timestamp_sec), 3),
        "sampled_frames": sampled_frames,
        "speaker_only_skipped": speaker_only_skipped,
        "last_timestamp_sec": round(float(last_timestamp_sec), 3),
        "stopped_by_max_frames": stopped_by_max_frames,
        "stopped_by_user": stopped_by_user,
        "effective_profile": profile,
        "live": True,
    }
    return keyframes, stats


def write_manifest(
    output_dir: Path,
    video_path: Path,
    keyframes: list[Keyframe],
    args: argparse.Namespace,
    stats: dict[str, float | int | bool],
) -> None:
    manifest = {
        "video": str(video_path),
        "event_name": args.event_name,
        "profile": args.profile,
        "effective_profile": args.effective_profile,
        "profile_detection": args.profile_detection,
        "sample_every": args.sample_every,
        "diff_threshold": args.diff_threshold,
        "min_gap": args.min_gap,
        "strong_threshold": args.strong_threshold,
        "strong_min_gap": args.strong_min_gap,
        "avoid_speaker_only": args.avoid_speaker_only,
        "max_frames": args.max_frames,
        "domain": {
            "requested": getattr(args, "domain_requested", "auto"),
            "resolved": getattr(args, "domain_requested", "auto") if getattr(args, "domain_requested", "auto") != "auto" else "",
            "confidence": 1.0 if getattr(args, "domain_requested", "auto") != "auto" else 0.0,
            "signals": ["人工指定"] if getattr(args, "domain_requested", "auto") != "auto" else [],
        },
        "task_type": getattr(args, "task_type", "business_review"),
        "stats": stats,
        "keyframes": [asdict(frame) for frame in keyframes],
    }
    if hasattr(args, "source"):
        manifest["source"] = args.source
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(output_dir: Path, event_name: str, keyframes: list[Keyframe]) -> None:
    lines = [
        f"# {event_name} 关键帧候选",
        "",
        "| # | 时间戳 | 保留原因 | 差异分 | 截图 |",
        "|---:|---|---|---:|---|",
    ]
    for frame in keyframes:
        lines.append(
            f"| {frame.index} | `{frame.timestamp}` | {frame.reason} | {frame.diff_score:.4f} | ![]({frame.path}) |"
        )
    (output_dir / "keyframes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(output_dir: Path, event_name: str, keyframes: list[Keyframe]) -> None:
    cards = []
    for frame in keyframes:
        cards.append(
            f"""
            <article class="card">
              <img src="{html.escape(frame.path)}" alt="keyframe {frame.index}">
              <div class="meta">
                <strong>#{frame.index:04d}</strong>
                <span>{html.escape(frame.timestamp)}</span>
                <span>{html.escape(frame.reason)}</span>
                <span>diff {frame.diff_score:.4f}</span>
              </div>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(event_name)} 关键帧候选</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 18px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: white;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      overflow: hidden;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #111827;
    }}
    .meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      padding: 10px 12px;
      font-size: 13px;
      color: #4b5563;
    }}
    strong {{
      color: #111827;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(event_name)} 关键帧候选</h1>
  <main class="grid">
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract visually distinct keyframe candidates from a press conference video.",
    )
    parser.add_argument("video", type=Path, help="Path to a local video file.")
    parser.add_argument("--event-name", default=None, help="Readable event name.")
    parser.add_argument("--output-root", type=Path, default=Path("pressconf/raw"))
    parser.add_argument(
        "--profile",
        choices=sorted(CAPTURE_PROFILES),
        default="auto",
        help="Capture strategy. auto probes the video and chooses lecture or program.",
    )
    parser.add_argument("--sample-every", type=float, default=None, help="Seconds between sampled frames.")
    parser.add_argument("--diff-threshold", type=float, default=None, help="Visual difference threshold.")
    parser.add_argument("--min-gap", type=float, default=None, help="Minimum seconds between kept frames.")
    parser.add_argument("--strong-threshold", type=float, default=None, help="Diff score for major visual changes.")
    parser.add_argument("--strong-min-gap", type=float, default=None, help="Minimum seconds between major-change frames.")
    parser.add_argument(
        "--allow-speaker-only",
        action="store_true",
        help="Keep close-up presenter-only candidates instead of filtering likely talking-head frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum candidate frames to keep. 0 means no limit.",
    )
    parser.add_argument("--max-width", type=int, default=1600, help="Resize saved screenshots to this width.")
    args = parser.parse_args()
    args.effective_profile = args.profile
    args.profile_detection = {}
    args.avoid_speaker_only = not args.allow_speaker_only
    if args.sample_every is None:
        args.sample_every = CAPTURE_PROFILES[args.profile].sample_every
    if args.diff_threshold is None:
        args.diff_threshold = CAPTURE_PROFILES[args.profile].diff_threshold
    if args.min_gap is None:
        args.min_gap = CAPTURE_PROFILES[args.profile].min_gap
    if args.strong_threshold is None:
        args.strong_threshold = CAPTURE_PROFILES[args.profile].strong_threshold
    if args.strong_min_gap is None:
        args.strong_min_gap = CAPTURE_PROFILES[args.profile].strong_min_gap
    return args


def apply_auto_profile(args: argparse.Namespace, video_path: Path) -> None:
    if args.profile != "auto":
        args.effective_profile = args.profile
        return

    detected, detection_stats = detect_capture_profile(video_path)
    profile = CAPTURE_PROFILES[detected]
    args.effective_profile = detected
    args.profile_detection = detection_stats

    # Keep explicit user overrides; only replace values that still match auto defaults.
    auto = CAPTURE_PROFILES["auto"]
    if args.sample_every == auto.sample_every:
        args.sample_every = profile.sample_every
    if args.diff_threshold == auto.diff_threshold:
        args.diff_threshold = profile.diff_threshold
    if args.min_gap == auto.min_gap:
        args.min_gap = profile.min_gap
    if args.strong_threshold == auto.strong_threshold:
        args.strong_threshold = profile.strong_threshold
    if args.strong_min_gap == auto.strong_min_gap:
        args.strong_min_gap = profile.strong_min_gap


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    apply_auto_profile(args, video_path)
    event_name = args.event_name or video_path.stem
    event_slug = slugify(event_name)
    output_dir = args.output_root / event_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    keyframes, stats = extract_keyframes(
        video_path=video_path,
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
    write_manifest(output_dir, video_path, keyframes, args, stats)
    write_markdown(output_dir, event_name, keyframes)
    write_html(output_dir, event_name, keyframes)

    print(f"Saved {len(keyframes)} keyframe candidates to {output_dir}")
    if args.profile == "auto":
        print(f"Auto profile: {args.effective_profile} ({args.profile_detection})")
    if stats["duration_sec"]:
        print(
            "Video duration: "
            f"{format_timestamp(float(stats['duration_sec']))}; "
            f"sampled until {format_timestamp(float(stats['last_timestamp_sec']))}"
        )
    if stats["stopped_by_max_frames"]:
        print("Warning: stopped early because --max-frames was reached.")
    print(f"Review HTML: {output_dir / 'index.html'}")
    print(f"Markdown index: {output_dir / 'keyframes.md'}")


if __name__ == "__main__":
    main()
