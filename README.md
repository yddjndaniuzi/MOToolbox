# MOtoolbox

MOtoolbox is a local workspace for marketing-officer workflows.

Current MVP:

- `pressconf keyframes`: extract candidate key screenshots from a launch-event video and generate a review index.

## Quick Start

### Web App

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pressconf.web
```

Then open:

```text
http://127.0.0.1:5058
```

### macOS App Packaging

Build a double-clickable Mac app with bundled Python dependencies, `yt-dlp`, OpenCV, and ffmpeg:

```bash
scripts/package_macos.sh
```

To also bundle the current Obsidian Markdown knowledge base:

```bash
scripts/package_macos.sh --with-vault
```

The app stores user config and generated working files under:

```text
~/Library/Application Support/MOtoolbox/
```

On macOS, use **文件与缓存** on the home page (or in the admin console) to view the active paths and open them in Finder. Result and preview pages also provide **在访达中打开图包** for the current event. Under the data directory, `pressconf/raw/<event-slug>/frames/` contains screenshots, `pressconf/downloads/` stores downloads, and `pressconf/uploads/` stores uploads. Speech models are cached separately in `~/.cache/huggingface/hub/`. Source runs use the repository as the data directory unless `MOTOOLBOX_DATA_DIR` is set.

Packaging excludes local config, API keys, uploads, downloads, generated `raw/` drafts, indexes, and caches from the app bundle.

### macOS WebView Shell

For a native-feeling desktop window without rewriting the web app in SwiftUI, open the Swift package in Xcode:

```bash
open macos/MOtoolboxShell/Package.swift
```

The shell is an AppKit + `WKWebView` wrapper. It starts the existing local Flask app through `start_web.sh`, waits for `http://127.0.0.1:5058/healthz`, and loads the tool in its own macOS window.

You can also build a lightweight `.app` from the command line:

```bash
scripts/build_macos_shell.sh
```

The default output is:

```text
dist/MOtoolboxShell.app
```

To build a portable app that can be copied to another Mac without the source repository, bundle the PyInstaller backend inside the WebView shell:

```bash
scripts/package_portable_macos_app.sh
```

Only Apple Silicon (`arm64`) builds are maintained. To make the target explicit:

```bash
scripts/package_portable_macos_app.sh --target-arch=arm64
```

The default portable output is:

```text
dist/MOtoolbox.app
```

The portable app is unsigned by default. On another Mac, Gatekeeper may require opening it once via right-click → Open, or removing the quarantine attribute after copying:

```bash
xattr -dr com.apple.quarantine /path/to/MOtoolbox.app
```

The web app supports local video paths, video uploads, `auto` / `lecture` / `program` capture profiles, and result previews.

Current web workflow:

- Submit a local path, uploaded video, online video URL, or live-stream URL.
- Online videos are downloaded with `yt-dlp` before analysis.
- Live streams are resolved with `yt-dlp`, then sampled and recorded in real time until the user stops the job. An optional duration can be set as an automatic stop.
- Watch background capture progress in the browser.
- Review screenshots in a grid.
- Mark useful frames as selected, hide noisy frames, and copy selected screenshots as Markdown.
- Use the review-video tab to fetch SRT/ASR transcripts from media review videos and generate text-only sentiment, viewpoint, and quote analysis.

### CLI

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pressconf.keyframes /path/to/event.mp4 --event-name "Apple WWDC 2026"
```

Outputs are written to:

```text
pressconf/raw/<event-slug>/
```

For real launch-event recordings, the default is designed to prioritize completeness:

```bash
python -m pressconf.keyframes /path/to/event.mp4 --event-name "vivo S16"
```

Choose a capture profile by launch-event format:

```bash
# Live keynote: presenter + PPT, e.g. traditional stage-based product events.
python -m pressconf.keyframes /path/to/event.mp4 --event-name "vivo S16" --profile lecture

# Edited video program: dense cuts and motion, e.g. post-COVID Apple events.
python -m pressconf.keyframes /path/to/event.mp4 --event-name "Apple Event" --profile program
```

Profile behavior:

- `auto`: probes the first 10 minutes, then chooses `lecture` or `program` from visual-change statistics.
- `lecture`: keeps frames when the deck appears to change. This is best for stable PPT pages.
- `program`: keeps one useful frame every few seconds for time coverage, and additionally keeps major visual cuts.

Useful tuning knobs:

- `--sample-every 1`: sample more densely when the deck changes quickly.
- `--diff-threshold 0.01`: keep more visually similar slides.
- `--min-gap 4`: in `program` mode, reduce this to keep denser coverage.
- `--max-frames 800`: set a hard cap only when you want a smaller candidate pool. The default `0` means no limit.

### 更新采集引擎

在「后台控制台 → 采集引擎更新」查看当前 yt-dlp 版本并更新到官方最新稳定版；采集页在线视频说明中也有入口。macOS 版下载官方通用可执行文件，校验 SHA256 和实际运行版本后原子替换，下一次采集无需重启即可生效。更新保存在应用数据目录的 `tools/yt-dlp`，不会修改已签名的应用包；下载或验证失败保留原版本。
