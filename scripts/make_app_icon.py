"""生成 MOtoolbox Apple Silicon 应用图标。

设计语言：奶油暖底 + 深棕，呼应 Web UI 配色；主体为简报卡片
（蓝色判断标题条 + 子弹点行），右下角播放圆徽代表发布会视频源。
播放徽章使用深棕色。

依赖 Pillow（仅生成时需要，产物 .icns 已入库）：
    python3 -m venv /tmp/iconenv && /tmp/iconenv/bin/pip install pillow
    /tmp/iconenv/bin/python scripts/make_app_icon.py
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 1024
OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

CREAM_TOP = (250, 245, 230)
CREAM_BOTTOM = (233, 222, 196)
CARD = (255, 253, 246)
BROWN = (94, 72, 41)
BROWN_SOFT = (138, 115, 82)
BLUE_TITLE = (45, 96, 196)


def rounded_mask(size: int, box: tuple[int, int, int, int], radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    return mask


def vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    gradient = Image.new("RGB", (1, size))
    for y in range(size):
        t = y / (size - 1)
        gradient.putpixel((0, y), tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return gradient.resize((size, size))


def draw_icon() -> Image.Image:
    badge_color = BROWN
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # 外层圆角方块（macOS Big Sur 规格：1024 画布、824 主体、半径约 185）
    plate_box = (100, 100, 924, 924)
    plate = vertical_gradient(SIZE, CREAM_TOP, CREAM_BOTTOM).convert("RGBA")
    canvas.paste(plate, (0, 0), rounded_mask(SIZE, plate_box, 185))

    draw = ImageDraw.Draw(canvas)
    # 外层细描边，避免浅色图标在浅色背景下糊边
    draw.rounded_rectangle(plate_box, radius=185, outline=(120, 98, 64, 70), width=4)
    # 简报卡片投影
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((292, 268, 736, 812), radius=44, fill=(70, 52, 26, 90))
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(22)))
    draw = ImageDraw.Draw(canvas)

    # 简报卡片
    card_box = (280, 244, 724, 788)
    draw.rounded_rectangle(card_box, radius=44, fill=CARD, outline=(*BROWN_SOFT, 120), width=3)

    # 蓝色判断标题条（概述的灵魂）
    draw.rounded_rectangle((332, 312, 600, 358), radius=23, fill=BLUE_TITLE)

    # 子弹点行
    rows = [(418, 640), (492, 600), (566, 648)]
    for y, line_end in rows:
        draw.ellipse((338, y, 366, y + 28), fill=BROWN_SOFT)
        draw.rounded_rectangle((392, y + 1, line_end, y + 27), radius=13, fill=(*BROWN_SOFT, 110))

    # 播放圆徽（发布会视频源）
    cx, cy, r = 700, 740, 138
    halo = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse((cx - r - 8, cy - r + 6, cx + r + 8, cy + r + 22), fill=(60, 44, 20, 110))
    canvas = Image.alpha_composite(canvas, halo.filter(ImageFilter.GaussianBlur(16)))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=badge_color, outline=CARD, width=10)
    t = 58  # 播放三角
    draw.polygon(
        [(cx - t + 12, cy - t), (cx - t + 12, cy + t), (cx + t + 6, cy)],
        fill=CREAM_TOP,
    )

    # 裁回外层圆角，确保所有元素不越界
    final = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    final.paste(canvas, (0, 0), rounded_mask(SIZE, plate_box, 185))
    return final


def write_icns(image: Image.Image, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{name}.png"
    image.save(png_path)
    icns_path = OUT_DIR / f"{name}.icns"
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            image.resize((base, base), Image.LANCZOS).save(iconset / f"icon_{base}x{base}.png")
            image.resize((base * 2, base * 2), Image.LANCZOS).save(iconset / f"icon_{base}x{base}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)], check=True)
    return icns_path


def main() -> None:
    if shutil.which("iconutil") is None:
        raise SystemExit("需要 macOS 自带的 iconutil。")
    path = write_icns(draw_icon(), "MOtoolbox")
    print(f"standard: {path}")


if __name__ == "__main__":
    main()
