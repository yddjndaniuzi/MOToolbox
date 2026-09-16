# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


ROOT = Path.cwd()
PACKAGE_RESOURCES = ROOT / "build" / "package_resources"
APP_NAME = os.environ.get("MOTOOLBOX_PACKAGE_NAME", "MOtoolbox")
APP_DISPLAY_NAME = os.environ.get("MOTOOLBOX_DISPLAY_NAME", APP_NAME)
BUNDLE_IDENTIFIER = os.environ.get("MOTOOLBOX_BUNDLE_IDENTIFIER", "com.motoolbox.pressconf")
TARGET_ARCH = os.environ.get("MOTOOLBOX_TARGET_ARCH") or None
CONSOLE = os.environ.get("MOTOOLBOX_CONSOLE", "").strip() == "1"

datas = [
    (str(ROOT / "pressconf" / "templates"), "pressconf/templates"),
    (str(ROOT / "pressconf" / "static"), "pressconf/static"),
    (str(ROOT / "pressconf" / "asr_hotwords"), "pressconf/asr_hotwords"),
    (str(ROOT / "pressconf" / "domains"), "pressconf/domains"),
]

embedded_vault = PACKAGE_RESOURCES / "pressconf" / "embedded_vault"
if embedded_vault.exists():
    datas.append((str(embedded_vault), "pressconf/embedded_vault"))

datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("faster_whisper", includes=["assets/*"])
datas += collect_data_files("mlx_whisper", includes=["assets/*"])
datas += collect_data_files("mlx", includes=["lib/*"])
datas += collect_data_files("cv2", includes=["data/haarcascade_frontalface_default.xml"])
datas += collect_data_files("certifi")

hiddenimports = []
hiddenimports += collect_submodules("yt_dlp")
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("mlx_whisper")
hiddenimports += collect_submodules("mlx")
hiddenimports += collect_submodules("scipy._external.array_api_compat")

binaries = collect_dynamic_libs("mlx")


a = Analysis(
    [str(ROOT / "pressconf" / "mac_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # mlx-whisper declares torch for checkpoint conversion utilities, but the
    # runtime transcriber uses only MLX. Excluding torch keeps the app bundle
    # hundreds of MB smaller.
    excludes=["torch"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=CONSOLE,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=TARGET_ARCH,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=None,
    bundle_identifier=BUNDLE_IDENTIFIER,
    info_plist={
        "CFBundleDisplayName": APP_DISPLAY_NAME,
        "CFBundleName": APP_DISPLAY_NAME,
        "NSHighResolutionCapable": True,
    },
)
