# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build for the Windows chip-matching GUI."""

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve()
APP_NAME = "芯片替代料匹配工具"

datas = [
    (
        str(PROJECT_ROOT / "products" / "MCC" / "clean"),
        "products/MCC/clean",
    ),
    (str(PROJECT_ROOT / "README.md"), "."),
    (str(PROJECT_ROOT / "试用说明.txt"), "."),
]

a = Analysis(
    [str(PROJECT_ROOT / "app.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 这些包存在于开发用 .venv，但运行 GUI 并不需要；明确排除可避免
    # pandas 的可选集成把整套科学计算/Jupyter 工具链带入成品。
    excludes=["sklearn", "scipy", "IPython", "jedi", "parso", "zmq"],
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
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
