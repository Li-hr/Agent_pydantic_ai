# -*- coding: utf-8 -*-
"""简单图像工具：裁剪 + 阈值分割。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _load_gray(image_path: str) -> Tuple[Image.Image, np.ndarray]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图像不存在: {path}")
    img = Image.open(path)
    gray = img.convert("L")
    return gray, np.asarray(gray)


def _default_out(image_path: str, suffix: str) -> Path:
    src = Path(image_path)
    return src.with_name(f"{src.stem}_{suffix}{src.suffix or '.png'}")


def crop_image(
    image_path: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    output_path: Optional[str] = None,
) -> str:
    """按像素框裁剪图像，坐标为左上/右下（含左上，不含右下边界惯例同 PIL）。"""
    img = Image.open(image_path)
    w, h = img.size
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError(f"裁剪框非法: ({left},{top},{right},{bottom})，图像尺寸={w}x{h}")

    cropped = img.crop((left, top, right, bottom))
    out = Path(output_path) if output_path else _default_out(image_path, "crop")
    out.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out)
    return str(out.resolve())


def threshold_segment(
    image_path: str,
    threshold: int = 128,
    invert: bool = False,
    output_path: Optional[str] = None,
) -> str:
    """灰度阈值分割：默认 >= threshold 为前景(255)，否则背景(0)。"""
    if not 0 <= threshold <= 255:
        raise ValueError(f"threshold 需在 0–255，收到 {threshold}")

    _, arr = _load_gray(image_path)
    if invert:
        mask = arr < threshold
    else:
        mask = arr >= threshold
    binary = (mask.astype(np.uint8) * 255)

    out = Path(output_path) if output_path else _default_out(image_path, "thresh")
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(out)

    fg = int(mask.sum())
    total = int(mask.size)
    print(f"[threshold_segment] fg={fg}/{total} ({100.0 * fg / total:.1f}%) -> {out}")
    return str(out.resolve())
