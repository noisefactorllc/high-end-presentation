"""Image and video I/O. Images are returned as float32 HxWx3 sRGB in [0, 1]."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".gif"}


def is_video(path):
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def load_image(path):
    """Load any common still format. Alpha is composited over black."""
    im = Image.open(path)
    im.load()
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, np.float32) / (65535.0 if im.mode.startswith("I;16") else float(max(np.asarray(im).max(), 1)))
        a = np.repeat(a[..., None], 3, -1)
        return np.clip(a, 0, 1)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
    a = np.asarray(im, np.float32) / 255.0
    if a.shape[-1] == 4:
        a = a[..., :3] * a[..., 3:4]
    return np.ascontiguousarray(a[..., :3])


def load_image_any_depth(path):
    """Like load_image, but keeps 16-bit precision for 16-bit RGB PNG/TIFF."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is not None and raw.dtype == np.uint16 and raw.ndim == 3:
        a = raw.astype(np.float32) / 65535.0
        if a.shape[-1] == 4:
            a = a[..., :3] * a[..., 3:4]
        return np.ascontiguousarray(a[..., ::-1][..., :3])
    return load_image(path)


def save_image(path, srgb, quality=95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.asarray(srgb, np.float32), 0, 1)
    if path.suffix.lower() == ".png":
        cv2.imwrite(str(path), (a[..., ::-1] * 65535 + 0.5).astype(np.uint16))
    else:
        Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path, quality=quality)
    return path


def _open(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    return cap


def video_info(path):
    cap = _open(path)
    try:
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()
    return info


def iter_frames(path, step=1):
    """Yield (index, float32 sRGB frame) for every step-th frame."""
    cap = _open(path)
    try:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
                yield i, bgr[..., ::-1].astype(np.float32) / 255.0
            i += 1
    finally:
        cap.release()


def read_frame(path, index):
    for i, frame in iter_frames(path):
        if i == index:
            return frame
    raise IndexError(f"frame {index} not in {path}")


def resize(a, width, height, interpolation=cv2.INTER_AREA):
    return cv2.resize(a, (int(width), int(height)), interpolation=interpolation)
