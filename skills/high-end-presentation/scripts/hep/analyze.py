"""Measurable facts about the piece: palette, tone, representative frame.

The agent adds the interpretive half (themes, mood) to the job brief.
"""
import cv2
import numpy as np

from . import color, media


def _work_size(img, max_side=256):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    return media.resize(img, max(1, round(w * s)), max(1, round(h * s))) if s < 1 else img


def palette(img, k=6, seed=0):
    """Dominant colors by k-means in CIELAB, sorted by pixel share."""
    small = _work_size(img)
    lab = color.rgb_to_lab(small).reshape(-1, 3).astype(np.float32)
    k = int(min(k, len(np.unique(lab.round(1), axis=0))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
    cv2.setRNGSeed(seed)
    _, labels, centers = cv2.kmeans(lab, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k).astype(np.float64)
    out = []
    for i in np.argsort(-counts):
        rgb = color.lab_to_rgb(centers[i][None, None])[0, 0]
        out.append({
            "hex": color.to_hex(rgb),
            "rgb": [round(float(c), 4) for c in rgb],
            "lab": [round(float(c), 2) for c in centers[i]],
            "weight": round(float(counts[i] / counts.sum()), 4),
        })
    return out


def tone_stats(img):
    """key: mean lightness 0-1; contrast: lightness spread; warmth: +warm/-cool;
    saturation: mean chroma, roughly 0-1."""
    lab = color.rgb_to_lab(_work_size(img)).reshape(-1, 3)
    L = lab[:, 0] / 100.0
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    p5, p95 = np.percentile(L, [5, 95])
    return {
        "key": round(float(L.mean()), 4),
        "contrast": round(float(p95 - p5), 4),
        "warmth": round(float(np.mean(lab[:, 2]) / 50.0 + np.mean(lab[:, 1]) / 100.0), 4),
        "saturation": round(float(np.clip(chroma.mean() / 60.0, 0, 2)), 4),
    }


def _sharpness(img):
    g = color.luma(_work_size(img, 512))
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def representative_frame(path, samples=24):
    """Pick the frame that is sharp and close to the clip's average look."""
    info = media.video_info(path)
    total = max(info["frames"], 1)
    step = max(1, total // samples)
    cands = [(i, f) for i, f in media.iter_frames(path, step=step)]
    if not cands:
        raise ValueError(f"no frames in {path}")
    means = np.array([color.rgb_to_lab(_work_size(f, 64)).reshape(-1, 3).mean(0) for _, f in cands])
    center = np.median(means, 0)
    dist = np.linalg.norm(means - center, axis=1)
    sharp = np.array([_sharpness(f) for _, f in cands])
    score = (sharp / (sharp.max() + 1e-9)) - 0.5 * (dist / (dist.max() + 1e-9))
    best = int(np.argmax(score))
    return cands[best][0], cands[best][1]


def analyze_piece(path):
    """Return the analysis dict, and the still used to represent the piece."""
    if media.is_video(path):
        index, still = representative_frame(path)
        info = media.video_info(path)
        source = {"kind": "video", "frame_index": index, **info}
    else:
        still = media.load_image_any_depth(path)
        source = {"kind": "image", "width": still.shape[1], "height": still.shape[0]}
    return {
        "source": source,
        "aspect": round(still.shape[1] / still.shape[0], 4),
        "palette": palette(still),
        "tone": tone_stats(still),
    }, still
