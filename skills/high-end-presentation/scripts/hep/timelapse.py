"""Turn a locked-camera video of the scene into one long exposure.

Two passes over the frames, all registered onto the plate:
  1. background B: the clip's first and last frames (the video starts and
     ends on the clean plate); a per-pixel median when the ends disagree;
  2. per pixel: presence P (share of frames that differ from B), mean color
     C of those frames, lighten max L, plain mean M.

Ghosts:  alpha = 1 - (1 - P) ** k   (k = energy ghost density). A person who
stood still has P near 1 and stays solid; a passer-by has small P and is
faint but boosted by k.
Trails:  bright light above the ghost image, from L.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import color, locate, media

ENERGY = {
    "quiet":    {"ghost_k": 1.6, "trail_gain": 0.35, "occlusion": 0.0},
    "calm":     {"ghost_k": 1.8, "trail_gain": 0.6,  "occlusion": 0.0},
    "lively":   {"ghost_k": 2.2, "trail_gain": 0.9,  "occlusion": 0.2},
    "bustling": {"ghost_k": 2.6, "trail_gain": 1.2,  "occlusion": 0.4},
}

MEDIAN_SAMPLES = 41


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


@dataclass
class Stack:
    B: np.ndarray          # background, linear
    M: np.ndarray          # plain mean, linear
    L: np.ndarray          # lighten max, linear
    P: np.ndarray          # presence 0-1
    C: np.ndarray          # mean color while present, linear
    canvas_scale: float    # canvas px = plate px * canvas_scale
    count: int
    max_shift: float       # largest jitter, fraction of width
    moved: bool
    notes: list = field(default_factory=list)


def register(frame0, plate):
    """Homography video px -> plate px. Falls back to a centered cover-fit when
    features do not match (for example a nearly blank scene)."""
    loc = locate.find_piece(frame0, plate, piece_side=1400, scene_side=1400, min_inliers=40)
    fh, fw = frame0.shape[:2]
    ph, pw = plate.shape[:2]
    if loc.ok:
        return loc.H, "sift"
    s = max(pw / fw, ph / fh)
    tx, ty = (pw - fw * s) / 2, (ph - fh * s) / 2
    return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]], np.float64), "cover-fit"


def _gray_small(frame, s):
    g = color.luma(frame)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _jitter(g0, gt, s):
    """3x3 map frame_t px -> frame_0 px from ECC (Euclidean) at scale s."""
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(g0, gt, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
    except cv2.error:
        return np.eye(3), False
    W3 = np.vstack([W, [0, 0, 1]]).astype(np.float64)
    S = np.diag([s, s, 1.0])
    W_full = np.linalg.inv(S) @ W3 @ S          # frame_0 px -> frame_t px
    return np.linalg.inv(W_full), True


def stack(video_path, plate, *, canvas_scale=None, jitter=True):
    """Register every frame onto the plate and accumulate the stacks."""
    info = media.video_info(video_path)
    frame0 = media.read_frame(video_path, 0)
    H, how = register(frame0, plate)
    ph, pw = plate.shape[:2]
    if canvas_scale is None:
        # canvas at roughly video pixel density
        canvas_scale = min(1.0, np.sqrt((info["width"] * info["height"]) / float(pw * ph)) * 1.05)
    cw, ch = round(pw * canvas_scale), round(ph * canvas_scale)
    Mc = np.diag([canvas_scale, canvas_scale, 1.0]) @ H
    es = min(1.0, 320.0 / max(frame0.shape[:2]))
    g0 = _gray_small(frame0, es)
    notes = [f"registration:{how}"]

    jit, shifts, failures = {}, [], 0

    def warped(i, f):
        nonlocal failures
        J = np.eye(3)
        if jitter and i > 0:
            J, ok = _jitter(g0, _gray_small(f, es), es)
            failures += (not ok)
        jit[i] = J
        shifts.append(float(np.hypot(J[0, 2], J[1, 2]) / f.shape[1]))
        return cv2.warpPerspective(f, Mc @ J, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # pass 1: register every frame; keep the clip's ends and a median subsample.
    # The video starts and ends on the clean plate, so its ends are the true
    # background even when a viewer stands still for most of the clip.
    from collections import deque
    total = max(info["frames"], 1)
    step = max(1, total // MEDIAN_SAMPLES)
    head, tail, samples = [], deque(maxlen=3), []
    for i, f in media.iter_frames(video_path):
        a = warped(i, f)
        if len(head) < 3:
            head.append(a)
        tail.append(a)
        if i % step == 0:
            samples.append((a * 255 + 0.5).astype(np.uint8))
    ends_gap = float(np.abs(np.mean(head, 0) - np.mean(tail, 0)).mean())
    if ends_gap < 0.03:
        B_srgb = np.mean(list(head) + list(tail), 0).astype(np.float32)
        notes.append("background:clip-ends")
    else:
        B_srgb = np.median(np.stack(samples), axis=0).astype(np.float32) / 255.0
        notes.append(f"background:median (clip ends differ by {ends_gap:.3f}; long-standing viewers may vanish)")
    del samples, head, tail
    B = color.srgb_to_linear(B_srgb)

    # pass 2: presence, colors, lighten, mean
    sumM = np.zeros_like(B, np.float64)
    sumC = np.zeros_like(B, np.float64)
    sumP = np.zeros(B.shape[:2], np.float64)
    L = np.zeros_like(B)
    n = 0
    for i, f in media.iter_frames(video_path):
        a = cv2.warpPerspective(f, Mc @ jit.get(i, np.eye(3)), (cw, ch), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        lin = color.srgb_to_linear(a)
        w = smoothstep(0.035, 0.09, np.abs(a - B_srgb).max(-1))
        sumM += lin
        sumC += lin * w[..., None]
        sumP += w
        np.maximum(L, lin, out=L)
        n += 1
    P = (sumP / max(n, 1)).astype(np.float32)
    C = np.where(sumP[..., None] > 1e-6, sumC / np.maximum(sumP[..., None], 1e-6), B).astype(np.float32)
    M = (sumM / max(n, 1)).astype(np.float32)
    max_shift = max(shifts) if shifts else 0.0
    if failures:
        notes.append(f"jitter-estimate-failed-frames:{failures}")
    moved = max_shift > 0.01
    if moved:
        notes.append(f"camera-moved:{max_shift:.3f}")
    return Stack(B, M, L, P, C, canvas_scale, n, max_shift, moved, notes)


def _up(a, shape):
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def develop(plate_srgb, st, piece_mask, energy, quad=None):
    """Composite the long exposure onto the full-resolution plate.

    plate_srgb: repaired plate (the relit original is already in the piece region).
    piece_mask: soft mask of the piece at plate resolution.
    Returns (out_linear, report)."""
    e = ENERGY[energy] if isinstance(energy, str) else energy
    alpha = 1.0 - (1.0 - np.clip(st.P, 0, 1)) ** e["ghost_k"]
    G = alpha[..., None] * (st.C - st.B)
    ghost_img = st.B + G
    excess = np.maximum(st.L - np.maximum(ghost_img, st.B), 0)
    T = excess * smoothstep(0.03, 0.2, color.luma(excess))[..., None]
    D = G + e["trail_gain"] * T
    # dead zone against compression noise
    mag = np.abs(D).max(-1, keepdims=True)
    D = D * smoothstep(0.002, 0.008, mag)

    shape = plate_srgb.shape
    plate_lin = color.srgb_to_linear(plate_srgb)
    m_canvas = cv2.resize(piece_mask, (st.B.shape[1], st.B.shape[0]), interpolation=cv2.INTER_AREA)
    area = max(float(m_canvas.sum()), 1.0)
    sigma = max(2.0, 0.08 * np.sqrt(area))
    # lighting over the piece: low-frequency ratio of mean exposure to background
    K = cv2.GaussianBlur(st.M, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(st.B, (0, 0), sigma), 1e-4)
    K = np.clip(K, 0.3, 1.6)
    D_low = cv2.GaussianBlur(D, (0, 0), sigma)
    D_hf = D - D_low
    inside = m_canvas > 0.5
    occ_raw = float((np.abs(D_hf).max(-1)[inside] > 0.03).mean()) if inside.any() else 0.0

    D_up, K_up, Dhf_up = _up(D, shape), _up(K, shape), _up(D_hf, shape)
    m = piece_mask[..., None]
    outside = plate_lin + D_up
    in_piece = plate_lin * K_up + e["occlusion"] * Dhf_up
    out = outside * (1 - m) + in_piece * m
    report = {
        "frames": st.count,
        "canvas_scale": round(st.canvas_scale, 4),
        "presence_mean": round(float(st.P.mean()), 4),
        "ghost_coverage": round(float((alpha > 0.05).mean()), 4),
        "trail_peak": round(float(T.max()), 4),
        "piece_light_range": [round(float(K[inside].min()), 3), round(float(K[inside].max()), 3)] if inside.any() else None,
        "occlusion_raw": round(occ_raw, 4),
        "occlusion_allowed": e["occlusion"],
        "max_shift": round(st.max_shift, 4),
        "notes": list(st.notes),
    }
    if st.P.mean() < 0.002:
        report["notes"].append("no-motion: the video shows almost no activity")
    if occ_raw > 0.05 and e["occlusion"] == 0:
        report["notes"].append(f"occlusion-removed:{occ_raw:.3f} (people crossed the piece; energy forbids it)")
    return np.clip(out, 0, None).astype(np.float32), report
