"""Turn a locked-camera video of the scene into one long exposure.

Two passes over the frames, all registered onto the plate:
  1. background B: the clip's first and last frames (the video starts and
     ends on the clean plate); a per-pixel median when the ends disagree;
  2. per pixel: presence P (share of frames that differ from B), mean color
     C of those frames, lighten max L, plain mean M.

Motion blur:  alpha = 1 - (1 - P) ** k over the mean colour while present.
This is the streak layer: continuous smears along each path. k is solved per
clip so the mean opacity meets the energy's target.

Figures: a plain long exposure nearly erases anyone who keeps moving. On top
of the streaks, a handful of real frames spread through the clip are layered
over the plate as a multiple exposure: each frame's moving subjects at partial
opacity. Every passer-by appears several times along their path as a
translucent figure, and people who stop build up towards solid.

Frames are used as the video model delivers them. The video prompt asks for
slow motion, so movement per frame stays small and trails stay continuous.
Trails:  bright light above the ghost image, from L (the mean of the few
brightest frames at each pixel).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import color, locate, media

ENERGY = {
    # ghost_opacity: mean ghost opacity over the pixels people passed through.
    # The density exponent k is solved per clip, so a busy clip does not turn
    # into a veil and a sparse one does not vanish.
    # exposures / exposure_opacity: the multiple-exposure figure layer.
    "quiet":    {"ghost_opacity": 0.28, "exposures": 5,  "exposure_opacity": 0.65,
                 "trail_gain": 0.35},
    "calm":     {"ghost_opacity": 0.37, "exposures": 7,  "exposure_opacity": 0.55,
                 "trail_gain": 0.6},
    "lively":   {"ghost_opacity": 0.45, "exposures": 10, "exposure_opacity": 0.45,
                 "trail_gain": 0.9},
    "bustling": {"ghost_opacity": 0.52, "exposures": 14, "exposure_opacity": 0.38,
                 "trail_gain": 1.2},
}

MEDIAN_SAMPLES = 41
MAX_EXPOSURES = 16
TRAIL_TOP_K = 6  # trails average the k brightest frames per pixel, so a light
                 # seen in a single frame fades instead of printing a dot
GLASS_REFLECTION = 0.7  # share of moving light that reflects in glazing over the piece




def solve_density(P, target, active=0.02):
    """k such that mean(1 - (1 - P) ** k) over active pixels equals target."""
    vals = P[P > active]
    if vals.size == 0:
        return 1.0
    if vals.size > 200_000:
        vals = np.random.default_rng(0).choice(vals, 200_000, replace=False)
    lo, hi = 0.25, 60.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if (1 - (1 - vals) ** mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


@dataclass
class Stack:
    B: np.ndarray          # background, linear
    M: np.ndarray          # plain mean, linear
    L: np.ndarray          # mean of the brightest frames per pixel, linear
    P: np.ndarray          # presence 0-1
    C: np.ndarray          # mean color while present, linear
    canvas_scale: float    # canvas px = plate px * canvas_scale
    count: int
    max_shift: float       # largest jitter, fraction of width
    moved: bool
    notes: list = field(default_factory=list)
    exposures: list = field(default_factory=list)  # [(frame_lin f16, presence f16)] in clip order


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

    # pass 2: presence, colors, lighten, mean; keep evenly spaced exposures
    picks = set(np.linspace(0.06 * total, 0.94 * total, MAX_EXPOSURES).round().astype(int).tolist())
    exposures = []
    sumM = np.zeros_like(B, np.float64)
    sumC = np.zeros_like(B, np.float64)
    sumP = np.zeros(B.shape[:2], np.float64)
    top = np.zeros((TRAIL_TOP_K,) + B.shape, np.float32)
    top_l = np.zeros((TRAIL_TOP_K,) + B.shape[:2], np.float32)
    rows, cols = np.indices(B.shape[:2])
    n = 0
    for i, f in media.iter_frames(video_path):
        a = cv2.warpPerspective(f, Mc @ jit.get(i, np.eye(3)), (cw, ch), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        lin = color.srgb_to_linear(a)
        w = smoothstep(0.035, 0.09, np.abs(a - B_srgb).max(-1))
        sumM += lin
        sumC += lin * w[..., None]
        sumP += w
        lum = color.luma(lin)
        slot = np.argmin(top_l, axis=0)
        brighter = lum > top_l[slot, rows, cols]
        r, c, k = rows[brighter], cols[brighter], slot[brighter]
        top[k, r, c] = lin[brighter]
        top_l[k, r, c] = lum[brighter]
        if i in picks:
            exposures.append((lin.astype(np.float16), w.astype(np.float16)))
        n += 1
    L = top.mean(0) if n >= TRAIL_TOP_K else top.max(0)
    del top, top_l
    P = (sumP / max(n, 1)).astype(np.float32)
    C = np.where(sumP[..., None] > 1e-6, sumC / np.maximum(sumP[..., None], 1e-6), B).astype(np.float32)
    M = (sumM / max(n, 1)).astype(np.float32)
    max_shift = max(shifts) if shifts else 0.0
    if failures:
        notes.append(f"jitter-estimate-failed-frames:{failures}")
    moved = max_shift > 0.01
    if moved:
        notes.append(f"camera-moved:{max_shift:.3f}")
    return Stack(B, M, L, P, C, canvas_scale, n, max_shift, moved, notes, exposures)


def _up(a, shape):
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def figures_only(w, display_hard, min_outside=0.15):
    """Inside the display, keep only moving regions that extend beyond it.
    A person crossing the piece enters from outside the frame; a region that
    changes only inside the frame is the video model distorting the artwork."""
    fg = (w > 0.3).astype(np.uint8)
    n, lab = cv2.connectedComponents(fg, connectivity=8)
    if n <= 1:
        return w * (1 - display_hard)
    flat = lab.ravel()
    total = np.bincount(flat, minlength=n).astype(np.float64)
    outside = np.bincount(flat, weights=(1 - display_hard).ravel(), minlength=n)
    keep = (outside >= np.maximum(40, min_outside * total)).astype(np.float32)
    keep[0] = 0
    kept = cv2.GaussianBlur(keep[lab], (0, 0), 1.5)
    return w * ((1 - display_hard) + display_hard * np.clip(kept, 0, 1))


def multiple_exposure(base, exposures, count, opacity, display_hard=None):
    """Layer `count` evenly spaced exposures' moving subjects over base."""
    if not exposures or count <= 0:
        return base
    idx = np.linspace(0, len(exposures) - 1, min(count, len(exposures))).round().astype(int)
    out = base.copy()
    for i in idx:
        frame, w = exposures[i]
        w = w.astype(np.float32)
        if display_hard is not None:
            w = figures_only(w, display_hard)
        a = cv2.GaussianBlur(w, (0, 0), 1.0)[..., None] * opacity
        out += a * (frame.astype(np.float32) - out)
    return out


def develop(plate_srgb, st, piece_mask, energy, quad=None, glazed=False):
    """Composite the long exposure onto the full-resolution plate.

    plate_srgb: repaired plate (the relit original is already in the piece region).
    piece_mask: soft mask of the piece at plate resolution.
    glazed: the piece is behind glass, so moving light reflects over it.
    Returns (out_linear, report)."""
    e = ENERGY[energy] if isinstance(energy, str) else energy
    shape = plate_srgb.shape
    plate_lin = color.srgb_to_linear(plate_srgb)
    m_canvas = cv2.resize(piece_mask, (st.B.shape[1], st.B.shape[0]), interpolation=cv2.INTER_AREA)
    area = max(float(m_canvas.sum()), 1.0)
    sigma = max(2.0, 0.08 * np.sqrt(area))
    # The display: the piece plus a band that covers its frame. The video model
    # drifts and distorts the artwork and frame during a clip; inside the
    # display only figures that come in from outside it are kept.
    band_px = max(2, int(round(0.015 * np.sqrt(area))))
    hard = cv2.dilate((m_canvas > 0.02).astype(np.uint8), np.ones((2 * band_px + 1,) * 2, np.uint8))
    display = np.maximum(cv2.GaussianBlur(hard.astype(np.float32), (0, 0), band_px), m_canvas)
    hard = hard.astype(np.float32)

    k = e["ghost_k"] if "ghost_k" in e else solve_density(st.P, e["ghost_opacity"])
    alpha = 1.0 - (1.0 - np.clip(st.P, 0, 1)) ** k
    streaks = st.B + (alpha * (1 - display))[..., None] * (st.C - st.B)
    ghost_img = multiple_exposure(streaks, st.exposures, e.get("exposures", 0), e.get("exposure_opacity", 0.0),
                                  display_hard=hard)
    G = ghost_img - st.B
    # trails: only light that is bright in absolute terms (lamps, headlights,
    # glints). People in pale clothes are figures, not trails.
    excess = np.maximum(st.L - np.maximum(ghost_img, st.B), 0)
    bright = smoothstep(0.5, 0.9, color.luma(st.L)) * smoothstep(0.08, 0.3, color.luma(excess))
    T = excess * bright[..., None] * (1 - display)[..., None]
    D = G + e["trail_gain"] * T
    # dead zone against compression noise
    mag = np.abs(D).max(-1, keepdims=True)
    D = D * smoothstep(0.004, 0.012, mag)
    # lighting over the display: low-frequency luminance ratio, limited to what
    # passing people and room light can plausibly do
    lm = cv2.GaussianBlur(color.luma(st.M), (0, 0), sigma)
    lb = cv2.GaussianBlur(color.luma(st.B), (0, 0), sigma)
    K = np.clip(lm / np.maximum(lb, 1e-4), 0.55, 1.15)
    K = 1 + (K - 1) * display
    inside = m_canvas > 0.5
    activity = np.clip(np.abs(D).max(-1) / 0.03, 0, 1)
    covered = float((activity[inside] > 0.5).mean()) if inside.any() else 0.0

    out = plate_lin * _up(K, shape)[..., None] + _up(D, shape)
    if glazed:
        # Glass in front of the piece reflects moving light: trails and lit
        # passers-by add light over it.
        full_T = excess * bright[..., None]
        refl = (e["trail_gain"] * full_T + 0.35 * np.maximum(G, 0)) * display[..., None]
        out = out + GLASS_REFLECTION * _up(refl, shape)
        activity = np.maximum(activity, np.clip(color.luma(refl) / 0.03, 0, 1))
    report = {
        "frames": st.count,
        "ghost_density": round(float(k), 3),
        "canvas_scale": round(st.canvas_scale, 4),
        "presence_mean": round(float(st.P.mean()), 4),
        "ghost_coverage": round(float((alpha > 0.05).mean()), 4),
        "trail_peak": round(float(T.max()), 4),
        "piece_light_range": [round(float(K[inside].min()), 3), round(float(K[inside].max()), 3)] if inside.any() else None,
        "display_band_px": band_px,
        "piece_covered": round(covered, 4),
        "glazed": bool(glazed),
        "exposures_used": min(e.get("exposures", 0), len(st.exposures)),
        "max_shift": round(st.max_shift, 4),
        "notes": list(st.notes),
    }
    if st.P.mean() < 0.002:
        report["notes"].append("no-motion: the video shows almost no activity")
    report["activity"] = _up(activity, shape).astype(np.float16)
    return np.clip(out, 0, None).astype(np.float32), report
