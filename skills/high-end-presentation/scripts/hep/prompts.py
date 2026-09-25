"""Fixed prompt clauses. The agent writes the creative scene and motion text;
these clauses are always appended and are not negotiable."""

ENERGY_MOTION = {
    "quiet": ("Over the clip, one or two people drift in slowly. One of them stops beside the artwork and "
              "looks at it for a long, still moment, then leaves. Light shifts very gently."),
    "calm": ("A few people pass at an unhurried pace. Two of them stop beside the artwork for a while to "
             "look at it before moving on. Soft light and shadow move across the room."),
    "lively": ("A steady flow of people walks through the space. Some pass by, several stop to look at the "
               "artwork, then continue. Light and shadows move with the activity."),
    "bustling": ("A busy crowd moves through the scene in both directions. Most people pass quickly; a few "
                 "stop in front of the artwork to look. Moving lights and shadows sweep across the space."),
}

KEEP_CLEAR = ("People stay beside the artwork or behind the camera's line to it; nobody walks between the "
              "camera and the artwork.")


def still_prompt(scene_text, piece_w, piece_h):
    from math import gcd
    g = gcd(piece_w, piece_h) or 1
    ratio = f"{piece_w // g}:{piece_h // g}" if max(piece_w // g, piece_h // g) < 50 else f"{piece_w / piece_h:.3f}:1"
    return (
        f"{scene_text.strip()}\n\n"
        "The provided image is the artwork on display. Reproduce it exactly as given: the same composition, "
        f"marks, detail, and colors, at its exact aspect ratio ({ratio}). Do not crop, redraw, restyle, extend, "
        "mirror, or add anything to the artwork. The whole artwork is visible and nothing covers it. "
        "The scene contains no people. Photographic realism; the room's light falls naturally on the artwork."
    )


def video_prompt(motion_text, energy):
    parts = [motion_text.strip() or ENERGY_MOTION[energy], ENERGY_MOTION[energy] if motion_text.strip() else ""]
    if energy in ("quiet", "calm"):
        parts.append(KEEP_CLEAR)
    parts.append(
        "Locked-off static camera on a tripod: no camera movement, no zoom, no refocus. The artwork on display "
        "never changes. The clip begins and ends on this exact empty scene. No text, captions, or logos."
    )
    return "\n".join(p for p in parts if p)


VIDEO_NEGATIVE = ("camera movement, pan, zoom, dolly, shaky camera, refocus, morphing artwork, changing artwork, "
                  "text, captions, watermark, logo")
