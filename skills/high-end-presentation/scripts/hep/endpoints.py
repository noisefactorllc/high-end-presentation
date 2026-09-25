"""fal endpoint adapters: build arguments and read results."""
IMAGE_DEFAULT = "fal-ai/nano-banana-pro/edit"
VIDEO_DEFAULT = "fal-ai/kling-video/v3/pro/image-to-video"
DEPTH_DEFAULT = "fal-ai/image-preprocessors/depth-anything/v2"

ASPECTS = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
RESOLUTIONS = {"1K", "2K", "4K"}


LONG_SIDE = {"1K": 1024, "2K": 2048, "4K": 4096}


def image_size(aspect, resolution):
    """Explicit pixel size for endpoints that take one (multiples of 16)."""
    aw, ah = (int(x) for x in aspect.split(":"))
    long = LONG_SIDE[resolution]
    w, h = (long, long * ah / aw) if aw >= ah else (long * aw / ah, long)
    return {"width": int(round(w / 16) * 16), "height": int(round(h / 16) * 16)}


def image_args(endpoint, prompt, piece_uris, aspect, resolution, seed=None):
    if endpoint.startswith("openai/gpt-image"):
        return {"prompt": prompt, "image_urls": list(piece_uris), "image_size": image_size(aspect, resolution),
                "quality": "high", "output_format": "png", "num_images": 1}
    args = {"prompt": prompt, "image_urls": list(piece_uris), "aspect_ratio": aspect, "resolution": resolution,
            "output_format": "png", "num_images": 1}
    if seed is not None:
        args["seed"] = seed
    return args


def image_url(result):
    return result["images"][0]["url"]


def video_args(endpoint, prompt, negative, plate_uri, duration):
    if endpoint.startswith("bytedance/seedance"):
        return {"prompt": prompt, "image_url": plate_uri, "end_image_url": plate_uri, "duration": str(duration),
                "resolution": "1080p", "generate_audio": False}
    if not endpoint.startswith("fal-ai/kling-video/v3"):
        raise ValueError(f"no argument adapter for video endpoint {endpoint}")
    return {"prompt": prompt, "negative_prompt": negative, "start_image_url": plate_uri, "end_image_url": plate_uri,
            "duration": str(duration), "generate_audio": False}


def video_url(result):
    return result["video"]["url"]


def depth_args(endpoint, plate_uri):
    return {"image_url": plate_uri}


def depth_url(result):
    return result["image"]["url"]
