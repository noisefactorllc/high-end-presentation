"""Minimal fal queue client (urllib only).

Every submitted request is recorded in a caller-owned state dict (persisted
by the caller) before waiting, so an interrupted stage resumes the same
request instead of paying for a new one.
"""
import base64
import hashlib
import io
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

QUEUE = "https://queue.fal.run"


class FalError(RuntimeError):
    pass


class FalTimeout(FalError):
    pass


def key_from_env():
    key = os.environ.get("FAL_KEY", "").strip()
    if not key:
        raise FalError("FAL_KEY is not set. Export your fal API key as FAL_KEY and rerun.")
    return key


def data_uri(path_or_array, max_side=None, fmt="JPEG", quality=95):
    """Encode a file (or float sRGB array) as a data URI. max_side downsizes."""
    from PIL import Image
    import numpy as np

    if isinstance(path_or_array, (str, Path)):
        p = Path(path_or_array)
        if max_side is None:
            mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
        im = Image.open(p).convert("RGB")
    else:
        a = np.clip(np.asarray(path_or_array, np.float32), 0, 1)
        im = Image.fromarray((a * 255 + 0.5).astype(np.uint8))
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, fmt, quality=quality) if fmt == "JPEG" else im.save(buf, fmt)
    mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def args_digest(args):
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:16]


class FalClient:
    def __init__(self, key, base=QUEUE, interval=4.0):
        self.key = key
        self.base = base.rstrip("/")
        self.interval = interval

    def _req(self, url, body=None, method=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method or ("POST" if body is not None else "GET"))
        req.add_header("Authorization", f"Key {self.key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            raise FalError(f"fal HTTP {e.code} for {url}: {detail}") from None
        except urllib.error.URLError as e:
            raise FalError(f"fal unreachable ({url}): {e.reason}") from None

    def submit(self, endpoint, args):
        r = self._req(f"{self.base}/{endpoint}", args)
        for k in ("request_id", "status_url", "response_url"):
            if k not in r:
                raise FalError(f"unexpected submit response from {endpoint}: {str(r)[:500]}")
        return {"request_id": r["request_id"], "status_url": r["status_url"], "response_url": r["response_url"]}

    def status(self, handle):
        return self._req(handle["status_url"])

    def result(self, handle):
        return self._req(handle["response_url"])

    def wait(self, handle, timeout):
        deadline = time.monotonic() + timeout
        while True:
            st = self.status(handle)
            s = st.get("status")
            if s == "COMPLETED":
                if st.get("error"):
                    raise FalError(f"fal request {handle['request_id']} failed: {st.get('error')}")
                return self.result(handle)
            if s not in ("IN_QUEUE", "IN_PROGRESS"):
                raise FalError(f"fal request {handle['request_id']} status {s}: {str(st)[:500]}")
            if time.monotonic() >= deadline:
                raise FalTimeout(f"fal request {handle['request_id']} still {s} after {timeout:.0f}s; rerun to resume")
            time.sleep(self.interval)

    def run(self, endpoint, args, state, slot, timeout, save=lambda: None):
        """Submit (or resume) a request recorded at state[slot]; return its result.

        A stored handle is reused only for the same endpoint and arguments."""
        digest = args_digest(args)
        h = state.get(slot)
        if not h or h.get("endpoint") != endpoint or h.get("args") != digest or h.get("done"):
            h = {"endpoint": endpoint, "args": digest, **self.submit(endpoint, args), "submitted": time.time()}
            state[slot] = h
            save()
        result = self.wait(h, timeout)
        h["done"] = True
        save()
        return result


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("data:"):
        path.write_bytes(base64.b64decode(url.split(",", 1)[1]))
        return path
    req = urllib.request.Request(url, headers={"User-Agent": "high-end-presentation"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return path
