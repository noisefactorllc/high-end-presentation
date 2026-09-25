import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hep import fal


class FakeFal:
    """A fake queue: each request needs `polls` status calls before completing."""

    def __init__(self, polls=1, fail=False):
        self.polls, self.fail = polls, fail
        self.submits, self.status_calls = 0, 0
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                assert self.headers["Authorization"] == "Key k"
                n = int(self.headers["Content-Length"])
                json.loads(self.rfile.read(n))
                fake.submits += 1
                base = f"http://127.0.0.1:{fake.port}/requests/r{fake.submits}"
                self._send(200, {"request_id": f"r{fake.submits}", "status_url": base + "/status",
                                 "response_url": base})

            def do_GET(self):
                if self.path.endswith("/status"):
                    fake.status_calls += 1
                    done = fake.status_calls >= fake.polls
                    self._send(200, {"status": "COMPLETED" if done else "IN_PROGRESS"})
                elif fake.fail:
                    self._send(422, {"detail": "bad input"})
                else:
                    self._send(200, {"images": [{"url": "x"}]})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def client(self):
        return fal.FalClient("k", base=f"http://127.0.0.1:{self.port}", interval=0.01)


def test_submit_poll_result():
    f = FakeFal(polls=3)
    state = {}
    r = f.client().run("a/b", {"prompt": "x"}, state, "still", timeout=5)
    assert r == {"images": [{"url": "x"}]}
    assert f.submits == 1 and state["still"]["done"]


def test_error_result_raises():
    f = FakeFal(fail=True)
    with pytest.raises(fal.FalError, match="422"):
        f.client().run("a/b", {}, {}, "s", timeout=5)


def test_timeout_keeps_handle_and_resume_does_not_resubmit():
    f = FakeFal(polls=10**9)
    state, saves = {}, []
    with pytest.raises(fal.FalTimeout):
        f.client().run("a/b", {"p": 1}, state, "video", timeout=0.05, save=lambda: saves.append(1))
    assert state["video"]["request_id"] == "r1" and saves
    f.polls = 0
    r = f.client().run("a/b", {"p": 1}, state, "video", timeout=5)
    assert r["images"] and f.submits == 1


def test_changed_args_resubmit():
    f = FakeFal(polls=1)
    state = {}
    c = f.client()
    c.run("a/b", {"p": 1}, state, "s", timeout=5)
    c.run("a/b", {"p": 2}, state, "s", timeout=5)
    assert f.submits == 2


def test_missing_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(fal.FalError, match="FAL_KEY"):
        fal.key_from_env()


def test_data_uri_downsizes(tmp_path):
    import numpy as np
    uri = fal.data_uri(np.zeros((100, 400, 3), np.float32), max_side=200)
    assert uri.startswith("data:image/jpeg;base64,")


def test_image_size_follows_aspect_and_resolution():
    from hep import endpoints
    assert endpoints.image_size("16:9", "2K") == {"width": 2048, "height": 1152}
    assert endpoints.image_size("4:5", "2K") == {"width": 1632, "height": 2048}
    args = endpoints.image_args("openai/gpt-image-2.5/flare/edit", "p", ["u1", "u2"], "1:1", "1K")
    assert args["image_size"] == {"width": 1024, "height": 1024} and args["image_urls"] == ["u1", "u2"]
    assert "aspect_ratio" not in args
