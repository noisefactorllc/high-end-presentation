"""Job directory state. job.json is the single record of a presentation run."""
import json
import os
import time
from pathlib import Path

ORDER = ["init", "analyze", "brief", "still", "video", "develop", "finish"]


class JobError(RuntimeError):
    pass


class Job:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "job.json"
        if not self.path.exists():
            raise JobError(f"no job at {self.root} (run `init` first)")
        self.data = json.loads(self.path.read_text())

    @classmethod
    def create(cls, root, data):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stages": {}, "fal": {},
                "warnings": [], **data}
        tmp = root / "job.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, root / "job.json")
        return cls(root)

    def save(self):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=float))
        os.replace(tmp, self.path)

    def file(self, *parts):
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def done(self, stage, **info):
        self.data["stages"][stage] = {"done": True, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **info}
        self.save()

    def is_done(self, stage):
        return bool(self.data["stages"].get(stage, {}).get("done"))

    def require(self, *stages):
        """Require the named stages and every stage before them in ORDER."""
        last = max(ORDER.index(s) for s in stages)
        missing = [s for s in ORDER[:last + 1] if not self.is_done(s)]
        if missing:
            raise JobError(f"run these stages first: {', '.join(missing)}")

    def warn(self, msg):
        if msg not in self.data["warnings"]:
            self.data["warnings"].append(msg)
            self.save()

    def __getitem__(self, k):
        return self.data[k]

    def get(self, k, d=None):
        return self.data.get(k, d)
