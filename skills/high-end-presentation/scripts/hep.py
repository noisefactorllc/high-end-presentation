#!/usr/bin/env python3
"""Stage runner for the high-end-presentation skill.

Usage: <python> hep.py <stage> --job DIR [options]
Stages: init, analyze, brief, still, draft, video, develop, finish, status.
Prints one JSON object. Exit codes: 0 ok, 1 error, 2 fidelity gate failed,
3 fal request still pending (rerun the same stage to resume).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hep import fal, stages  # noqa: E402
from hep.job import Job, JobError  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hep.py")
    ap.add_argument("stage", choices=["init", "analyze", "brief", "still", "draft", "video", "develop", "finish",
                                      "status"])
    ap.add_argument("--job", required=True, help="job directory")
    ap.add_argument("--piece")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--scene", default="freeform")
    ap.add_argument("--energy", default="calm")
    ap.add_argument("--aspect", default="4:5")
    ap.add_argument("--resolution", default="2K")
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--echo", action="store_true", help="moving piece: add a faint echo of its motion")
    ap.add_argument("--frame", type=int, help="moving piece: use this frame instead of the automatic choice")
    ap.add_argument("--name")
    ap.add_argument("--image-endpoint")
    ap.add_argument("--video-endpoint")
    ap.add_argument("--brief-file", help="JSON file with the brief (stage brief)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=None, help="seconds to wait for a fal request")
    a = ap.parse_args(argv)
    try:
        if a.stage == "init":
            if not a.piece:
                raise JobError("--piece is required for init")
            out = stages.init(a.job, a.piece, a.prompt, scene=a.scene, energy=a.energy, aspect=a.aspect,
                              resolution=a.resolution, duration=a.duration, echo=a.echo, frame=a.frame,
                              name=a.name, image_endpoint=a.image_endpoint, video_endpoint=a.video_endpoint)
        else:
            job = Job(a.job)
            if a.stage == "brief":
                if not a.brief_file:
                    raise JobError("--brief-file is required for brief")
                out = stages.brief(job, json.loads(Path(a.brief_file).read_text()))
            elif a.stage == "still":
                out = stages.still(job, attempts=a.attempts, timeout=a.timeout or 600)
            elif a.stage == "video":
                out = stages.video(job, timeout=a.timeout or 900)
            else:
                out = getattr(stages, a.stage)(job)
        print(json.dumps(out, indent=2, default=float))
        return 0
    except stages.GateFailed as e:
        print(json.dumps({"error": "fidelity-gate", "detail": json.loads(str(e))}, indent=2))
        return 2
    except fal.FalTimeout as e:
        print(json.dumps({"pending": str(e)}))
        return 3
    except (JobError, fal.FalError, ValueError) as e:
        print(json.dumps({"error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
