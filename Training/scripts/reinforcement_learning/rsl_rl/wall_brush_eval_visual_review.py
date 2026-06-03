#!/usr/bin/env python3
"""Read or update wall-brush eval JSON visual-review metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
MODULE_CANDIDATES = [
    SCRIPT_PATH.parents[3]
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "interactive_motion_tracking"
    / "g1"
    / "wall_brush_success.py",
    SCRIPT_PATH.parents[1]
    / "vendor"
    / "DreamControl"
    / "Training"
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "interactive_motion_tracking"
    / "g1"
    / "wall_brush_success.py",
]
MODULE_PATH = next((path for path in MODULE_CANDIDATES if path.exists()), MODULE_CANDIDATES[0])
spec = importlib.util.spec_from_file_location("wall_brush_success", MODULE_PATH)
wall_brush_success = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wall_brush_success
assert spec.loader is not None
spec.loader.exec_module(wall_brush_success)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _suspicious_prior_id(payload: dict) -> int:
    return int(payload.get("summary", {}).get("suspicious_prior_id", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--view_prefix", default=None)
    parser.add_argument("--video_length", type=int, default=220)
    parser.add_argument("--print_prior", action="store_true")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    payload = _load(eval_path)
    prior_id = _suspicious_prior_id(payload)
    if args.print_prior:
        print(prior_id)
    if args.update:
        view_prefix = args.view_prefix or eval_path.stem
        entries = wall_brush_success.build_wall_brush_visual_review_entries(
            args.checkpoint,
            prior_id,
            view_prefix,
            video_length=args.video_length,
        )
        payload["visual_review"] = entries
        payload.setdefault("summary", {})["visual_review"] = entries
        eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
