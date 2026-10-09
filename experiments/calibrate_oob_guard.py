#!/usr/bin/env python3
"""Measure the ingest notebook's out-of-body detector on folders of real frames.

Point it at frames you know are in-body to get the false-reject rate per folder
(target: <= ~2%), or at known out-of-body frames to get the catch rate. Runs the
detector straight from ingest_notebooks/smart_frames/notebook.py, locally only.

    python3 experiments/calibrate_oob_guard.py <dir> [<dir> ...]

Each <dir> is searched recursively for .jpg/.png; results are grouped by the
immediate sub-folder of <dir>.
"""
import collections
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_oob_guard import _nb  # noqa: E402  (loads the detector from the notebook)


def main(dirs):
    detect = _nb["_is_oob_frame"]
    print(f"thresholds: TISSUE_HUE_MIN={_nb['TISSUE_HUE_MIN']} "
          f"NEAR_WHITE_BLUE_MAX={_nb['NEAR_WHITE_BLUE_MAX']} LIT_MIN={_nb['LIT_MIN']}")
    for d in map(Path, dirs):
        groups = collections.defaultdict(list)
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rel = p.relative_to(d)
                groups[rel.parts[0] if len(rel.parts) > 1 else "."].append(p)
        for name, paths in sorted(groups.items()):
            flagged = 0
            for p in paths:
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if max(rgb.shape[:2]) > 640:  # match the notebook's analysis resolution
                    sc = 640 / max(rgb.shape[:2])
                    rgb = cv2.resize(rgb, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                flagged += detect(rgb)[0]
            print(f"{d.name}/{name:30s} n={len(paths):4d}  flagged OOB={100 * flagged / len(paths):5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
