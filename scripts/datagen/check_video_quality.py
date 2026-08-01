"""
QC script for generated plant motion videos.

Detects three failure modes:
  1. Static  — video has almost no motion
  2. Camera  — background region moves (camera pan/orbit instead of plant motion)
  3. Drift   — plant appearance changes drastically between first and last frame

Usage:
    python scripts/check_video_quality.py --video_dir data/plants_video
    python scripts/check_video_quality.py --video_dir data/plants_video --output qc_report.json
    python scripts/check_video_quality.py --video_dir data/plants_video --failed_only
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Thresholds ─────────────────────────────────────────────────────────────
# Tune these based on empirical results; current values from smoke-test
# observations on 33-frame 16fps Wan2.1 outputs.
STATIC_THRESH   = 0.06  # mean optical-flow magnitude below this → static
                        # calibrated on smoke-test: s43_wind_light=0.121 (good),
                        # s42_wind_light=0.035 (bad/static)
CAMERA_THRESH   = 0.6   # background/foreground flow ratio above this → camera motion
DRIFT_THRESH    = 0.30  # mean absolute pixel diff (0-1) first↔last → structural drift
                        # calibrated: s42_wind_intense=0.433 (bad), s43_wind_intense=0.191 (good)


def read_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def compute_flow_magnitude(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0,
    )
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def corner_mask(h: int, w: int, margin: float = 0.15) -> np.ndarray:
    """Boolean mask covering the four corner rectangles (background region)."""
    mh, mw = int(h * margin), int(w * margin)
    mask = np.zeros((h, w), dtype=bool)
    mask[:mh, :mw] = True
    mask[:mh, -mw:] = True
    mask[-mh:, :mw] = True
    mask[-mh:, -mw:] = True
    return mask


def check_video(video_path: Path) -> dict:
    frames = read_frames(video_path)
    if len(frames) < 4:
        return {"path": str(video_path), "error": f"only {len(frames)} frames", "flag": "error"}

    h, w = frames[0].shape[:2]
    bg_mask = corner_mask(h, w)

    motion_scores = []
    bg_ratios = []

    for i in range(len(frames) - 1):
        mag = compute_flow_magnitude(frames[i], frames[i + 1])
        mean_global = mag.mean()
        mean_bg     = mag[bg_mask].mean()

        motion_scores.append(float(mean_global))
        # bg ratio: how much of the motion is in the background corners
        bg_ratios.append(float(mean_bg / (mean_global + 1e-6)))

    mean_motion = float(np.mean(motion_scores))
    mean_bg_ratio = float(np.mean(bg_ratios))

    # Structural drift: first vs last frame pixel diff (normalised)
    first = frames[0].astype(np.float32) / 255.0
    last  = frames[-1].astype(np.float32) / 255.0
    drift = float(np.abs(first - last).mean())

    flags = []
    if mean_motion < STATIC_THRESH:
        flags.append("static")
    if mean_bg_ratio > CAMERA_THRESH:
        flags.append("camera_motion")
    if drift > DRIFT_THRESH:
        flags.append("drift")

    return {
        "path":         str(video_path),
        "n_frames":     len(frames),
        "mean_motion":  round(mean_motion, 4),
        "mean_bg_ratio":round(mean_bg_ratio, 4),
        "drift":        round(drift, 4),
        "flags":        flags,
        "ok":           len(flags) == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="data/plants_video")
    parser.add_argument("--output", type=str, default=None,
                        help="Save full JSON report to this path")
    parser.add_argument("--failed_only", action="store_true",
                        help="Print only flagged videos")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="JSON string to override thresholds, e.g. "
                             '\'{"static": 0.4, "camera": 0.5, "drift": 0.3}\'')
    args = parser.parse_args()

    global STATIC_THRESH, CAMERA_THRESH, DRIFT_THRESH
    if args.thresholds:
        overrides = json.loads(args.thresholds)
        STATIC_THRESH  = overrides.get("static",  STATIC_THRESH)
        CAMERA_THRESH  = overrides.get("camera",  CAMERA_THRESH)
        DRIFT_THRESH   = overrides.get("drift",   DRIFT_THRESH)

    video_dir = Path(args.video_dir)
    video_paths = sorted(video_dir.rglob("motion.mp4"))

    if not video_paths:
        print(f"No motion.mp4 files found in {video_dir}")
        sys.exit(1)

    print(f"Checking {len(video_paths)} videos in {video_dir} ...")
    print(f"Thresholds: static<{STATIC_THRESH}  camera_bg>{CAMERA_THRESH}  drift>{DRIFT_THRESH}\n")

    results = []
    n_ok = n_flag = n_err = 0

    for vp in video_paths:
        r = check_video(vp)
        results.append(r)

        if "error" in r:
            n_err += 1
            tag = "[ERROR ]"
        elif r["ok"]:
            n_ok += 1
            tag = "[OK    ]"
            if args.failed_only:
                continue
        else:
            n_flag += 1
            tag = "[FLAG  ]"

        flag_str = ", ".join(r.get("flags", [])) or r.get("error", "")
        print(
            f"{tag} {Path(r['path']).parent.name:60s}  "
            f"motion={r.get('mean_motion', 0):5.3f}  "
            f"bg_ratio={r.get('mean_bg_ratio', 0):5.3f}  "
            f"drift={r.get('drift', 0):5.3f}  "
            f"{flag_str}"
        )

    print(f"\n── Summary ────────────────────────────────")
    print(f"  Total   : {len(results)}")
    print(f"  OK      : {n_ok}")
    print(f"  Flagged : {n_flag}  ({100*n_flag/max(len(results),1):.1f}%)")
    print(f"  Errors  : {n_err}")

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w") as f:
            json.dump({"thresholds": {
                "static": STATIC_THRESH,
                "camera": CAMERA_THRESH,
                "drift":  DRIFT_THRESH,
            }, "results": results}, f, indent=2)
        print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
