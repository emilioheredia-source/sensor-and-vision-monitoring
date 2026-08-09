"""Shoot several frames under one fixed lighting and measure each.

    pixi run python tools/repeatability_trial.py --count 10

The five-lighting series changes the lamp between shots, so its spread mixes
what the lighting did with whatever the detector does from one shot to the
next. This holds the lighting fixed and takes several shots anyway, which
isolates the second thing: sensor noise, auto-exposure settling, and anything
else that varies between two photos of an unchanged scene.

The camera is opened once and read repeatedly rather than reopened per shot,
since reopening means renegotiating the connection each time and the point is
shots that are close together with nothing touched in between.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import slot as slot_module
import interface as interface_module
from capture import open_camera, looks_like_no_signal

sys.path.insert(0, str(HERE))
from crop_to_gauge import gauge_crop

FIELDS = ["shot", "captured", "image", "upper_mm", "lower_mm", "thickness_mm",
          "centre_mm", "mm_per_px", "cross_check_pct"]


def measure(bgr):
    marks = slot_module.refine_bar_centres(bgr, slot_module.find_marks(bgr))
    window = slot_module.window_from_marks(marks)
    if window is None:
        return None, f"{len(marks)} marks found, need exactly two"
    reading = interface_module.find_interfaces(bgr, window)
    if len(reading["interfaces"]) < 2:
        return None, f"{len(reading['interfaces'])} boundaries found, need two"
    upper, lower = reading["interfaces"][0], reading["interfaces"][-1]
    layer = reading["layer"]
    scale, across = window["mm_per_px"], window["mm_per_px_across"]
    return {
        "upper_mm": round(upper["mm_on_scale"], 2),
        "lower_mm": round(lower["mm_on_scale"], 2),
        "thickness_mm": round(layer["thickness_mm"], 2),
        "centre_mm": round(layer["centre_mm"], 2),
        "mm_per_px": round(scale, 5),
        "cross_check_pct": round(100 * abs(scale - across) / scale, 2),
    }, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between shots")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--settle", type=float, default=12.0)
    parser.add_argument("--dir", type=Path, default=HERE.parent / "data" / "repeatability")
    args = parser.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)

    print(f"opening camera {args.camera}")
    capture = open_camera(args.camera)
    if capture is None:
        raise SystemExit(f"could not read from camera {args.camera}")

    print(f"reading for up to {args.settle:.0f} s, waiting for a real scene")
    deadline = time.time() + args.settle
    while time.time() < deadline:
        ok, candidate = capture.read()
        if ok and candidate is not None and looks_like_no_signal(candidate) is None:
            break
    else:
        capture.release()
        raise SystemExit("no real scene arrived before the camera settled")

    rows = []
    for shot in range(1, args.count + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            print(f"   shot {shot}: no frame, skipping")
            continue

        raw = args.dir / f"shot_{shot:02d}.png"
        cv2.imwrite(str(raw), frame)

        marks = slot_module.find_marks(frame)
        cropped = raw
        if len(marks) >= 2:
            x, y, w, h = gauge_crop(frame, marks)
            frame = frame[y:y + h, x:x + w]
            cropped = args.dir / f"shot_{shot:02d}_cropped.png"
            cv2.imwrite(str(cropped), frame)

        row, problem = measure(frame)
        if problem is not None:
            print(f"   shot {shot}: could not measure, {problem}")
            time.sleep(args.interval)
            continue

        row.update({"shot": shot,
                    "captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image": cropped.name})
        rows.append(row)
        print(f"   shot {shot}: upper {row['upper_mm']:.2f}  lower {row['lower_mm']:.2f}  "
              f"centre {row['centre_mm']:.2f}  thickness {row['thickness_mm']:.2f}")
        time.sleep(args.interval)

    capture.release()

    if not rows:
        raise SystemExit("no shot could be measured")

    csv_path = args.dir / "readings.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} of {args.count} shots measured, one lighting, nothing touched\n")
    print(f"{'measurement':<14} {'mean':>8} {'range':>8} {'std dev':>8}")
    for key, name in (("upper_mm", "upper"), ("lower_mm", "lower"),
                      ("thickness_mm", "thickness"), ("centre_mm", "centre")):
        values = np.array([r[key] for r in rows], dtype=float)
        print(f"{name:<14} {values.mean():8.2f} {values.max() - values.min():8.2f} "
              f"{values.std(ddof=1):8.2f}")

    print(f"\nrecorded in {csv_path}")


if __name__ == "__main__":
    main()
