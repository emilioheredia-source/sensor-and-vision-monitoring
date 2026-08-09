"""Capture one frame under the current lighting, measure it, and record the row.

    pixi run python tools/lighting_trial.py --label "lamp in front"

Nothing about the rig may change between shots except the light. The liquid is
not touched and the camera is not moved, so every reading should come back the
same, and whatever spread appears is what the lighting costs.

Frames, stage pictures and one row per shot go into data/lighting/, and
lighting_report.py turns the accumulated rows into a comparison.
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import slot as slot_module
import interface as interface_module

FIELDS = ["label", "captured", "image", "marks", "mm_per_px", "mm_per_px_across",
          "cross_check_pct", "upper_mm", "lower_mm", "thickness_mm", "centre_mm",
          "upper_step", "lower_step"]


def measure(bgr):
    """Run the whole chain on one frame and return a row, or a reason it failed."""
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
        "marks": len(marks),
        "mm_per_px": round(scale, 5),
        "mm_per_px_across": round(across, 5),
        "cross_check_pct": round(100 * abs(scale - across) / scale, 2),
        "upper_mm": round(upper["mm_on_scale"], 2),
        "lower_mm": round(lower["mm_on_scale"], 2),
        "thickness_mm": round(layer["thickness_mm"], 2),
        "centre_mm": round(layer["centre_mm"], 2),
        "upper_step": round(upper["step"], 5),
        "lower_step": round(lower["step"], 5),
    }, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="where the light is, in words")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--settle", type=float, default=12.0)
    parser.add_argument("--dir", type=Path, default=HERE.parent / "data" / "lighting")
    args = parser.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in args.label).strip("_").lower()
    raw = args.dir / f"{slug}.png"

    subprocess.run([sys.executable, str(HERE / "capture.py"),
                    "--camera", str(args.camera), "--settle", str(args.settle),
                    "--out", str(raw)], check=True)

    bgr = cv2.imread(str(raw))
    cropped = args.dir / f"{slug}_cropped.png"
    marks = slot_module.find_marks(bgr)
    if len(marks) >= 2:
        sys.path.insert(0, str(HERE))
        from crop_to_gauge import gauge_crop
        x, y, w, h = gauge_crop(bgr, marks)
        bgr = bgr[y:y + h, x:x + w]
        cv2.imwrite(str(cropped), bgr)

    row, problem = measure(bgr)
    if problem is not None:
        raise SystemExit(f"could not measure this frame: {problem}")

    row.update({"label": args.label,
                "captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": cropped.name})

    csv_path = args.dir / "readings.csv"
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    subprocess.run([sys.executable, str(HERE / "show_stages.py"), str(cropped)],
                   check=True, stdout=subprocess.DEVNULL)

    print(f"\n{args.label}")
    print(f"   upper boundary {row['upper_mm']:6.1f} mm      step {row['upper_step']:+.4f}")
    print(f"   lower boundary {row['lower_mm']:6.1f} mm      step {row['lower_step']:+.4f}")
    print(f"   layer {row['thickness_mm']:.1f} mm thick, centred {row['centre_mm']:.1f} mm")
    print(f"   scale {row['mm_per_px']:.4f} mm/px, cross-check "
          f"{row['cross_check_pct']:.1f}% apart")
    print(f"   recorded in {csv_path}")


if __name__ == "__main__":
    main()
