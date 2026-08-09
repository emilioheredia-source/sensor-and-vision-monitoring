"""Find the window, read the levels in it, and draw the result.

    pixi run python tools/try_measure.py frame_cropped.png --window-height-mm 100
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import slot as slot_module
import interface as interface_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--window-height-mm", type=float, default=100.0,
                        help="the printed height of the opening, which sets the scale")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    # Preferred route: the mask is printed, so two bars place the window by
    # construction. Segmenting the opening is the fallback, for a frame where
    # the marks cannot be found.
    sub = bgr
    marks = slot_module.find_marks(bgr)
    marks = slot_module.refine_bar_centres(bgr, marks)
    found = slot_module.window_from_marks(marks)

    if found is not None:
        mm_per_px = found["mm_per_px"]
        across = found["mm_per_px_across"]
        print(f"{len(marks)} marks found, window placed from them")
        print(f"   {mm_per_px:.4f} mm/px from the bar separation, {across:.4f} from "
              f"the bar length, differing by "
              f"{100 * abs(mm_per_px - across) / mm_per_px:.1f}%")
    else:
        print(f"{len(marks)} marks, not the two needed, so segmenting the opening")
        found = slot_module.find_slot(bgr)["slot"]
        if found is None:
            raise SystemExit("no window found either way, nothing to measure")
        mm_per_px = args.window_height_mm / found["bbox"][3]

    x, y, w, h = found["bbox"]
    reading = interface_module.find_interfaces(sub, found, args.window_height_mm)
    print(f"window {w}x{h} px at ({x},{y}), {mm_per_px:.4f} mm per pixel\n")
    if not reading["interfaces"]:
        print("no boundary found in the profile")

    for boundary in reading["interfaces"]:
        print(f"{boundary['name']:<48} {boundary['mm_on_scale']:6.1f} mm   "
              f"step {boundary['step']:+7.4f}")

    layer = reading["layer"]
    if layer and "thickness_mm" in layer:
        print(f"\nmiddle layer {layer['thickness_mm']:.1f} mm thick, centred at "
              f"{layer['centre_mm']:.1f} mm, running {layer['bottom_mm']:.1f} to "
              f"{layer['top_mm']:.1f} mm")
    print("\nheights are millimetres above the lower bar, "
          "the same zero as the printed scale")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 1, 1]})

    canvas = sub.copy()
    cv2.drawContours(canvas, [found["hull"]], -1, (60, 220, 60), 2)
    for boundary in reading["interfaces"]:
        row = y + boundary["row"]
        cv2.line(canvas, (x - 15, row), (x + w + 15, row), (40, 160, 250), 2)
    axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[0].set_title("window in green, boundaries found in orange", fontsize=10)
    axes[0].axis("off")

    rows = np.arange(h)
    axes[1].plot(reading["profile"], rows, color="lightsteelblue", linewidth=1,
                 label="column average")
    axes[1].plot(reading["smoothed"], rows, color="steelblue", linewidth=2,
                 label="smoothed")
    axes[1].set_ylabel("row down the window")
    axes[1].set_xlabel(interface_module.DEFAULTS["channel"])
    axes[1].set_title("the profile", fontsize=10)

    axes[2].plot(reading["slope"], rows, color="crimson", linewidth=1.5)
    axes[2].axvline(0, color="grey", linewidth=0.8)
    axes[2].set_xlabel("change per row")
    axes[2].set_title("its derivative, a boundary is a peak", fontsize=10)

    for ax in axes[1:]:
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        for boundary in reading["interfaces"]:
            ax.axhline(boundary["row"], color="darkorange", linestyle="--", linewidth=1.4)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = args.out or args.image.with_name(args.image.stem + "_measured.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
