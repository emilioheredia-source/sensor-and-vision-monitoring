"""Show what each colour channel does across the boundaries.

    pixi run python tools/show_channels.py data/lighting/backlit_cropped.png

The camera widget used to set the rig up plots one curve per channel along its
region of interest, and this is that view made reproducible: the red, green and
blue profiles down the same strip the detector measures, and underneath them the
blue fraction it actually uses.

The point of the picture is that the two liquids are close in brightness and far
apart in colour, so a boundary that is faint in grey is obvious in blue.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import slot as slot_module
import interface as interface_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    marks = slot_module.refine_bar_centres(bgr, slot_module.find_marks(bgr))
    window = slot_module.window_from_marks(marks)
    if window is None:
        raise SystemExit("could not place the window from the marks")

    x, y, w, h = window["bbox"]
    x0, x1 = window["strip"]
    reading = interface_module.find_interfaces(bgr, window)
    depth = np.arange(h) * window["mm_per_px"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5),
                             gridspec_kw={"width_ratios": [1.1, 1, 1]})

    canvas = bgr.copy()
    cv2.rectangle(canvas, (x0, y), (x1, y + h), (220, 60, 220), 2)
    for boundary in reading["interfaces"]:
        cv2.line(canvas, (x - 20, y + boundary["row"]),
                 (x + w + 20, y + boundary["row"]), (40, 160, 250), 2)
    axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[0].set_title("the strip the profile is taken over", fontsize=10)
    axes[0].axis("off")

    for index, (name, colour) in enumerate((("blue", "tab:blue"),
                                            ("green", "tab:green"),
                                            ("red", "tab:red"))):
        profile, _ = interface_module.column_profile(bgr, window, {"channel": name})
        axes[1].plot(profile, depth, color=colour, linewidth=1.8, label=name)
    grey, _ = interface_module.column_profile(bgr, window, {"channel": "gray"})
    axes[1].plot(grey, depth, color="black", linewidth=1.4, linestyle="--", label="grey")
    axes[1].set_xlabel("mean value across the strip")
    axes[1].set_ylabel("mm down from the top of the window")
    axes[1].set_title("each channel down the strip", fontsize=10)

    axes[2].plot(reading["smoothed"], depth, color="tab:blue", linewidth=2)
    axes[2].set_xlabel("blue / (R+G+B)")
    axes[2].set_title("blue as a fraction of the total,\nwhich is what gets measured",
                      fontsize=10)

    for ax in axes[1:]:
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        for boundary in reading["interfaces"]:
            ax.axhline(boundary["row"] * window["mm_per_px"], color="darkorange",
                       linestyle="--", linewidth=1.4)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    out = args.out or args.image.with_name(args.image.stem + "_channels.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    for name in ("gray", "blue", "green", "red", "blue_fraction"):
        profile, _ = interface_module.column_profile(bgr, window, {"channel": name})
        smoothed = interface_module._smooth(profile, 9)
        slope = np.gradient(smoothed)
        rows = [b["row"] for b in reading["interfaces"]]
        span = np.nanmax(smoothed) - np.nanmin(smoothed)
        steps = "  ".join(f"{100 * abs(slope[r]) / span:5.2f}%" for r in rows)
        print(f"   {name:<14} step at each boundary, as a share of its own range: {steps}")


if __name__ == "__main__":
    main()
