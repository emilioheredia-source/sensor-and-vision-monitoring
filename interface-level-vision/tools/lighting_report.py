"""Turn the accumulated lighting trials into a comparison figure and a table.

    pixi run python tools/lighting_report.py

Reads data/lighting/readings.csv and the frames beside it, and writes a picture
of every frame with its boundaries drawn, the profiles overlaid, and a markdown
table of the numbers.
"""

import argparse
import csv
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


def spread(values):
    """Range and standard deviation of a column, as a pair of strings."""
    values = np.array(values, dtype=float)
    return values.max() - values.min(), values.std(ddof=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=HERE.parent / "data" / "lighting")
    args = parser.parse_args()

    with (args.dir / "readings.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no readings recorded yet")

    # --- every frame, with what was found on it ----------------------------
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 6.4))
    axes = np.atleast_1d(axes)
    profiles = []

    for ax, row in zip(axes, rows):
        bgr = cv2.imread(str(args.dir / row["image"]))
        marks = slot_module.refine_bar_centres(bgr, slot_module.find_marks(bgr))
        window = slot_module.window_from_marks(marks)
        reading = interface_module.find_interfaces(bgr, window)
        profiles.append((row["label"], reading, window))

        x, y, w, h = window["bbox"]
        canvas = bgr.copy()
        for boundary in reading["interfaces"]:
            cv2.line(canvas, (x - 20, y + boundary["row"]),
                     (x + w + 20, y + boundary["row"]), (40, 160, 250), 2)
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row['label']}\n{row['upper_mm']} / {row['lower_mm']} mm",
                     fontsize=9)
        ax.axis("off")

    fig.suptitle("The same rig under five lightings, nothing moved but the lamp",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.dir / "comparison_frames.png", dpi=120)
    plt.close(fig)

    # --- the profiles on one axis ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    for label, reading, window in profiles:
        depth = np.arange(len(reading["smoothed"])) * window["mm_per_px"]
        axes[0].plot(reading["smoothed"], depth, linewidth=1.8, label=label)
        axes[1].plot(reading["slope"], depth, linewidth=1.4, label=label)

    axes[0].set_xlabel(interface_module.DEFAULTS["channel"])
    axes[0].set_ylabel("mm down from the top of the window")
    axes[0].set_title("the profile under each lighting", fontsize=10)
    axes[1].set_xlabel("change per row")
    axes[1].set_title("its derivative: a boundary is a peak", fontsize=10)
    axes[1].axvline(0, color="grey", linewidth=0.8)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.dir / "comparison_profiles.png", dpi=120)
    plt.close(fig)

    # --- the table ----------------------------------------------------------
    columns = [("upper_mm", "upper mm"), ("lower_mm", "lower mm"),
               ("thickness_mm", "layer mm"), ("centre_mm", "centre mm"),
               ("upper_step", "upper step"), ("lower_step", "lower step"),
               ("cross_check_pct", "scale check %")]

    lines = ["| lighting | " + " | ".join(name for _, name in columns) + " |",
             "|---" * (len(columns) + 1) + "|"]
    for row in rows:
        lines.append("| " + row["label"] + " | "
                     + " | ".join(row[key] for key, _ in columns) + " |")

    lines.append("")
    lines.append("| measurement | range | standard deviation |")
    lines.append("|---|---|---|")
    for key, name in columns[:4]:
        rng, sd = spread([r[key] for r in rows])
        lines.append(f"| {name} | {rng:.2f} mm | {sd:.2f} mm |")

    table = "\n".join(lines)
    (args.dir / "readings.md").write_text(table + "\n")
    print(table)
    print(f"\nwrote comparison_frames.png, comparison_profiles.png and readings.md "
          f"into {args.dir}")


if __name__ == "__main__":
    main()
