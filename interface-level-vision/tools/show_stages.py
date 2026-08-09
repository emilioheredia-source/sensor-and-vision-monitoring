"""Save one picture per step, so the whole process can be looked at.

    pixi run python tools/show_stages.py "tests/third mask_cropped.png"

Writes a numbered sequence into a folder beside the image. Every step is a
picture of what the code had at that point, rather than a description of it.
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

ORANGE, GREEN, CYAN, MAGENTA = ((40, 160, 250), (60, 220, 60), (250, 200, 40), (220, 60, 220))


def save(fig, out_dir, number, name):
    path = out_dir / f"{number:02d}_{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"   {path.name}")
    return path


def picture(bgr, title, out_dir, number, name):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return save(fig, out_dir, number, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    out_dir = args.out_dir or args.image.with_name(args.image.stem + "_stages")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}")

    picture(bgr, "1. the frame as captured", out_dir, 1, "frame")

    # --- what the mark finder thresholds on -------------------------------
    opts = slot_module.MARK_DEFAULTS
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (opts["blur"], opts["blur"]), 0)
    binary = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, opts["block"], opts["offset"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (opts["open_kernel"],) * 2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(binary, cmap="gray")
    ax.set_title(f"2. brighter than the local mean\n"
                 f"(adaptive threshold, {opts['block']} px window)", fontsize=10)
    ax.axis("off")
    save(fig, out_dir, 2, "threshold_for_marks")

    # --- the bars -----------------------------------------------------------
    marks = slot_module.find_marks(bgr)
    canvas = bgr.copy()
    for mark in marks:
        cv2.drawContours(canvas, [mark["hull"]], -1, ORANGE, 3)
    picture(canvas, f"3. shapes far wider than they are tall: {len(marks)} bars",
            out_dir, 3, "marks")

    if len(marks) != 2:
        raise SystemExit(f"{len(marks)} marks found, need two to continue")

    # --- sub-pixel bar centres ---------------------------------------------
    rough = sorted(marks, key=lambda m: m["centre"][1])
    marks = slot_module.refine_bar_centres(bgr, marks)
    fine = sorted(marks, key=lambda m: m["centre"][1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    grey_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for ax, before, after, label in zip(axes, rough, fine, ("upper bar", "lower bar")):
        x, y, w, h = before["bbox"]
        band = max(int(1.6 * h), 4)
        y0, y1 = max(y - band, 0), min(y + h + band, grey_f.shape[0])
        half = max(int(0.55 * w / 2), 1)
        cx = int(round(before["centre"][0]))
        rows = grey_f[y0:y1, max(cx - half, 0):cx + half + 1].mean(axis=1)
        ax.plot(np.arange(y0, y1), rows, color="steelblue", linewidth=1.8)
        ax.axvline(before["centre"][1], color="grey", linestyle=":",
                   label=f"box centre {before['centre'][1]:.2f}")
        ax.axvline(after["centre"][1], color="crimson", linestyle="--",
                   label=f"centroid {after['centre'][1]:.2f}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("row")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("4. each bar's centre, refined to sub-pixel from its brightness peak",
                 fontsize=10)
    save(fig, out_dir, 4, "bar_centres")

    # --- the window, placed rather than found -------------------------------
    window = slot_module.window_from_marks(marks)
    x, y, w, h = window["bbox"]
    x0, x1 = window["strip"]

    canvas = bgr.copy()
    cv2.rectangle(canvas, (x, y), (x + w, y + h), GREEN, 2)
    cv2.rectangle(canvas, (x0, y), (x1, y + h), MAGENTA, 2)

    # Both bar centres, because those two lines are the whole calibration: the
    # distance between them is a printed 117 mm and everything else follows.
    # The window's own size is derived from that and never used to find it,
    # since its edges are soft and depend on how the opening was cut.
    top_y = int(round(fine[0]["centre"][1]))
    bottom_y = int(round(fine[-1]["centre"][1]))
    separation = fine[-1]["centre"][1] - fine[0]["centre"][1]

    for row, tag in ((top_y, "upper bar"), (bottom_y, "lower bar = scale zero")):
        cv2.line(canvas, (x - 55, row), (x + w + 55, row), CYAN, 2)
        cv2.putText(canvas, tag, (x + w + 6, row - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1)

    cv2.arrowedLine(canvas, (x - 45, top_y), (x - 45, bottom_y), CYAN, 2, tipLength=0.02)
    cv2.arrowedLine(canvas, (x - 45, bottom_y), (x - 45, top_y), CYAN, 2, tipLength=0.02)
    cv2.putText(canvas, "117 mm", (x - 42, (top_y + bottom_y) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1)

    picture(canvas, f"5. calibration is bar to bar: {separation:.1f} px = 117 mm, "
                    f"so {window['mm_per_px']:.4f} mm/px\n"
                    f"window placed from that (green), profile strip (magenta)",
            out_dir, 5, "window_and_strip")

    # --- the profile and the boundaries -------------------------------------
    reading = interface_module.find_interfaces(bgr, window)
    rows = np.arange(h)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    axes[0].plot(reading["profile"], rows, color="lightsteelblue", linewidth=1)
    axes[0].plot(reading["smoothed"], rows, color="steelblue", linewidth=2)
    axes[0].set_xlabel(interface_module.DEFAULTS["channel"])
    axes[0].set_ylabel("row down the window")
    axes[0].set_title("6. the strip averaged across, one value per row", fontsize=10)
    axes[1].plot(reading["slope"], rows, color="crimson", linewidth=1.5)
    axes[1].axvline(0, color="grey", linewidth=0.8)
    axes[1].set_xlabel("change per row")
    axes[1].set_title("its derivative: a boundary is a peak", fontsize=10)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        for boundary in reading["interfaces"]:
            ax.axhline(boundary["row"], color="darkorange", linestyle="--", linewidth=1.4)
    save(fig, out_dir, 6, "profile")

    # --- the answer ---------------------------------------------------------
    canvas = bgr.copy()
    cv2.rectangle(canvas, (x0, y), (x1, y + h), MAGENTA, 1)
    for boundary in reading["interfaces"]:
        row = y + boundary["row"]
        cv2.line(canvas, (x - 30, row), (x + w + 30, row), ORANGE, 2)
        cv2.putText(canvas, f"{boundary['mm_on_scale']:.1f} mm",
                    (x + w + 36, row + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 1)
    layer = reading["layer"]
    caption = "7. the reading"
    if layer and "thickness_mm" in layer:
        caption += (f": middle layer {layer['thickness_mm']:.1f} mm thick, "
                    f"centred at {layer['centre_mm']:.1f} mm")
    picture(canvas, caption, out_dir, 7, "reading")

    print("\n" + caption.split(". ", 1)[1] if ". " in caption else caption)
    for boundary in reading["interfaces"]:
        print(f"   {boundary['name']:<20} {boundary['mm_on_scale']:6.1f} mm")


if __name__ == "__main__":
    main()
