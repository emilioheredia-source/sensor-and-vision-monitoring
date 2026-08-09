"""Crop a frame down to the gauge, the way a real installation would be aimed.

    pixi run python tools/crop_to_gauge.py frame.png

The bench photos take in a whole desk. A camera bolted in front of a sight glass
would not, so cropping to the mask first is closer to the real case as well as
removing clutter that was never going to be there.

The crop is placed from the calibration bars, since they are found before
anything else is. It is wider on the side the operator scale sits, so the scale
stays in the picture.
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import slot as slot_module


def gauge_crop(bgr, marks, left=1.0, right=1.9, above=0.35, below=0.35):
    """Box around the bars, in multiples of the bar width and separation."""
    xs = [m["bbox"][0] for m in marks] + [m["bbox"][0] + m["bbox"][2] for m in marks]
    ys = [m["centre"][1] for m in marks]
    bar_width = max(m["bbox"][2] for m in marks)
    separation = max(ys) - min(ys)

    x0 = int(min(xs) - left * bar_width)
    x1 = int(max(xs) + right * bar_width)
    y0 = int(min(ys) - above * separation * 0.5)
    y1 = int(max(ys) + below * separation * 0.5)

    height, width = bgr.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, width), min(y1, height)
    return x0, y0, x1 - x0, y1 - y0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    marks = slot_module.find_marks(bgr)
    if len(marks) < 2:
        raise SystemExit(f"found {len(marks)} marks, need at least two to place a crop")

    x, y, w, h = gauge_crop(bgr, marks)
    out = args.out or args.image.with_name(args.image.stem + "_cropped.png")
    cv2.imwrite(str(out), bgr[y:y + h, x:x + w])

    print(f"{len(marks)} marks found")
    print(f"frame {bgr.shape[1]}x{bgr.shape[0]} -> crop ({x},{y}) {w}x{h}, "
          f"{100 * w * h / (bgr.shape[0] * bgr.shape[1]):.0f}% of it")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
