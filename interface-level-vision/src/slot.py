"""Find the sight-glass opening in a frame, and give back a region of interest.

Plain OpenCV: adaptive threshold, trace the outlines, and keep the one whose
shape matches an obround. No training, no model file, and every rejection can be
explained by pointing at a number.

Four decisions do the work, and each was arrived at by trying the obvious thing
first and watching it fail on a real frame of the rig.

**Adaptive threshold rather than Otsu.** A global threshold cannot cope with a
desk lamp in one corner and a shadowed mask in the middle. On the test frame it
merged the window into the lit background and the window stopped existing as a
separate outline at all.

**Reject anything touching the frame edge.** This is what actually separates the
window from the clutter, and brightness never could, since a lamp and a sheet of
paper in the background are both brighter than the lit window. The furniture and
lighting that beat it on brightness run off the edge of the picture. The window
does not.

**Take the convex hull of each candidate.** An obround is convex, so its hull is
itself and nothing is lost. What is gained is that a window whose interior came
out ragged, because two liquids threshold differently, is restored to the shape
it actually has. On the test frame this moved the window from an extent of 0.57,
which fails, to 0.907, which is 0.004 away from the exact figure.

**Match the extent to the aspect ratio.** A rectangle fills its bounding box, an
ellipse fills 79% of it, and an obround sits between at a fraction that depends
only on how tall it is. Comparing the two is a scale-free test of shape.
"""

import cv2
import numpy as np


def obround_extent(aspect):
    """Fraction of its bounding box an obround fills, given height / width.

    Area is w*(h - w) + pi*(w/2)**2 for a shape taller than it is wide, hence
    dividing by w*h leaves something that depends only on the aspect ratio.
    """
    return 1.0 - (4.0 - np.pi) / (4.0 * aspect)


DEFAULTS = {
    "blur": 7,
    "block": 151,             # adaptive threshold window, odd, ~the slot's width
    "offset": -10,            # how far above the local mean counts as bright
    "min_area_fraction": 0.004,
    "max_area_fraction": 0.35,
    "min_aspect": 1.4,        # height / width
    "max_aspect": 5.0,
    "extent_tolerance": 0.05,
    "border_margin": 2,
}


def _binary_image(bgr, opts):
    """Locally bright areas as white, cleaned of specks and pinholes."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (opts["blur"], opts["blur"]), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, opts["block"], opts["offset"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return binary


def _touches_border(x, y, w, h, frame_shape, margin):
    height, width = frame_shape[:2]
    return (x <= margin or y <= margin
            or x + w >= width - margin or y + h >= height - margin)


def _describe(contour, frame_shape, margin):
    """Measure one outline through its convex hull.

    Returns None when the outline runs off the edge of the frame, since the
    window never does and the background clutter always does.
    """
    hull = cv2.convexHull(contour)
    x, y, w, h = cv2.boundingRect(hull)
    if w == 0 or h == 0 or _touches_border(x, y, w, h, frame_shape, margin):
        return None

    area = cv2.contourArea(hull)
    if area <= 0:
        return None

    aspect = h / w
    extent = area / (w * h)
    return {
        "hull": hull,
        "bbox": (x, y, w, h),
        "area": area,
        "aspect": aspect,
        "extent": extent,
        "expected_extent": obround_extent(aspect),
        "extent_error": abs(extent - obround_extent(aspect)),
        "rotated": cv2.minAreaRect(hull),
    }


def _rejection(candidate, frame_area, opts):
    """Why this candidate is not the window, or None if it survives."""
    area_fraction = candidate["area"] / frame_area
    if area_fraction < opts["min_area_fraction"]:
        return f"too small, {100 * area_fraction:.2f}% of the frame"
    if area_fraction > opts["max_area_fraction"]:
        return f"too large, {100 * area_fraction:.0f}% of the frame"
    if not opts["min_aspect"] <= candidate["aspect"] <= opts["max_aspect"]:
        return f"aspect {candidate['aspect']:.2f} is outside the expected range"
    if candidate["extent_error"] > opts["extent_tolerance"]:
        return (f"fills {candidate['extent']:.3f} of its box where an obround this "
                f"shape fills {candidate['expected_extent']:.3f}")
    return None


def find_slot(bgr, **overrides):
    """Locate the window opening.

    Returns the winning candidate under "slot", every rejected one with the
    reason it went under "rejected", and the thresholded image under "binary",
    so a failure can be read rather than guessed at. "slot" is None when
    nothing survived.
    """
    opts = {**DEFAULTS, **overrides}
    binary = _binary_image(bgr, opts)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = float(bgr.shape[0] * bgr.shape[1])
    kept, rejected = [], []
    for contour in contours:
        candidate = _describe(contour, bgr.shape, opts["border_margin"])
        if candidate is None:
            continue
        reason = _rejection(candidate, frame_area, opts)
        if reason is None:
            kept.append(candidate)
        elif candidate["area"] / frame_area > 0.002:      # ignore dust
            rejected.append({**candidate, "reason": reason})

    # The window is the biggest thing on the mask shaped like this.
    slot = max(kept, key=lambda c: c["area"]) if kept else None
    return {"slot": slot, "rejected": rejected, "binary": binary}


MARK_DEFAULTS = {
    "blur": 3,
    "block": 51,              # small, because a bar is thin
    "offset": -10,
    "open_kernel": 3,
    "min_area_fraction": 0.0002,
    "max_area_fraction": 0.01,
    "min_elongation": 3.0,    # width / height
    "min_fill": 0.75,         # a bar nearly fills its bounding box
}


def find_marks(bgr, **overrides):
    """Find the calibration bars: small, bright, much wider than they are tall.

    This runs its own pass rather than reusing the one that finds the window,
    because the two are at different scales and the cleanup that helps one
    destroys the other. Opening with a 9 by 9 kernel twice takes about eight
    pixels off every edge, which rescues a ragged window and erases a bar
    thirteen pixels tall. Marks therefore get a smaller threshold window, since
    a bar is thin, and a single pass with a 3 by 3 kernel.
    """
    opts = {**MARK_DEFAULTS, **overrides}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (opts["blur"], opts["blur"]), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, opts["block"], opts["offset"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (opts["open_kernel"], opts["open_kernel"]))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(bgr.shape[0] * bgr.shape[1])

    marks = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        x, y, w, h = cv2.boundingRect(hull)
        area = cv2.contourArea(hull)
        if h == 0 or w == 0 or area <= 0:
            continue
        fraction, elongation, fill = area / frame_area, w / h, area / (w * h)
        if not opts["min_area_fraction"] < fraction < opts["max_area_fraction"]:
            continue
        if elongation < opts["min_elongation"] or fill < opts["min_fill"]:
            continue
        marks.append({"hull": hull, "bbox": (x, y, w, h), "area": area,
                      "elongation": elongation, "fill": fill,
                      "centre": (x + w / 2.0, y + h / 2.0)})
    return marks


def mask_region(marks, pad=0.15, frame_shape=None):
    """Bounding box of the marks, grown a little, as a crop for the mask.

    Aimed at a real installation, where the camera looks at the gauge rather
    than across a room, so cropping to the marks removes the background before
    anything else has to cope with it. A stray bar-shaped find elsewhere on the
    mask, the ruler for instance, does no harm here, since it lies inside the
    region wanted anyway.

    Returns None when there are too few marks to bound anything.
    """
    if len(marks) < 2:
        return None

    xs = [m["bbox"][0] for m in marks] + [m["bbox"][0] + m["bbox"][2] for m in marks]
    ys = [m["bbox"][1] for m in marks] + [m["bbox"][1] + m["bbox"][3] for m in marks]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    dx, dy = int(pad * (x1 - x0)), int(pad * (y1 - y0))
    x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy

    if frame_shape is not None:
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, frame_shape[1]), min(y1, frame_shape[0])
    return (x0, y0, x1 - x0, y1 - y0)


# The printed mask, in millimetres. These are the arguments the sheet was made
# with, so they are known rather than estimated, and they belong in config
# rather than in the logic.
MASK_GEOMETRY_MM = {
    "window_width": 40.0,
    "window_height": 100.0,
    "outline": 3.0,             # white ring around the opening
    "mark_width_fraction": 0.7,  # bar length as a fraction of the window width
    "mark_gap": 4.0,            # clearance between a bar and the ring
    "mark_thickness": 3.0,
}


def window_from_marks(marks, geometry=None):
    """Work out where the window is from the two bars, without looking for it.

    Segmenting the window assumes it is brighter than the mask, and that fails
    the moment the vessel holds something dark: on the second bench frame the
    upper part of the opening is darker than the surround, so thresholding
    returned three fragments instead of one shape.

    None of that is necessary. The mask is printed, hence the window's position
    relative to the bars is known exactly. Two bars give the vertical scale from
    the distance between their centres, the rotation from their own angle, and
    the horizontal reference from their ends, and the opening follows by
    construction.

    This also gives millimetres per pixel as a by-product, which is the number
    every measurement downstream needs.

    Returns None when there are not exactly two bars to work from.
    """
    geometry = {**MASK_GEOMETRY_MM, **(geometry or {})}
    if len(marks) != 2:
        return None

    top, bottom = sorted(marks, key=lambda m: m["centre"][1])
    separation_mm = (geometry["window_height"] + 2 * geometry["outline"]
                     + 2 * geometry["mark_gap"] + geometry["mark_thickness"])
    separation_px = bottom["centre"][1] - top["centre"][1]
    if separation_px <= 0:
        return None

    mm_per_px = separation_mm / separation_px

    # The bars are centred on the opening and narrower than it, so their centre
    # line is the window's centre line, and their length gives the scale
    # sideways as an independent check on the vertical one.
    centre_x = float(np.mean([m["centre"][0] for m in (top, bottom)]))
    bar_length_px = float(np.mean([m["bbox"][2] for m in (top, bottom)]))
    bar_length_mm = geometry["window_width"] * geometry["mark_width_fraction"]

    width_px = geometry["window_width"] / mm_per_px
    height_px = geometry["window_height"] / mm_per_px
    x = centre_x - width_px / 2
    y = top["centre"][1] + (geometry["mark_gap"] + geometry["outline"]
                            + geometry["mark_thickness"] / 2) / mm_per_px

    return {
        "bbox": (int(round(x)), int(round(y)), int(round(width_px)), int(round(height_px))),
        # The lower bar's centre is where the printed scale reads zero, so a row
        # in the image converts to the same number an operator reads off the
        # mask. The window itself starts above it, by the ring plus the gap plus
        # half a bar, which is why a height measured from the window's own edge
        # is not the number on the scale.
        "scale_zero_y": bottom["centre"][1],
        # The strip the profile is taken over: as wide as the bars and no wider.
        # The bars are printed narrower than the opening precisely so that a
        # strip this width runs down inside it without ever touching the rim,
        # hence the column average sees only what is behind the glass.
        "strip": (int(round(max(m["bbox"][0] for m in (top, bottom)))),
                  int(round(min(m["bbox"][0] + m["bbox"][2] for m in (top, bottom))))),
        "mm_per_px": mm_per_px,
        "mm_per_px_across": bar_length_mm / bar_length_px,
        "separation_px": separation_px,
        "hull": np.array([[[int(round(x)), int(round(y))]],
                          [[int(round(x + width_px)), int(round(y))]],
                          [[int(round(x + width_px)), int(round(y + height_px))]],
                          [[int(round(x)), int(round(y + height_px))]]], dtype=np.int32),
    }


def roi_from_slot(slot, pad_x=0.35, pad_y=0.12, frame_shape=None):
    """Grow the window's box into a region of interest that takes in the marks.

    The calibration bars sit just above and below the opening and reach past it
    to one side, hence the padding is wider sideways than vertically.
    """
    x, y, w, h = slot["bbox"]
    dx, dy = int(pad_x * w), int(pad_y * h)
    x0, y0 = x - dx, y - dy
    x1, y1 = x + w + dx, y + h + dy

    if frame_shape is not None:
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, frame_shape[1]), min(y1, frame_shape[0])
    return (x0, y0, x1 - x0, y1 - y0)


def refine_bar_centres(bgr, marks, strip_fraction=0.55, band_fraction=0.8,
                       max_spread=1.6, max_shift=0.5):
    """Find each bar's centre to sub-pixel precision from a brightness profile.

    find_marks gets a centre from the bounding box of a thresholded contour, so
    it depends on where the threshold happened to cut the bar's edges and it is
    quantised to whole pixels. A brightness-weighted centroid does not care
    where the threshold fell, and lands between pixels.

    Everything here is about not letting the bar's peak run into something else
    bright nearby, because the ring around the opening is only a few millimetres
    away and under front lighting it is brighter than the bar. Three things keep
    them apart, and all three were put in after a frame where the lower bar's
    centre moved 27 pixels toward the window:

    - the band searched is shorter than the gap to the ring, expressed as a
      fraction of the bar's own thickness, since the printed gap and the printed
      thickness are both known and their ratio does not change with zoom
    - the rows above half the peak height must not spread further than a small
      multiple of the bar's thickness, or the peak has merged with its neighbour
    - a centre that moves further than half a bar thickness is not believed, and
      the bounding-box centre is kept instead

    Each mark comes back with a sub-pixel "centre_y", and "refined" saying
    whether the centroid was accepted.
    """
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    refined = []

    for mark in marks:
        x, y, w, h = mark["bbox"]
        half = max(int(strip_fraction * w / 2), 1)
        centre_x = int(round(mark["centre"][0]))
        band = max(int(band_fraction * h), 2)

        y0, y1 = max(y - band, 0), min(y + h + band, grey.shape[0])
        x0, x1 = max(centre_x - half, 0), min(centre_x + half + 1, grey.shape[1])
        rows = grey[y0:y1, x0:x1].mean(axis=1)

        floor = rows.min()
        peak = int(np.argmax(rows))
        half_height = floor + (rows[peak] - floor) / 2.0

        lo = peak
        while lo > 0 and rows[lo - 1] >= half_height:
            lo -= 1
        hi = peak
        while hi < len(rows) - 1 and rows[hi + 1] >= half_height:
            hi += 1

        weights = rows[lo:hi + 1] - floor
        total = weights.sum()
        centre_y = mark["centre"][1]
        accepted = False

        if total > 0 and (hi - lo + 1) <= max_spread * h:
            candidate = float((np.arange(lo, hi + 1) * weights).sum() / total) + y0
            if abs(candidate - mark["centre"][1]) <= max_shift * h:
                centre_y, accepted = candidate, True

        refined.append({**mark, "centre_y": centre_y, "refined": accepted,
                        "centre": (mark["centre"][0], centre_y)})
    return refined
