"""Read the liquid levels inside a found window.

**Three regions, two interfaces.** Reading up the window there is a bottom, a
middle and a top, and what is wanted is where the bottom meets the middle and
where the middle meets the top. In a settler those are the aqueous solution, the
crud, and the organic solvent. On the bench rig they are water, oil and air.

When the middle layer is thin the two boundaries merge into one, and that is the
answer rather than a failure: a single interface with no measurable layer
between. The separation below which they count as one is a fraction of the
window height and is a setting, since how thin is too thin is a decision about
the process rather than about the image.

The window is known, so the measurement is one dimensional. Average across the
columns of the opening to get one value per row, and a boundary between two
liquids is a step in that profile. Differentiate, and a step becomes a peak.

**Which channel is profiled matters more than anything else here**, and the
choice is decided by the harder of the two boundaries rather than the easier.
Measured on the third bench frame, with each step expressed as a percentage of
its own profile's range so the comparison is scale free:

    channel          oil/air    oil/water
    grey              11.03%       2.77%
    saturation         9.54%       5.65%
    blue              10.00%       5.02%
    blue - red         7.48%       5.69%
    blue / red         9.55%       5.65%
    green / red       10.61%       5.85%
    blue / (R+G+B)     9.58%       6.03%

Grey wins on the oil and air boundary and is close to useless on the oil and
water one, which is the boundary that matters. The oil is yellow, so it is short
of blue, and the two liquids differ far more in colour than in brightness.

The default is blue as a fraction of the total, since it is the best of them on
the harder boundary. Being a ratio it also holds still when the lighting
changes, because scaling all three channels by the same factor leaves it alone,
which a difference such as blue minus red does not.

Every channel puts both boundaries in the same rows, so this is about how
confidently they are found rather than where.

Averaging across the columns is what beats the noise down: a single column is
noisy, and forty of them average to something a derivative can be taken of. The
scale comes from the window itself, which is printed at a known height, so a row
converts to millimetres without needing anything else in the frame.

Nothing here is trained. It is a column average, a smooth, a derivative, and a
peak pick.
"""

import cv2
import numpy as np

CHANNELS = {
    "saturation": lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1],
    "gray": lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
    "blue": lambda bgr: bgr[:, :, 0],
    "green": lambda bgr: bgr[:, :, 1],
    "red": lambda bgr: bgr[:, :, 2],
    "b_minus_r": lambda bgr: (bgr[:, :, 0].astype(np.float32)
                              - bgr[:, :, 2].astype(np.float32)),
    # Ratios rather than differences, so the value does not move when the lamp
    # does. Scaling every channel by the same factor leaves a ratio unchanged.
    "blue_fraction": lambda bgr: (bgr[:, :, 0].astype(np.float32)
                                  / (bgr.astype(np.float32).sum(axis=2) + 1.0)),
    "b_over_r": lambda bgr: (bgr[:, :, 0].astype(np.float32)
                             / (bgr[:, :, 2].astype(np.float32) + 1.0)),
    "b_over_g": lambda bgr: (bgr[:, :, 0].astype(np.float32)
                             / (bgr[:, :, 1].astype(np.float32) + 1.0)),
    "g_over_r": lambda bgr: (bgr[:, :, 1].astype(np.float32)
                             / (bgr[:, :, 2].astype(np.float32) + 1.0)),
}

DEFAULTS = {
    "channel": "blue_fraction",
    "inset_fraction": 0.06,   # ignore this much of the width at each edge
    "smooth_rows": 9,         # rolling mean along the profile, in rows
    "min_separation": 0.05,   # peaks closer than this, as a fraction of height, are one
    "edge_margin": 0.09,      # ignore peaks this close to the rim of the opening
}


def column_profile(bgr, slot, opts=None):
    """One value per row: the mean across a strip of the window at that height.

    The strip is as wide as the calibration bars and centred like them, which is
    why they are printed narrower than the opening. A strip that width runs down
    inside the window without ever reaching the rim, so nothing but liquid
    contributes and there is no edge of the opening to put a false step into the
    profile.

    Falling back to the window's own outline, eroded inward, only happens when
    the window was segmented rather than placed from the bars.
    """
    opts = {**DEFAULTS, **(opts or {})}
    if opts["channel"] not in CHANNELS:
        raise ValueError(f"unknown channel {opts['channel']}, "
                         f"expected one of {sorted(CHANNELS)}")
    values = CHANNELS[opts["channel"]](bgr).astype(np.float32)
    x, y, w, h = slot["bbox"]

    if "strip" in slot:
        x0, x1 = slot["strip"]
        # The opening is an obround, so it narrows to a rounded cap at each end
        # and a rectangular strip pokes past the curve there. Trim to where the
        # opening is at least as wide as the strip: for a cap of radius r and a
        # strip of half-width s that is r - sqrt(r*r - s*s) from either end.
        radius = w / 2.0
        half_strip = min((x1 - x0) / 2.0, radius)
        inset = int(np.ceil(radius - np.sqrt(max(radius ** 2 - half_strip ** 2, 0.0))))

        band = values[y:y + h, x0:x1]
        profile = np.full(h, np.nan)
        profile[inset:h - inset] = band[inset:h - inset].mean(axis=1)
        counts = np.zeros(h)
        counts[inset:h - inset] = x1 - x0
        return profile, counts

    inside = np.zeros(values.shape, np.uint8)
    cv2.drawContours(inside, [slot["hull"]], -1, 255, -1)
    erode_by = max(int(opts["inset_fraction"] * w), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_by + 1,) * 2)
    inside = cv2.erode(inside, kernel)

    counts = (inside[y:y + h, x:x + w] > 0).sum(axis=1)
    totals = np.where(inside[y:y + h, x:x + w] > 0,
                      values[y:y + h, x:x + w], 0).sum(axis=1)
    profile = np.divide(totals, counts, out=np.full(h, np.nan), where=counts > 0)
    return profile, counts


def _smooth(values, window):
    """Rolling mean that keeps the array length, ignoring gaps.

    Gaps are filled by carrying the nearest real value outward rather than by
    the profile's mean. The trimmed ends of the strip are gaps, and filling them
    with the mean puts a step where the real data starts, which the derivative
    then reports as a boundary that is not there.
    """
    filled = np.array(values, dtype=float)
    valid = np.flatnonzero(~np.isnan(filled))
    if len(valid) == 0:
        return filled
    filled[:valid[0]] = filled[valid[0]]
    filled[valid[-1] + 1:] = filled[valid[-1]]
    still_missing = np.isnan(filled)
    if still_missing.any():
        filled[still_missing] = np.interp(np.flatnonzero(still_missing),
                                          np.flatnonzero(~still_missing),
                                          filled[~still_missing])
    return np.convolve(filled, np.ones(window) / window, mode="same")


def on_printed_scale(slot, row_in_window):
    """Convert a row inside the window to the reading on the mask's own scale.

    Takes the row relative to the window, and gives back millimetres above the
    lower calibration bar, which is where the printed scale reads zero. Using
    the same datum as the operator means a reading by eye and a reading by
    machine can be compared without an offset between them to unpick.
    """
    if "scale_zero_y" not in slot:
        return None
    row_in_frame = slot["bbox"][1] + row_in_window
    return (slot["scale_zero_y"] - row_in_frame) * slot["mm_per_px"]


def find_interfaces(bgr, slot, window_height_mm=None, opts=None, max_interfaces=2):
    """Locate the boundaries inside the window.

    Returns the profile, its derivative, and the boundaries found, strongest
    first. Each carries its row within the window, how far down the window it
    sits as a fraction, the height of the step, and its position in millimetres
    when the window's real height is given.
    """
    opts = {**DEFAULTS, **(opts or {})}
    profile, counts = column_profile(bgr, slot, opts)
    height = len(profile)

    smoothed = _smooth(profile, opts["smooth_rows"])
    slope = np.gradient(smoothed)

    margin = int(opts["edge_margin"] * height)
    separation = max(int(opts["min_separation"] * height), 1)

    strength = np.abs(slope).copy()
    strength[:margin] = 0
    strength[height - margin:] = 0

    found = []
    for _ in range(max_interfaces):
        row = int(np.argmax(strength))
        if strength[row] <= 0:
            break
        found.append({
            "row": row,
            "fraction": row / float(height),
            "step": float(slope[row]),
            "strength": float(strength[row]),
            "mm_from_top": (None if window_height_mm is None
                            else window_height_mm * row / float(height)),
            "mm_from_bottom": (None if window_height_mm is None
                               else window_height_mm * (1.0 - row / float(height))),
            "mm_on_scale": on_printed_scale(slot, row),
        })
        strength[max(row - separation, 0):row + separation] = 0

    found.sort(key=lambda b: b["row"])
    for boundary, name in zip(found, ("middle to top", "bottom to middle")):
        boundary["name"] = name

    layer = None
    if len(found) >= 2:
        upper, lower = found[0], found[-1]
        layer = {
            "top_row": upper["row"],
            "bottom_row": lower["row"],
            "thickness_rows": lower["row"] - upper["row"],
            "centre_row": (upper["row"] + lower["row"]) / 2.0,
            "top_mm": upper["mm_on_scale"],
            "bottom_mm": lower["mm_on_scale"],
        }
        if layer["top_mm"] is not None and layer["bottom_mm"] is not None:
            layer["thickness_mm"] = layer["top_mm"] - layer["bottom_mm"]
            layer["centre_mm"] = (layer["top_mm"] + layer["bottom_mm"]) / 2.0
    elif len(found) == 1:
        found[0]["name"] = "single interface, middle layer too thin to resolve"

    return {"profile": profile, "smoothed": smoothed, "slope": slope,
            "counts": counts, "interfaces": found, "layer": layer}
