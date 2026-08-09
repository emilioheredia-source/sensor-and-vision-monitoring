"""Print a sight-glass mask with calibration fiducials, for the bench rig.

A settler is read through a small window in an opaque wall, so the detector only
ever sees a slice of the vessel. A clear jar shows everything, which is an
easier problem than the real one. Taping a printed mask over the jar puts the
difficulty back: a bounded aperture, a hard edge around it, and a surround that
reflects room light differently from the glass.

The window is an obround, straight sides with semicircular ends, which is what a
flat glass level gauge presents, since the glass sits in a frame whose gasket
cutout is rounded at both ends. A true ellipse is available too, for a round
sight port seen off-axis.

**The fiducial marks are the calibration.** A bar just above and just below the
opening, reaching past it either side, in the opposite colour to the surround.
Because they are printed on the mask they are fixed to the vessel, they sit in
the plane of the window, and they appear in every frame, so the
pixel-to-millimetre scale comes out of the image itself rather than out of a
ruler taped alongside, and it survives the camera being moved or re-zoomed.

**A white ring is left around the opening.** Finding the window by segmenting
its contents only works while the contents are brighter than the mask, and a
vessel holding something dark breaks that immediately. The ring is there
whatever is behind it, and traced from outside it is a filled obround, so the
same aspect and extent tests apply unchanged. When the contents happen to be
bright and merge with it, its outer edge is still in the same place.

Both bars sit close to the opening on purpose, so one region of interest covers the
window and both marks and the detector works from a single crop, and the
operator scale sits immediately beside the opening where it can be read without
parallax.

A bar carries the rotation on its own, since a line that is horizontal on the
mask is at the image's angle in the frame. Two bar centrelines and the two
overhanging ends give rotation, scale and shear, which is an affine map and is
what a squared-up camera needs. Recovering perspective as well would want four
well spread points, and the ends that sit at the edge of the opening are not
reliable enough to be those, since a pale mark against a bright window has
little contrast. Nothing is placed between the two bars, so the crop the
interface is measured in carries no mark that could be mistaken for one.

Measure the separation between the outer bars after printing and use the
measured value. That makes print scaling irrelevant, since the number that
matters is the one on the paper rather than the one that was asked for.

Three sheets are produced, since which is wanted depends on how the mask gets
made:

    aperture    black surround, window left white. Cut the white out and what
                remains is the mask. Marks are printed white on the black.
    silhouette  solid black window shape, for use as a template or an occluder.
    outline     thin lines only, for cutting by hand without printing a black
                field and emptying a toner cartridge.

Usage:
    pixi run python tools/make_sight_glass_mask.py
    pixi run python tools/make_sight_glass_mask.py --width-mm 60 --height-mm 150
    pixi run python tools/make_sight_glass_mask.py --shape ellipse --paper a4

Both PDF and PNG are written. Print the PDF, since it is vector and carries its
own physical size; the PNG is for looking at on screen.
"""

import argparse
from pathlib import Path as FilePath

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

MM_PER_INCH = 25.4

PAPER_SIZES_MM = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
}

DEFAULTS = {
    "shape": "slot",
    "window_width_mm": 40.0,
    "window_height_mm": 100.0,
    "surround_width_mm": 130.0,
    "surround_height_mm": 150.0,
    "fiducial_separation_mm": None,   # derived from the window unless given
    "fiducial_thickness_mm": 3.0,
    "outline_mm": 3.0,                # white ring left around the opening
    "scale_clearance_mm": 6.0,        # black gap between the ring and the scale
    "mark_width_fraction": 0.7,       # bar length as a fraction of the window width
    "mark_gap_mm": 4.0,               # clearance between a bar and the opening
    "scale_bar_mm": 100.0,
    "paper": "letter",
    "dpi": 300,
}


def make_page(paper):
    """Create a figure whose axes are addressed directly in millimetres.

    Returns the figure, an axes spanning the whole sheet with x and y in mm and
    equal aspect, and the page size, so a patch given a size in mm prints at
    that size.
    """
    page_w, page_h = PAPER_SIZES_MM[paper]
    fig = plt.figure(figsize=(page_w / MM_PER_INCH, page_h / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax, page_w, page_h


def stadium_path(centre, width, height):
    """Path for an obround: straight sides closed by a semicircle at each end.

    Built explicitly rather than from a rounded box style, because the box
    styles scale their corner radius by a mutation factor and will not reliably
    give ends that are true semicircles.

    Raises ValueError if the shape is not taller than it is wide, since there is
    then no straight section and the caller wants a circle instead.
    """
    if height < width:
        raise ValueError(f"a slot needs height >= width, got {height} x {width}")

    cx, cy = centre
    radius = width / 2
    straight = height - 2 * radius

    top = Path.arc(0, 180).transformed(
        Affine2D().scale(radius).translate(cx, cy + straight / 2))
    bottom = Path.arc(180, 360).transformed(
        Affine2D().scale(radius).translate(cx, cy - straight / 2))

    vertices = np.vstack([top.vertices, bottom.vertices, top.vertices[:1]])
    codes = np.concatenate([top.codes, bottom.codes, [Path.CLOSEPOLY]])
    codes[len(top.codes)] = Path.LINETO   # join the two arcs down the left side
    return Path(vertices, codes)


def make_window(shape, centre, width, height, **style):
    """Return the patch for the window opening itself."""
    if shape == "ellipse":
        return Ellipse(centre, width, height, **style)
    if shape == "slot":
        return PathPatch(stadium_path(centre, width, height), **style)
    raise ValueError(f"unknown window shape: {shape}")


def bar_extent(args):
    """Where a calibration bar starts and ends, relative to the window centre.

    Narrower than the opening and centred on it, so a region of interest the
    width of a bar runs straight down through the window. The column average
    inside that strip then sees only what is behind the glass, with no rim of
    the opening in it to put a false edge into the profile.

    The cost is a shorter lever arm for measuring rotation, since that improves
    with bar length. The white ring around the opening carries the rotation
    instead, being longer than any bar could be.
    """
    half = args.width_mm * args.mark_width_fraction / 2
    return (-half, half)


def fiducial_separation(args):
    """Vertical distance between the outer marks.

    Derived from the window rather than set independently, so the bars sit just
    clear of the opening and one region of interest covers the window and both
    of them. Overridable, since a taller calibration span is sometimes wanted.
    """
    if args.fiducial_separation_mm is not None:
        return args.fiducial_separation_mm
    return (args.height_mm + 2 * args.outline_mm
            + 2 * args.mark_gap_mm + args.fiducial_thickness_mm)


def draw_fiducials(ax, centre, args, colour):
    """Draw the calibration marks the machine measures against.

    One bar just above and one just below the opening, reaching past it either
    side, and nothing between them. They sit close to the window so that a
    single region of interest covers the opening and both marks, and the
    detector works from one crop.

    A bar carries the rotation on its own, since a line that is horizontal on
    the mask is at the image's angle in the frame. The two centrelines and the
    two overhanging ends give rotation, scale and shear, which is an affine map
    and enough to rectify the frame for a camera that is roughly square on. The
    ends at the window edge are not counted on, since a pale mark against a
    bright opening has little contrast to find a corner in.

    Nothing is placed at the middle height on purpose. A mark there would sit
    inside the same crop the interface is measured in, and column-averaging that
    crop would put a bright band into the profile at exactly the height an
    interface could occupy, which is a fake edge in the signal being measured. A
    third height would only test linearity along one line anyway, and could not
    separate lens distortion from tilt. That belongs in a printed grid
    photographed once as its own calibration step.

    Thickness barely affects how well the centre is located, since that comes
    from finding two edges and averaging them, and the uncertainty in that
    depends on edge sharpness rather than on the distance between the edges.
    Thickness only has to clear a floor: enough pixels across the bar that blur
    and compression do not swallow it. Length is what matters, because the
    rotation comes from the bar's angle and angular precision improves with the
    lever arm.

    Returns the horizontal extent of the marks, so a caller can size the region
    of interest to cover them.
    """
    cx, cy = centre
    thickness = args.fiducial_thickness_mm
    start, end = bar_extent(args)
    half_gap = fiducial_separation(args) / 2

    for offset in (+half_gap, -half_gap):
        ax.add_patch(Rectangle((cx + start, cy + offset - thickness / 2),
                               end - start, thickness, facecolor=colour, edgecolor="none"))
    return (cx + start, cx + end)


def draw_operator_scale(ax, centre, args, colour):
    """A numbered scale beside the window, for a person to read the level off.

    It is zeroed on the lowest fiducial and runs to the highest, so the operator
    and the detector read the same ruler. A reading taken by eye can then be
    compared with the measured one directly, with no offset between two
    coordinate systems to account for.

    It sits hard against the edge of the opening, with the fiducial columns
    pushed outboard of it. Anything read at an angle picks up a parallax error
    proportional to the sideways distance between the thing being read and the
    scale it is read against, so that distance is kept as small as the printing
    allows.
    """
    cx, cy = centre
    half_gap = fiducial_separation(args) / 2
    x_axis = cx + args.width_mm / 2 + args.outline_mm + args.scale_clearance_mm

    ax.plot([x_axis, x_axis], [cy - half_gap, cy + half_gap], color=colour, linewidth=1.2)
    for value_mm in range(0, int(fiducial_separation(args)) + 1):
        y = cy - half_gap + value_mm
        labelled = value_mm % 25 == 0
        if labelled:
            run, width = 4.5, 1.2
        elif value_mm % 10 == 0:
            run, width = 3.0, 1.2
        elif value_mm % 5 == 0:
            run, width = 2.0, 0.9
        else:
            run, width = 1.0, 0.6
        ax.plot([x_axis, x_axis + run], [y, y], color=colour, linewidth=width)
        if labelled:
            ax.text(x_axis + run + 1.0, y, f"{value_mm}", ha="left", va="center",
                    fontsize=7, color=colour)

    ax.text(x_axis, cy + half_gap + 4, "mm", ha="center", va="bottom",
            fontsize=7, color=colour)


def draw_scale_bar(ax, x, y_centre, length_mm):
    """A vertical ruled bar, for checking what the printer actually did."""
    y0 = y_centre - length_mm / 2
    ax.plot([x, x], [y0, y0 + length_mm], color="black", linewidth=1.2)
    for tick_mm in range(0, int(length_mm) + 1, 10):
        run = 4 if tick_mm % 50 == 0 else 2.5
        ax.plot([x, x + run], [y0 + tick_mm, y0 + tick_mm], color="black", linewidth=1.0)
    ax.text(x - 3, y_centre, f"{length_mm:.0f} mm", rotation=90,
            ha="right", va="center", fontsize=8)


def draw_aperture(ax, args, centre):
    """Black surround, white ring, and a dashed line showing where to cut.

    The white area is the opening plus the ring around it, and the dashed line
    inside it is the cut. Cutting there leaves a white ring of the requested
    width framing the hole, which is what makes the window findable when its
    contents have no contrast against the mask.
    """
    cx, cy = centre
    ax.add_patch(Rectangle(
        (cx - args.surround_width_mm / 2, cy - args.surround_height_mm / 2),
        args.surround_width_mm, args.surround_height_mm,
        facecolor="black", edgecolor="none"))

    ring = 2 * args.outline_mm
    ax.add_patch(make_window(args.shape, centre,
                             args.width_mm + ring, args.height_mm + ring,
                             facecolor="white", edgecolor="none"))
    if args.outline_mm > 0:
        ax.add_patch(make_window(args.shape, centre, args.width_mm, args.height_mm,
                                 facecolor="none", edgecolor="black",
                                 linewidth=0.6, linestyle=(0, (4, 3))))
    draw_fiducials(ax, centre, args, colour="white")
    draw_operator_scale(ax, centre, args, colour="white")


def draw_silhouette(ax, args, centre):
    """Solid black window shape."""
    ax.add_patch(make_window(args.shape, centre, args.width_mm, args.height_mm,
                             facecolor="black", edgecolor="none"))
    draw_fiducials(ax, centre, args, colour="black")
    draw_operator_scale(ax, centre, args, colour="black")


def draw_outline(ax, args, centre):
    """Thin lines only, to cut by hand without printing a black field."""
    cx, cy = centre
    ax.add_patch(make_window(args.shape, centre, args.width_mm, args.height_mm,
                             facecolor="none", edgecolor="black", linewidth=0.8))
    ax.add_patch(Rectangle(
        (cx - args.surround_width_mm / 2, cy - args.surround_height_mm / 2),
        args.surround_width_mm, args.surround_height_mm,
        facecolor="none", edgecolor="black", linewidth=0.5, linestyle=(0, (6, 4))))
    draw_fiducials(ax, centre, args, colour="black")
    draw_operator_scale(ax, centre, args, colour="black")


STYLES = {
    "aperture": draw_aperture,
    "silhouette": draw_silhouette,
    "outline": draw_outline,
}


def build_sheet(style, args, out_dir):
    """Draw one sheet in the given style and write it as PDF and PNG."""
    fig, ax, page_w, page_h = make_page(args.paper)
    centre = (page_w / 2, page_h / 2)

    STYLES[style](ax, args, centre)
    draw_scale_bar(ax, 22, page_h / 2, args.scale_bar_mm)
    ax.text(page_w / 2, page_h - 12,
            f"sight glass mask — {args.shape}, {style} — "
            f"window {args.width_mm:.0f} x {args.height_mm:.0f} mm",
            ha="center", va="top", fontsize=9)

    written = []
    for suffix, save_kwargs in ((".pdf", {}), (".png", {"dpi": args.dpi})):
        path = out_dir / f"sight_glass_{args.shape}_{style}{suffix}"
        fig.savefig(path, facecolor="white", **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", choices=["slot", "ellipse"], default=DEFAULTS["shape"],
                   help="slot is the obround a flat glass level gauge presents")
    p.add_argument("--width-mm", type=float, default=DEFAULTS["window_width_mm"],
                   help="window width across the vessel")
    p.add_argument("--height-mm", type=float, default=DEFAULTS["window_height_mm"],
                   help="window height, the direction the interface moves in")
    p.add_argument("--surround-width-mm", type=float, default=DEFAULTS["surround_width_mm"])
    p.add_argument("--surround-height-mm", type=float, default=DEFAULTS["surround_height_mm"])
    p.add_argument("--fiducial-separation-mm", type=float,
                   default=DEFAULTS["fiducial_separation_mm"],
                   help="distance between the outer marks. Derived from the window "
                        "height if not given, so the bars sit just clear of it")
    p.add_argument("--scale-clearance-mm", type=float,
                   default=DEFAULTS["scale_clearance_mm"],
                   help="black gap between the white ring and the scale, so the "
                        "two do not touch and merge into one shape")
    p.add_argument("--outline-mm", type=float, default=DEFAULTS["outline_mm"],
                   help="width of the white ring left around the opening. Zero for none")
    p.add_argument("--mark-width-fraction", type=float,
                   default=DEFAULTS["mark_width_fraction"],
                   help="bar length as a fraction of the window width. Below 1 so a "
                        "strip the width of a bar passes down through the opening")
    p.add_argument("--mark-gap-mm", type=float, default=DEFAULTS["mark_gap_mm"],
                   help="clearance between a bar and the edge of the opening")
    p.add_argument("--fiducial-thickness-mm", type=float,
                   default=DEFAULTS["fiducial_thickness_mm"])
    p.add_argument("--scale-bar-mm", type=float, default=DEFAULTS["scale_bar_mm"])
    p.add_argument("--paper", choices=sorted(PAPER_SIZES_MM), default=DEFAULTS["paper"])
    p.add_argument("--dpi", type=int, default=DEFAULTS["dpi"])
    p.add_argument("--out-dir", type=FilePath,
                   default=FilePath(__file__).resolve().parent.parent / "rig")
    p.add_argument("--style", choices=sorted(STYLES) + ["all"], default="all")
    return p.parse_args()


def check_geometry(args):
    """Fail early and clearly, rather than producing a quietly wrong sheet."""
    reach_x = args.width_mm / 2 + args.outline_mm + args.scale_clearance_mm + 22
    if reach_x > args.surround_width_mm / 2:
        raise SystemExit(
            f"the marks and scale reach {reach_x:.0f} mm from centre but the surround only "
            f"reaches {args.surround_width_mm / 2:.0f} mm. Raise --surround-width-mm.")

    reach_y = fiducial_separation(args) / 2 + args.fiducial_thickness_mm
    if reach_y > args.surround_height_mm / 2:
        raise SystemExit(
            f"the marks reach {reach_y:.0f} mm from centre but the surround only reaches "
            f"{args.surround_height_mm / 2:.0f} mm. Raise --surround-height-mm.")


def main():
    args = parse_args()
    check_geometry(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    styles = sorted(STYLES) if args.style == "all" else [args.style]

    for style in styles:
        for path in build_sheet(style, args, args.out_dir):
            print(f"wrote {path}")

    print(f"\n{args.shape} window {args.width_mm:.0f} mm wide by {args.height_mm:.0f} mm tall, "
          f"on {args.paper}, with calibration marks "
          f"{fiducial_separation(args):.0f} mm apart.")
    print("Print the PDF at 100%, with 'fit to page' off. Then measure the distance "
          "between the two marks and use the measured value, not the nominal one.")


if __name__ == "__main__":
    main()
