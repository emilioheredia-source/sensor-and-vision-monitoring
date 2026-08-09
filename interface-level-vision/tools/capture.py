"""Grab a frame from the camera and save it.

    pixi run python tools/capture.py --out tests/latest.png

Two things this does that a bare VideoCapture call does not, both learned from
the rig rather than guessed. The first read commonly fails while the device
wakes up, so it retries a few times a second apart, which is what the CLS camera
widget already does. And the first frames come out with the exposure and white
balance still settling, so a number of them are thrown away before one is kept.
"""

import argparse
import time
from pathlib import Path

import cv2


def open_camera(index, attempts=4, wait=1.0):
    """Open a camera, retrying while the device wakes up."""
    capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for attempt in range(1, attempts + 1):
        ok, _ = capture.read()
        if ok:
            return capture
        print(f"   read {attempt} of {attempts} failed, waiting")
        time.sleep(wait)
    capture.release()
    return None


def looks_like_no_signal(frame, max_brightness=60, max_saturation=12):
    """True when the frame looks like a placeholder card rather than a scene.

    A camera utility with nothing plugged into it still hands over a picture,
    and it is a dark grey card with a message on it. Measuring a liquid level
    from that would give a number with nothing behind it, so it is worth
    refusing. Real frames of the rig are neither this dark nor this colourless.

    Returns the reason as a string, or None when the frame looks like a scene.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = float(hsv[:, :, 2].mean())
    saturation = float(hsv[:, :, 1].mean())
    if brightness < max_brightness and saturation < max_saturation:
        return (f"mean brightness {brightness:.0f} and saturation {saturation:.0f}, "
                f"which is a dark colourless card rather than a scene")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--settle", type=float, default=6.0,
                        help="seconds to keep reading before giving up on a real scene")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="save even if the frame looks like a placeholder")
    args = parser.parse_args()

    print(f"opening camera {args.camera}")
    capture = open_camera(args.camera)
    if capture is None:
        raise SystemExit(f"could not read from camera {args.camera}")

    # A camera utility takes a few seconds to hand over real video, showing its
    # own placeholder until then, and the exposure is still moving after that.
    # So keep reading and stop at the first frame that looks like a scene,
    # rather than counting frames and hoping.
    print(f"reading for up to {args.settle:.0f} s, waiting for a real scene")
    deadline = time.time() + args.settle
    frame, complaint = None, "no frame was read at all"
    while time.time() < deadline:
        ok, candidate = capture.read()
        if not ok or candidate is None:
            continue
        frame = candidate
        complaint = looks_like_no_signal(frame)
        if complaint is None:
            print(f"   scene at {args.settle - (deadline - time.time()):.1f} s")
            break

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    capture.release()

    if frame is None:
        raise SystemExit("camera opened but no frame was read")
    if complaint is not None and not args.force:
        raise SystemExit(
            f"the camera returned {complaint}.\n"
            f"Nothing was saved. Check the camera is connected, awake, and in a "
            f"stills mode, or pass --force to keep the frame anyway.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)
    print(f"{int(width)}x{int(height)}, wrote {args.out}")


if __name__ == "__main__":
    main()
