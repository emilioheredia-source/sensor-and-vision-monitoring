# How the software is put together

[PLAN.md](PLAN.md) covers the rig and the experiment. This covers the code.

The shape is the one a LabVIEW block diagram has: a source block, then
processing blocks, then a measurement block, then somewhere for the answer to
go. Each block does one thing and can be swapped without touching the others.

```
  SOURCE            PROCESS               MEASURE              SINK
  ------            -------               -------              ----
  live camera  -->  crop to ROI      -->  find the        -->  save frame + metadata
  saved folder      grey / colour         interface            write the number to CSV
  video file        normalise             convert px->mm       show it on screen
                                                               publish a PV (later)
```

## One decision carries the whole design

**A source is an iterator of frames, and nothing downstream knows which kind it
is.** The live camera and a folder of saved images present the same interface,
hence the detector is written once and runs on both.

That is worth stating plainly because it is what makes the project honest. The
detector cannot be quietly tuned against live conditions and then reported
against saved ones, since it is the same code path. It also means the whole
thing can be developed offline from recorded frames and switched to live by
changing one line, which is how it would go into a plant.

## The thing that flows through

One object, carrying the image and everything learned about it so far.

```python
@dataclass
class Frame:
    image: np.ndarray      # the pixels
    meta: dict             # timestamp, source, exposure, frame id
    results: dict          # what the stages worked out: interface_px, interface_mm
```

Every stage appends what it did and what settings it used to `meta["pipeline"]`.
That costs nothing and means a saved measurement can be traced back to the
settings that produced it, which is the difference between a demo and something
anyone would let near a plant.

## The pieces

| file | what is in it |
|---|---|
| `frames.py` | the `Frame` above, and nothing else |
| `sources.py` | `CameraSource`, `FolderSource`, `VideoSource` |
| `stages.py` | `Crop`, `ToGray`, `Normalise`, and whatever else earns its place |
| `detect.py` | the interface finder, working in pixels only |
| `calibrate.py` | pixels to millimetres, and the ruler fit behind it |
| `sinks.py` | `ImageSink`, `CsvSink`, `DisplaySink`, later `EpicsSink` |
| `pipeline.py` | wires source, stages and sinks together and runs them |

**Detection and calibration are separate on purpose.** The detector returns a
pixel row. Calibration turns it into millimetres. Recalibrating after the camera
is nudged then costs nothing, since no detection has to be redone.

## Storage: files on disk and a CSV, nothing cleverer

Captured frames go to `data/raw/` as **PNG**, one row per frame in
`data/manifest.csv`.

PNG rather than JPEG for a specific reason rather than taste: JPEG artifacts sit
exactly on sharp edges, and a sharp edge is the thing being measured. Lossy
compression would put noise precisely where the measurement is taken.

CSV rather than HDF5, Parquet or a database because the ground truth is typed in
by hand off a ruler, and a format that cannot be opened and corrected in a text
editor is the wrong format for hand-entered data. At a few thousand frames there
is nothing to gain by being cleverer. If it ever grows past that, the manifest
becomes Parquet and nothing else changes.

```
data/
  raw/                 frame_0001.png ...
  manifest.csv         filename, timestamp, source, exposure, lamp position, notes
  ground_truth.csv     filename, true_oil_water_mm, true_oil_air_mm, notes
```

Two files rather than one, because the manifest is written by the machine and
the ground truth is written by hand, and mixing those invites overwriting the
measurements on the next capture run.

**No annotation tool is needed.** The ground truth is a ruler reading in
millimetres, not a click on a screen. If the messy-band question gets picked up
later and labels get richer, that is when Label Studio or CVAT earns its place.

## Configuration in one file

Camera index, ROI, exposure, calibration constants and paths go in a
`config.toml`, read with `tomllib` from the standard library. Parameters
scattered through the code is how the calibration silently stops matching the
capture.

## The camera, and the trap in it

Capture is `cv2.VideoCapture`. Two things matter more than the library choice:

**Turn the automatics off.** Auto-exposure, auto-white-balance and auto-focus
will each change the image between calibration and measurement, and the
measurement will drift for reasons that have nothing to do with the liquid. They
get set once, explicitly, and the values get written into the frame metadata.

**Retry the first read.** Measured on this machine with OpenCV 5 headless, the
DSHOW backend reports itself as available and then cannot capture by index at
all, so the default backend is the one to use, and its first read commonly fails
while the device warms up. Four attempts a second apart clears it, which is what
the CLS camera widget already does.

## What each library is doing

Being precise about this, since "we used OpenCV" says very little:

- **OpenCV** — camera capture, colour conversion, and the primitives if they are
  needed: Canny, Hough, undistortion.
- **NumPy** — the actual detection. Averaging across columns to get one vertical
  profile, differentiating it, and picking the peak is array arithmetic.
- **pandas** — the manifest, the ground truth join, the error table.
- **matplotlib** — the error plot, which is the deliverable.
- **PyQt** — live display, if the display is worth having.

Nothing new to install. The workspace `pixi.toml` already carries all of it.

## Left for later, but the slot is there

- **`EpicsSink`** publishing the level as a process variable, over `pyepics` or
  `p4p`. It is one class implementing the same `write(frame)` as the CSV sink,
  which is the point of having a sink interface at all.
- **A threaded source.** Live capture and processing on one thread is fine at
  the few frames per second this needs. If it ever needs more, the source gets
  wrapped in a queue and nothing downstream changes.
- **A learned detector.** Same `detect.py` interface, so it drops in beside the
  classical one and gets measured against it on the same frames. Only worth
  reaching for on the emulsion band, never on the clean line.
