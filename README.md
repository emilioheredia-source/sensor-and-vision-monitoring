# Process monitoring, control and machine vision

Three projects: a rare earth separation plant simulated from its chemistry with
a controller on top, machine condition read from sensors, and a liquid level
read from a camera.

## sx-cascade-control

A rare earth solvent extraction circuit, built on a computer from the chemistry
and published data, with a model predictive controller on top. It splits
lanthanum and cerium from the NdPr magnet metals over 32 mixer-settler stages.

<img src="sx-cascade-control/figures/diag_flowsheet.png" width="600" alt="the circuit: extraction, scrub and strip sections and the streams between them">

The controller is tested on twenty plants that differ from its model inside
published ranges, eight upsets each, and on a harder plant built from a 2018
textbook. The work also looks at where instruments pay off, at learning the
plant's numbers from its readings with a person reviewing each update, at
putting a controller onto a plant in steps, and at an OPC UA link with a
historian. It's a simulation; none of it is plant data.

**[Write-up, method and results →](sx-cascade-control/README.md)**

## process-anomaly-detector

Detects turbofan engines going bad before they fail, on NASA C-MAPSS data: 100
engines, 21 sensors, every engine run until it failed.

<img src="process-anomaly-detector/figures/02_engine_traces.png" width="600" alt="health index climbing before failure on three held-out engines">

A health index built from each engine's own healthy baseline gives about 100
flights of warning on engines it has never seen, catches all of them, and
raises no false alarms. PCA monitoring and Isolation Forest are measured
against it over ten random splits, using the same code and the same threshold
rule, and land within noise of it.

**[Write-up, method and results →](process-anomaly-detector/README.md)**

## interface-level-vision

Reads a liquid-liquid interface level from a camera: a jar of water and oil on
a desk, read through a window shaped like the sight glass on a solvent
extraction settler.

<img src="interface-level-vision/figures/stage_reading.png" width="380" alt="both boundaries read off in millimetres against the printed scale">

A printed mask gives the scale and the calibration. Checked against a hand
reading, and across five different lightings and ten repeat shots.

**[Write-up, method and results →](interface-level-vision/README.md)**

## Running them

Each project has its own tools and its own `Running it` section in its README.
One pixi environment covers all three. The sx-cascade-control workbench starts
with `pixi run workbench`.

```powershell
pixi install
cd process-anomaly-detector\notebooks
pixi run python 01_explore.py
```

The scripts use `# %%` cell markers so they open as notebooks in VS Code, but
they are ordinary Python files and run from the command line. Plots open in
windows; set `HEADLESS=1` to write them to `figures/` instead.

The C-MAPSS data is not in this repo; `process-anomaly-detector/README.md` says
where to get it.

## Licence

MIT, see [LICENSE](LICENSE).
