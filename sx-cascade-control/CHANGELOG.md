# Changelog

## v1.3 (2026-10-25): two blind spots in the controller, and their fixes

New:
- The controller refits the model's three uncertain numbers (the two separation factors and the
  acid exponent) every 2 hours on the last 2 hours of readings and rebuilds itself on the new model
  (notebook 51). On the plant where a slightly wrong model ran the acid out, the first refit had the
  numbers right; on all ten test plants it removes what a wrong model adds (notebook 53).
- The saponification is capped by the plant's acid balance, from what the plant shows (notebook
  52). On all ten plants, with every model tried, the extraction acid doesn't go under its floor, at
  no cost in time below spec (notebooks 53, 56). It governs the acid at the outlet: aiming 3 mmol/L
  above the floor kept two of three plants clear of it, not the third (notebook 57).
- A rest-point check in the MPC (MPCSettings.rest_check_every; off by default, so earlier results
  are unchanged; ARCHITECTURE decision 42): every 12 hours the model is solved to rest on the current
  handles and the long-run floors are set there (notebook 55). On the two plants that drifted toward
  off spec while reading 100 % purity, it holds them at 99.97 to 99.99 %.
- The slow drift measured over 600 hours, with recovery counted by losses and the metal stock
  (notebook 54).
- All three together, from a wrong model, over 600 hours (notebook 56): no stall, no drift, no acid
  under the floor on any of the ten plants, and the same or fewer hours below spec; without them the
  same wrong model drifts toward off spec on 7 plants, runs the acid out on 5 and stalls one.
- Five more ideas, each tested on its own (FINDINGS phase 10): the rest-point check every step
  instead of every 12 h, no better on the drift at twice the run time (notebook 58); the circuit's
  NdPr stock as a drift alarm, a cross-check that also fires when the stock only shifts (notebook 59);
  the controller holding recovery counted by losses, no better (notebook 60); a penalty that starts
  3 mmol/L before the acid floor, which keeps the true model off the floor on all ten plants but not a
  wrong model (notebook 62); the purity output in log impurity, which halves the drift but doesn't stop
  it (notebook 61). Three new MPC settings for them, all off by default (ARCHITECTURE decision 43).
- Correcting the model at its inputs instead of its outputs, screened on this plant's gains before
  building it (notebook 63): about ten times better with an accurate model, unstable on most plants
  with the model's real errors. Not built.
- LITERATURE.md: what this project takes from the literature, and where.
- A readability pass over the code and the notebooks: every function documented, long functions
  split into named helpers, and the few functions that keep more than four parameters say why.
  Checked against the reference runs to the last bit and, for each reworked notebook, against its
  stored results.
- The README's numbers and FINDINGS phase 10 checked against the result tables.

No earlier result changed. Found while checking: the operating window table from notebook 19
can't be reproduced as committed; the conclusion holds (details in CHANGES_TO_EARLIER_RESULTS.md,
item 4).

## v1.2 (2026-10-25): choosing what to fit, and testing whether it matters

New:
- Which numbers to fit, chosen by checking every candidate in the model: how much the readings
  change when it moves, which ones the readings can't tell apart, and how much getting it wrong
  changes the control. The fit now adjusts the two separation factors and the acid exponent, from
  four starting points (notebook 43).
- The estimators tested on plants with physics the model leaves out: entrainment between stages,
  settlers the liquid flows through, chemistry drifting over days, and a plant built from the
  book's relations (notebook 44).
- The smallest network that does the job: 2,754 weights, half the size used before, within 10 % of
  the best; the closed loop is the same for every size tried (notebook 45).
- What the controller's correction should do when a refitted model is swapped in (notebook 46).
- Does the model's accuracy matter to the plant: the controller run on models wrong by known
  amounts, on ten plants (notebooks 47, 48).
- The feed switch looked at in metal lost and against an optimizer that knows the plant
  (notebooks 49, 50).
- Steady states are now solved all the way to rest for gains and fits (notebook 24 has a
  `--settled` switch); a plant simulation that stalls now stops with a clear error instead of
  running for hours.
- The numbers in FINDINGS phases 8 and 9 and the section 4 note are checked automatically against
  the result tables (tests/test_findings_numbers.py).
- The notebooks share their common pieces in one file (notebooks/study_tools.py).
- Versioned releases, with this file.

Results that changed (details in [CHANGES_TO_EARLIER_RESULTS.md](CHANGES_TO_EARLIER_RESULTS.md)):
- The network against the physics fit: v1.1 said the network was about 4 to 7 times closer. With
  the right numbers fitted, the physics fit is closer: 0.002 against 0.004 on a typical plant.
- How lopsided the plant is: condition number 43 became 204 once the steady states were solved to
  rest. The conclusion holds and is stronger.
- How far the model's gains are from each plant's at its own design point: above the limit an
  inverting controller can stand on all 19 plants (v1.1 said 17 of 19), by a factor of 3 to 28
  (v1.1 said up to 130).

## v1.1 (2026-10-19)

README: new title, the results shown as a recording of the workbench running a feed switch, the
controller on twenty plants seeing only the analysers, and where to measure.

## v1.0 (2026-10-19)

First release: the circuit simulated from its chemistry, the design optimizer, the MPC over level
loops, the estimator and sensors, the workbench and the studies behind every result.
