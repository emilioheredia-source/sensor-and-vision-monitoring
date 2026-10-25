# Changelog

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
