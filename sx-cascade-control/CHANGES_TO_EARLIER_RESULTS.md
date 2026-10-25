# Changes to earlier results

Results from earlier versions that a later check changed, with what was said, what's true now, and
why. The old results stay in the documents, marked with dated notes, next to the new ones; the old
versions stay tagged. Listed by the version that changed them.

## In v1.2

### 1. The physics fit against the neural network

**What was said (README, v1.0 and v1.1):** the neural network reads the separation factors "about 4
to 7 times closer than the physics fit on the main case", and "the physics fit comes out biased when
the plant differs from the model in more ways than it fits".

**Why it was looked at again:** the physics model should be able to do better than that. The test
plants differ from the model in a known list of ways, and a fit that adjusts the right numbers
should find them.

**What was wrong:** two things in how the fit was set up, not in the physics.

- It adjusted the wrong third number. Besides the two separation factors it fitted one shared mixer
  transfer rate. On the main case the stages run close to equilibrium, so the readings can hardly
  see that rate (its uncertainty from the readings is about seven times its plausible range), and
  it often ended on its limit. The acid exponent, which the test plants really do have different
  (3.3 against the model's 3.0), wasn't fitted at all.
- The search could stop in a wrong place. Started only from the model's own numbers, it stopped in
  a second minimum on some of the plants that separate best. A plant that differs from the model
  only in the fitted numbers showed it: 3 of 10 fits ended 0.29 to 0.44 off. Starting from four
  points instead of one fixes it on all ten.

**What's true now:** fitting the two separation factors and the acid exponent, from four starts, on
the same ten plants and the same noisy readings. Error in the Pr/(La+Ce) separation factor:

|                                         | typical (median) | worst |
| --------------------------------------- | ---------------- | ----- |
| physics fit as published                | 0.058            | 0.368 |
| physics fit, acid exponent, four starts | 0.002            | 0.013 |
| neural network (unchanged)              | 0.004            | 0.179 |

On these plants the physics fit is about as good as the network on a typical plant and much better
on the worst one, and it takes about half a minute instead of a millisecond. It's also the closest
estimator on plants with entrainment, settlers the liquid flows through, and drifting chemistry
(FINDINGS phase 8, Test 4). With either estimate the controller has every plant on spec at 120
hours, so on these plants the difference doesn't show in purity and recovery.

### 2. How lopsided the plant is (condition number)

**What was said (FINDINGS section 4, v1.0 and v1.1):** the long-run gain from the scrub and organic
flows to impurity and loss has a condition number of 43 and a relative gain of 9.6 (18 and 2.4 with
ratio handles).

**What was wrong:** the steady states behind those gains hadn't finished settling. The solver
stopped after 250 hours of simulated time while the circuit's slowest mode was still moving, and
the warning that said so wasn't shown. Settled fully, at the same design point:

|                       | published | now      |
| --------------------- | --------- | -------- |
| condition number      | 43        | 204      |
| relative gain         | 9.6       | 4.0      |
| with ratio handles    | 18 / 2.4  | 87 / 1.5 |

The conclusion holds and is stronger: the plant answers the ratio of the two flows far more than
the two together.

### 3. How far the model's gains are from each plant's

**What was said (FINDINGS section 4):** the model's gain error is above 1 on 15 of 19 plants at each
plant's own design point, up to 130. (The count was corrected to 17 of 19 on Oct 21: the table
said 17.)

**What was wrong:** the same unsettled steady states as in 2.

**What's true now (notebook 24 with `--settled`):** above 1 on 19 of 19 plants, by a factor of 3.1 to
28.0. At the design handles, the ten plants that were 0.58 to 0.77 are 0.91 to 1.01. The conclusion
holds and is stronger: at its own design point, every plant's gains differ from the model's by more
than a controller that inverts the model can stand. On 13 plants some of the solves still moved
slightly after 4,000 simulated hours, so those plants' exact values are less certain; the six
plants where every solve settled give 17 to 28.

## What didn't change

The stage-by-stage profile at the design point is the same to four decimals when fully settled.
