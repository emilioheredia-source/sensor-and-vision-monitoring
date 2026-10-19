# Progress — sx-cascade-control

Newest first. **Status** says where things stand; **Log** says what happened
and why. Details of every decision: ARCHITECTURE.md.

## Status

| Phase | State |
| --- | --- |
| 0. Setup (plan, folders, env) | done — commit 0cbca8a |
| 1. Simulator | **done** — Kremser, balances, interface law; 48 tests green; operating points by optimization (design.py) with an 85 % loading cap; a perturbed plant the controller never sees (mismatch.py) |
| 2. Disturbance catalogue | **done** on the capped design, from the commissioned (trimmed) perturbed plant — figures/03_*.png, runs/disturbances_ndpr_plant.csv |
| 3. Stage-count study | **done** — NdPr: 95 % needs ~24 stages (~18 with ~99 % stages, ~19 with mixers sized to the flows), 32 give 99 %; Pr \| Nd: with the organic loaded freely both published circuits sit on the curve (60 stages → 98 %), with the 85 % cap the same plant sits at 92 % — `figures/04_stage_count_*.png`; exact-vs-fast and acid-floor checks done |
| 4. MPC | **done** — model-state linear MPC with output bias (integral action), steady-state floors, scrub-acid read-back, re-linearized each step, interface centering (fixed Oct 13, decision 26), OSQP with qpOASES as the backup (decision 30); on the perturbed plant: feed switch 931 → 8 mol lost, strong scrub acid 600 → 0, flowmeter bias 800 mol off spec → 0, drift 11 → 0.00 m·h; pump stop about even with open loop on this plant (0 mol off spec against none, 27 mol lost) but fragile: small differences decide whether it rides through or runs away (centering-weight sweep, `notebooks/15_*`; twenty plants); the fix is an interlock, not built — `runs/mpc_catalogue_ndpr_plant.csv`, `_matched.csv`, `runs/center_weight_sweep_ndpr_plant.csv` |
| 5. Sensors and actuators study | **done** on the perturbed plant — a feed switch needs a feed analyser (or the mid-cascade pair, which sees it arriving; the pair plus a feed analyser overreacts badly); a flowmeter bias needs an instrument at the product end (the pair alone isn't enough); product, raffinate and feed analysers do well on all three upsets; product and raffinate analysers without a feed analyser run away on the pump stop; past the patent's two flows each handle added makes a feed switch cheaper — `runs/sensor_actuator_ndpr_plant.csv`, `figures/07_sensor_placement_ndpr_plant.png`, `figures/08_sensors_ndpr_plant.png`, `figures/08_actuators_ndpr_plant.png` |
| 6. RL | **done** — PPO, 60 min, 16-stage perturbed circuit, five identical episodes against the MPC with the same sensors: holds purity on all five, a fifth less reagent, ~4× more NdPr lost on the feed-switch episodes (`runs/rl_eval_ndpr_small.csv`) |
| 7. Workbench | **first version done** — `pixi run workbench` (or `sx_workbench.bat`): choose run file, plant, controller, upset and sensor set (each explained), edit any setting, save the settings as a new run file, edit the run file or parameters.toml as text, jump from a setting to its line in the file or the code (decision 36), run, watch it live, compare saved runs; every panel a floating dock. Core restructured underneath (registry, runs, command line), bit-identical to the reference runs |
| 8. Structural mismatch | **first pass done** — settler and entrainment plants (`plant_models.py`); the full MPC holds all four upsets on both, the pump stop included at this weight; the pump stop is still fragile across weights and plants |
| 9. Twenty perturbed plants | **first version, superseded by 11-14** (its five hard plants were drawn outside the published ranges, decision 31) — 5 of 20 can't make both targets; there the controller cuts the loss by a quarter to a half on every upset but doesn't reach spec, because the model's gains are far too small there (adaptation, not ranking: ranked constraints tried and left off, `runs/ranked_constraints_ndpr_plant.csv`); on the other 15 the controller beats open loop on the feed switch 15/15, feed acid 15, stronger feed 14, strong scrub acid 14, flowmeter bias 8 (open loop clean on 7), pump stop better on 2 and worse on 5 — `runs/seed_spread_ndpr_plant.csv`, `figures/14_seed_spread_ndpr_plant.png` |
| 10. Stages sized to the flows | **done** (decision 28) — 130 L mixers, 520 L settlers: 95–96 % stages, ~19 stages for 95 %, less organic and reagent, a more linear plant 3.5× slower; now the main case (decision 34) — `runs/stage_sizing_ndpr.csv`, `figures/13_stage_sizing.png` |
| 11. Why the split is hard | **done** (FINDINGS_why_hard.md, notebooks 17, 19, 21, 22, 24) — purity goes as the flow ratio to the power of the stage count; the design sat on the edge; the plant is ill-conditioned like high-purity distillation; the literature (Xu's countercurrent theory, Cheng, Yang) agrees and runs the scrub ~60 % above its minimum |
| 12. Design and margin | **done** (notebooks 25, 26) — margin is where the plant is run: 99.8 / 97 widens the band about three times for ~15 % reagent; more stages make it cheaper |
| 13. Level loops, acidity floor | **done** (decisions 32, 33, notebook 20, 21) — level loops under the MPC beat jacklegs in it; the acidity floor stops the MPC over-saponifying into invalid chemistry |
| 14. Main case | **done** (decision 34, `configs/ndpr_main.toml`, notebook 14) — 20 realistic plants all on spec at the start; controller better on the feed switch 17/20, stronger feed 18, feed acid and strong scrub acid 20/20; pump stop still the weak upset |
| 15. Drifting plant | **first pass done** (decision 35, notebook 27) — up to 10 % drift in the separation factor over a week puts no plant off spec, even uncontrolled |
| 16. Book plant | **first pass done** (notebooks 29, 30, 14 --plant book) — plant_models.BookCircuit from Qi 2018: needs ~25 % more organic and reagent than the model; the controller on its ordinary model holds the feed and acid upsets on twenty book plants |
| 17. Identify, then control | **done** (notebook 31) — six noisy readings fitted to the physics model recover Pr/(La+Ce) within ~0.05; the MPC on the fitted model brings all seven hard plants to spec where the nominal model failed on all seven |
| 18. Instruments for identification | **done** (notebook 31 --sensors) — relative UV-Vis ratios (with a 15 % gain error) identify the separation factor best; product/raffinate XRF alone is enough for control |
| 19. Tracking drift | **done** (notebook 32) — refitting once a day keeps a plant whose separation factor slides for two weeks on spec; the fixed model loses it |
| 20. Drift or fault | **first pass** (notebook 33, one plant) — the daily fit's stage-rate factor jumps when eight scrub mixers foul, and a simple rule flags it while the separation factor keeps tracking the drift; the factor sits on its bound and moves the wrong way, so a per-section fit is the next step |
| 21. Feed-loss interlock | **done, rerun on the fixed code** (notebook 34, src/interlock.py) — freeze, re-sync, hand back: the pump stop goes from the controller's worst upset to less loss than the start on most of 20 plants, none off spec, no failed solves; also stopping saponification is worse. Rerun after the re-sync fix (decision 38), with open loop over the same 80 h: the interlock brings the MPC level with doing nothing (better on 9 of 20, worse on 11, +2.6 mol on average); without it the MPC is worse on 19 |
| 22. Learned estimator | **done on both plants** (notebook 35; on the main case about 4x / 7x more accurate than the physics fit, the linear baseline nearly as good, of three unconfirmed closed-loop plants, one confirmed at 400 h; plants 7 and 8 still settling, possibly toward off spec: open) — a PyTorch net trained on 1,200 simulated plants reads the Pr/(La+Ce) separation factor about 3.5 times more accurately than the physics fit on the hard plants and about 17 times on the easy ones, in half a millisecond; all seven hard plants to spec; only as good as the simulator's spread |
| 23. Deployment in steps | **done** (notebook 36, src/deployment.py, decision 37) — shadow, advisory, closed loop, move limits, watchdog, journal; on 20 plants the 20 % ceiling is too tight for the feed switch (7 plants off spec, 16,628 mol); sized from the upset (50 %), 6 plants and 5,104 mol, close to no guard; advisory did best |
| 24. Supervised model updates | **done** (notebook 37, src/model_registry.py, decision 39) — versions, checks, review, rollback; the review rejects the fault's fits a day before the trend flag |
| 25. Plant over OPC UA | **done** (notebook 38, src/plant_server.py, src/historian.py, decision 40) — controller as a client, historian; identical to the in-process run |
| 27. Identification on the main case | **done** (notebooks 31-33, 37, 39 with --config ndpr_main) — all ten plants on spec with the fitted model; drift held even by the fixed model, refitting saves 6 % organic; the scrub fault barely shows near equilibrium |
| 28. The rich feed as the base | **done** (configs/ndpr_main_yangibana.toml, notebook 14) — switching to the lean feed puts 14 of 20 uncontrolled plants off spec; the controller keeps all 20 on spec |
| 26. Where the instruments go | **done** (notebook 39) — the leverage is at the front in the mid-scrub; scrub points make a scrub rate factor identifiable |

**Since Oct 15** the main case is `configs/ndpr_main.toml` (decision 34); README and
FINDINGS_why_hard.md describe it. REPORT sections 3 to 8 and the rows above it are the first
version, on the 30 L plant, kept as the record. Next: estimating the separation factors and stage
efficiency online on the drifting plant (with and without an optical profile), fault flags from the
estimates' trend, operator clamping of handles; then the pump-stop interlock. The paragraph
below is from Oct 14 and kept as it was.

**Open decisions (Oct 14):** none. Every result in README,
REPORT and the appendices is from one code version (Oct 14 evening:
centering fixed, estimator budget in, backup solver in, every study rerun).
Next: model adaptation, correcting the controller's gains from plant data,
for plants far from the model (the five seeded plants that can't make both
targets); then the pump-stop interlock (freeze the handles below a
feed-flow threshold, reset from the plant when feed is back); each run on
the catalogue and the twenty plants. Also: why both solvers still fail on
seed 17. From the robustness plan: the sensor study and the structural matrix on
several seeds; entrainment at upset levels, combined with the perturbed
numbers and real sensor sets. Still queued: moving the notebooks onto
runs.py; the data-driven soft sensor. Four chosen values worth a glance when
you have a minute (all labelled in `params/parameters.toml`): Sm/Nd
separation factor = 10, organic density 0.85, feed = Eneabba with Sm+
removed upstream, the 85 % organic loading cap.

## Log

### Mon Oct 19

**README results redone, twice.** First as comparisons against a plant nobody touches; dropped, since no
plant runs untouched and the point is the work, not the score. Now: a recording of the workbench running the feed
switch (figures/workbench_feed_switch.gif, recorded by hand, cut to one frame in five at full resolution), the same
controller's purity and recovery on the twenty plants (new notebook 40, traces, with the workbench's product,
raffinate and feed analysers: 7 of 20 dip below spec, recovery back above 95 % within 8 h on all but it keeps moving
a few percent around the target; with the full state, as a check, 5 dip and recovery settles on the target), and where to measure (notebook 39). The workbench screenshot left the code
section. Number checks for the new sentences in tests/test_doc_numbers.py.

### Thu Oct 15 to Sat Oct 17

**Why the split is hard, and a new main case.** The five plants that couldn't make spec came
from a generator that drew separation factors outside the published ranges; the plants are now
drawn inside them (decision 31). On those, the hard plants are the ones that separate a little
worse than the model. Two things made them hard on the first plant: mixers too small for the flow
(stages at 81-85 %, so the plant's unknown efficiency mattered a lot) and no acidity floor in the
controller, which over-saponified into a region the chemistry model can't represent (decision 33;
also the real cause of the pump-stop weakness). The literature (D. Qi's book, chapters 3 and 4;
Cheng 2013; Yang 2015) explains the rest: purity goes as the flow ratio to the power of the stage
count, the cheapest design sits on the edge, and plants run with margin and hold the profile at
monitoring stages (FINDINGS_why_hard.md). Design study (notebooks 25, 26): margin is where the
plant is run, not only how it's built. Level loops under the MPC replace the jacklegs as handles
(decision 32). The main case (decision 34): sized stages, margin, acidity floor, level loops,
120 h trim; on twenty realistic plants every one starts on spec and the controller beats open
loop on the feed and acid upsets. A drifting plant (decision 35) shows that up to 10 % drift in
the separation factor over a week is absorbed by the margin. The first seed study's table is
kept as runs/history/.

### Wed Oct 14

**Evening: a backup solver, and ranked constraints tried** (decision 30). Looking at the five
plants that can't make both targets: the controller wasn't giving up purity by choice, it was
mostly frozen. On plants that far off spec the slack costs run into the millions and OSQP often
reported "primal infeasible inaccurate", a false alarm, since every output constraint has a
slack; on a failed solve the controller holds its handles (seed 3: every step of every run
failed). OSQP also wasn't repeatable there, because its step-size retuning follows its own setup
time by default. Fixed in `mpc.py`: `adaptive_rho_interval` = 25, and when OSQP fails the same QP
goes to qpOASES before the handles are held; each step records which solver answered. Every
closed-loop result rerun. On the five plants 401 of 3,200 steps still fail, 298 on seed 17.

Ranked constraints were built and tried both ways (`notebooks/16_ranked_constraints.py`): a
two-stage QP that finds the smallest spec shortfall first and caps it (`MPCSettings.ranked`),
and an exact penalty, a straight-line cost below spec on top of the squared one
(`w_purity_floor_linear`, 1000 and 100), each also used for the trim. They don't fix it. The
two-stage form hurt plants that can make spec (seed 2 down to 62 % purity on the feed switch,
4,592 mol off spec against 536; seed 1's trim ended at 99.48 %, short of target), helped seed 13
and hurt seed 3; the exact penalty changed little on seeds 1 and 2 and gave mixed results on 3
and 13. Both kept as settings, off. So this afternoon's "the controller gives up purity because
nothing ranks the spec first" was wrong. What's left on the hard plants is the model: on seed 3,
+10 L/h scrub moves purity +4.6 thousandths in the model and +64 on the plant (recovery −9.7
points against −1.5). Right sign, size off by about fifteen on purity, so the controller moves too
timidly. The next step is adapting the gains from plant data.

What moved with the rerun: on the five hard plants the controller now cuts the loss by a
quarter to a half on every upset (medians 2,490–3,570 mol open loop, 1,560–2,700 controlled) but
doesn't reach spec. On the fifteen others the results barely moved (strong scrub acid better on
14, was 13; pump stop better on 2 and worse on 5, worst 1,703 mol, was 1, 7 and 1,680). On seed 1
the pump stop now puts nothing off spec (was 190 mol), and so does every plant in the matrix. The
sensor study changed in places: the ECJTU pair with a feed analyser now puts 2,300 mol off spec on
the feed switch, and product and raffinate analysers without a feed analyser run away on the
pump stop (purity 66 %, 1,720 mol off spec). On the handles, saponification no longer makes the
feed switch cheap; past the patent's two flows each handle added does. The learned policy,
retrained, holds purity on all five episodes. Reagent on the pump stop is 6 % below open loop,
so "reagent within 5 %" became "at most 6 %". Seed 3's trim now ends at 49 %, not 47 %.

**Interface centering fixed, and everything rerun** (decision 26). The units slip found
yesterday is fixed in `mpc.py`: the pull of each interface toward its setpoint is now per cm²,
as meant. The commissioning trim uses the controller, so the trimmed point moved a little
(2,712 → 2,717 mol/h of reagent) and every result that starts from it moved a little: the 60 h
open-loop catalogue, both 40 h catalogues, the sensor and handle study, the structural matrix
(now one file from one run) and the learned policy (retrained). The stage sweeps, the design
point and the sensor placement didn't move. Some statements changed with the numbers: the
controller now uses 3.5 % less reagent on the feed switch; the mid-cascade pair alone no longer
holds a flowmeter bias; saponification, not scrub acidity, is the handle that makes a feed
switch cheap; the fixed linear MPC is as bad on the matched plant as on the new ones; the policy
dips to 96.9 % on one episode and loses about four times more NdPr, not three to fifteen.

**The pump stop is fragile.** A sweep of the centering weight from 0 to 0.02 on six cases
(`notebooks/15_*`): the feed switch, drift and flowmeter bias come out about the same at every
weight, the drift slightly better with the pull. The pump stop jumps around with no pattern. The
matched plant loses ~1,400 mol at 0 and ~1,300 at 0.02 but nothing at 0.0002 and 0.001; the
entrainment plant loses ~1,435 at 0.001. While the feed is off and refilling, recovery reads
nonsense and the controller acts on it; tiny differences decide whether it rides through or runs
away (cutting saponification and raising the organic while recovery collapses). The earlier
"nothing lost on the matched plant" was luck. Kept the weight at 0.02 and wrote the pump stop up
as a limit. The plant answer is an interlock: freeze the handles below a feed-flow threshold,
and when feed is back and settled, reset the bias and model state from the plant. Not built.

**Estimator prediction budget** (decision 27). With the ECJTU pair after the fix, the
estimator's adaptive integrator crawled (millions of steps for one half hour) and runs hung. The
prediction now gets 20,000 model evaluations, then falls back to implicit Euler for that step,
counted as `slow_predictions`. In the stalled case 4 of 80 steps fell back. Plant simulations
have no budget.

**Stages sized to the flows** (decision 28, side study). 130 L mixers and 520 L settlers instead
of the lab rig ×250: stages at 95–96 % instead of 81–85 %, ~19 stages for 95 % instead of 24, the
32-stage point on less organic (1,634 against 1,755 L/h) and reagent (2,172 against 2,308 mol/h),
still on the loading cap. The plant is more linear and about 3.5× slower (1,381 h against 393 h),
so 40 h runs see less of each upset. On it the pump stop isn't a problem and the flowmeter bias
puts nothing off spec even open loop. The main case stays the 30 L plant. Run files can now carry
`[parameter_edits]`.

**Twenty perturbed plants** (decision 29). Seeds 1–20, eight upsets, open loop and controller,
40 h, full state. Five can't reach 95 % recovery at spec with any handles (best 72–92 %), and on
those the trim gives up purity, which is the spec (seed 3: 47 %), because nothing ranks the
spec above recovery. Industrial MPC uses prioritized constraints for that; written up as a limit.
(That reading was wrong; see the evening entry.) On the other 15 the controller beats open loop on the feed switch 15/15 (median 931 → 0 mol),
feed acid 15, stronger feed 14, strong scrub acid 13, flowmeter bias 8 (open loop clean on 7);
pump stop better on 1 and worse on 7; drift and slow mixers no change in losses, with the drift's
interfaces kept in on all 15.

**Also:** the open-loop ramps fixed yesterday stay as quarter-hour steps. The workbench starts with
`pixi run workbench` or `sx_workbench.bat`, in a window titled "SX cascade workbench". All the
documents are updated to these results; `tests/test_doc_numbers.py` checks every quoted number
against its table.

### Tue Oct 13

**Cross-checked, reviewed, and one bug fixed.** Every decision and appendix pointer in the code and
the documents was checked against its target, every quoted parameter against the code, and about
forty quoted results against their tables; the few wrong ones are fixed. A chemical-engineering and
a control review added "Why:" and "Judgement call:" notes where choices are made (Appendix A.13 and
B.10 rank the open questions). The bug: an uncontrolled run held a ramped upset at zero for the
whole ramp and then applied it in one step, because the simulator only changed inputs at the
start and end of a ramp. Ramps are now applied in quarter-hour steps. Only the uncontrolled feed
switch and drift moved (40 h feed switch 930 -> 945 mol, drift 9.5 -> 10 m·h; 60 h feed switch
1,800 -> 1,810 mol, drift 16 -> 16.4 m·h); every controlled result is unchanged. The reference run
was re-recorded for that one array. The report's pH claim was also wrong: the design sits on the
85 % loading cap at about pH 1.6, not on a pH 3 bound.

**The code reads like a notebook.** Every file in src/, workbench/, notebooks/ and tests/ now
opens with a short essay on what it does, and walks through the code in a few lines of
explanation, then a few lines of code: what each variable is and its units, the equations in
plain text next to the code, a docstring on every function. Long functions were split to the
readability guidelines: the design optimizer from 97 lines to 35, the estimator's step from 63
to 7, the controller's set-up, linearization and QP into named steps. Checked: all 48 tests
pass, every reference array matches with a difference of exactly zero, and the estimator's
fixed-step path and the RL environment give identical results old against new.

Found on the way and left as it is (it would change every result): the controller's term that
pulls interfaces back to their setpoint is 10^8 times weaker than intended, because an offset in
metres is divided by 100 where it should be multiplied. So the controller keeps interfaces in
their window but doesn't recentre them. ARCHITECTURE.md, Open, says so.

### Fri Oct 9

**Afternoon: plants with physics the model doesn't have.** Two new plants in
`src/plant_models.py`, both in the workbench and the command line:
`settler` (transfer in the mixer only, the settler as three tanks in a row per
phase) and `entrainment` (droplets of each phase carried into the next stage,
more when an interface strays toward a weir). Matrix on the plant circuit, 40 h,
full state, three controllers by four plants by four upsets (48 runs, merged from two
`python -m src.cli matrix` runs into `runs/structural_matrix_ndpr_plant.csv`). Off-spec product, mol:

| Plant | Controller | Feed switch | Pump stop | Flowmeter bias | Interface drift |
| --- | --- | --- | --- | --- | --- |
| matched | open loop | 0 | 1357 | 1360 | 0 |
| matched | MPC linearized once | 1351 | 2209 | 1332 | 0 |
| matched | MPC | 215 | 0 | 0 | 0 |
| perturbed | MPC | 0 | 222 | 0 | 0 |
| settler | MPC linearized once | 4782 | 1642 | 1332 | 0 |
| settler | MPC | 0 | 0 | 0 | 0 |
| entrainment | MPC linearized once | 4487 | 1985 | 1360 | 0 |
| entrainment | MPC | 215 | 272 | 0 | 0 |

The full MPC copes with both. The one cost: on the pump stop with entrainment
the interfaces left the window and excess NdPr lost went from about zero to 84 mol. The
MPC linearized once does more than three times worse on a feed switch on both new plants.
Off-spec counts against a threshold, so 0 against 215 between similar runs is
the threshold, not a real difference. Open loop barely notices entrainment at
500 ppm. Next on this line: entrainment at upset levels, the two combined with
the perturbed numbers, and the same matrix with real sensor sets.

**08:26: the workbench.** `python -m workbench` from this folder. One window
of floating docks (Run, Settings, Saved runs, Scorecard, Circuit, Stages now,
Product, Interfaces, Handles, Losses and reagent, Controller's view); drag any
panel out onto another monitor, View > Reset layout to start over, the
arrangement is remembered. Choose and run on the left; every setting of the
run file, the parameter file and the controller is in the Settings tree, and an
edited value is marked with a dot. Finished runs are saved and can be overlaid
on the live plots. Built per the two guidelines: the presenter has no Qt in it
and talks to the view through an interface (tested with a fake view); the view
only draws. The first operating point for a given plant and settings takes
about 20 s (commission and trim), then it is cached.

**08:15: runs from a description.** `registry.py` (named plants, controllers,
sensor sets, each explained), `operating_point.py` (commission + trim, was
copied in five notebooks), `runs.py` (spec -> session -> result -> saved
folder), `cli.py` (list / run / matrix). The design rules that were hard-coded
in six notebooks (design purity, net product flow, pump ceiling and headroom)
are now `[design]` in the parameter file.

**08:04: the core restructured, numbers unchanged.** Reference runs recorded
first (`tests/baseline/`: 14 arrays, from trim to sensor sets to the design
optimizer). The 76-line closed loop became `loop.py`, a class that steps one
interval at a time; the controller's 139-line `step` and 112-line constructor
became named stages (measure, bias, linearize, predict, solve, advance). The
new code reproduces every reference array exactly, through the old calls and
through the new `runs.execute`. Four long functions still to do.

### Thu Oct 8 (build started 07:30; clock times are from the git log)

**01:30 (Oct 9): everything regenerated, nothing running.** The sensor
and actuator study (42 runs), both catalogues, the open-loop catalogue,
the placement study and the RL comparison all come from one code version.
RL: PPO, 51,508 steps in 60 min on the 16-stage perturbed circuit, five
identical episodes against the MPC with the same sensors — purity held as
well (99.2–99.5 % at worst against 99.4–99.5 %), a quarter less reagent,
three to fifteen times more NdPr lost (720–2,170 mol against 50–660).
The report's sections 6 and 7 and the README are written from these runs.

**23:18: the pump stop stays the one upset where doing nothing wins.** The
regenerated catalogue showed it going off spec under control (220 mol, an
interface out for an hour): the recovery the controller holds reads
nonsense with the feed off and low for hours after it returns, and it
chases it. Two fixes tried, both reverted: holding the handles while the
feed is off helped the full-information case (92 mol) and broke the
sensor-based loops on the same upset (purity 63 % with product, raffinate
and feed analysers); a recovery read against the feed of a residence time
ago helped the feed switch and made the pump stop worse again (620 mol).
Neither is an improvement to stand behind, so the controller is back to
before both (commit 7aaa2f9) and the report says what happens and why.
Catalogues, sensor study and RL regenerated once more on this code.

**22:15: a controller has to come back on time.** One more stall, in a
model state fed by estimates: the adaptive integrator crawled for minutes
in its retry ladder on the controller's model state after a pump stop. That
state now moves by implicit Euler with the circuit's own Jacobian, six
sub-steps, the same cost every step (one step of the linearization instead
was tried first and drifted the model state off the model over a run). The
estimator was moved to the same integrator and moved back within the hour:
its gain is built for the exact model, and at the analyser positions where
the gain is largest the cruder prediction turned a 5-thousandth purity
error into 15 points. When a linearization comes out numerically useless
the controller holds its handles and puts its model state back on the
model's rest point (putting it on the estimate was tried: a mid-cascade
estimate on a wrong model sits off the manifold, and every linearization
after it was useless too). Feed offsets back to a 50 % cap: at 100 % they
fed the controller's model a doubled feed and saturated it. The mid-cascade
analyser sets remain the fragile ones (7–14 failed solves on a feed
switch; the product and raffinate analysers none), which is the study's
finding. Everything is being regenerated once more on this code.

**21:03: three more things an estimator must not be allowed to do.** The
sensor study kept stalling on the mid-cascade analyser sets, each time for
a reason worth having: (1) with one 50 % cap on every offset, the estimator
explained what a wrong model could not with an organic flow 850 L/h above
what was commanded — a flow offset may now reach 15 % (a flowmeter is not
50 % wrong), a feed offset stays at 50 % (let it reach 100 % and it fed the
controller's model a doubled feed, which saturated it; a feed that really
doubles is what a feed analyser is for); (2) an interface
reading pushed through the Riccati gain rewrote the compositions, because
the model ties the interface tightly to the densities — an interface
reading now updates the interface and nothing else; (3) on a stopped feed
the estimate's recovery read 2260 % and the QP planned on it — what the
controller is told is clipped to what a purity and a recovery can be. With
these, the patent's two-analyser layout holds the feed switch as well as
product and raffinate analysers do; the differences between sensor sets
are now in the report's section 6, not in whether the loop survives.

**19:27: a zero that read as zero.** Analysers now read every component,
and a component the feed does not carry reads exactly zero with zero
relative noise; a zero on the diagonal of the measurement covariance turned
the Riccati equation singular, the gain enormous, and the estimator crawled
for minutes a step on any set with a product or raffinate analyser. Readings
and the gain carry an absolute floor of 1e-4 (chosen). The same run went
from stuck to 18 s. Also: the hydraulic drift upset is 10 cm (3 cm stayed
inside the window on this design: 9.5 m·h open loop, 0.02 under control),
and the Pr | Nd sweep is being rerun with the loading cap lifted, because
with the cap the 60-stage patent circuit sits at 92 % recovery on the
model's curve, no longer at its reported 99 %: the published plant runs its
organic fuller than my design rule allows, or its stages are better than
95 %. Both curves go in the validation chapter.

**19:01: measure what you control.** With the model perfect, two analysers
mid-cascade were enough. On the perturbed plant the estimator inferred the
product from them through a model that is a few percent wrong, reported
purity 99.98 % and recovery 92 % where the truth was 99.77 % and 95.2 %, and
the controller fixed a shortfall that was not there; purity went to 68 %.
Fixed three things (ARCHITECTURE.md 23): analysers read every component
(the Pr/Nd-only default was a Pr | Nd leftover; purity is about the
impurity); the controller takes purity and recovery from a product analyser
as measured and from the estimate only otherwise; the scrub liquor's acidity
is read back, which also cured the reflux-acid case (purity 47 % and 2240 mol
lost with the controller pushing a handle the plant was not following; 56 mol
lost against 689 open loop with the read-back). The placement study now
answers on the perturbed plant too: the best single analyser is at stage
25, late in the scrub (4.6 thousandths rms), the model's choice at stage 24
is close (4.8), the product stream 5.8, the ECJTU pair is 10.8, and a lab assay alone leaves the estimator
ten points off. Sensor study, catalogues and RL relaunched on this footing.

**18:33: the inverse crime, second pass — the controller was the weak part.**
The first run against the perturbed plant was a disaster worth keeping: the
plant sat at 98.9 % purity on the model's design handles, and the controller
cut organic, scrub and saponification to save reagent while purity fell to
96 % and the loss went from 156 mol (open loop) to 1042 mol in 20 h. Three
causes, each found by measurement and each now fixed (ARCHITECTURE.md 21,
22):

- *The design sat on a cliff.* Minimizing reagent, the optimizer loaded the
  organic to 95–99 % of capacity in every extraction and scrub stage, where
  D goes as the cube of what little free extractant is left. A loading cap
  of 85 % (chosen) costs 2 % reagent and buys a model that holds a few
  percent off the design.
- *The controller linearized the model at the plant's state.* A few percent
  off the model's own manifold the model's internal rates are large and its
  Jacobian is wrong in kind: slowest mode 15× too fast, steady-state purity
  gain per 100 L/h of organic +0.001 against −0.13 on the manifold, recovery
  gain +1.6 against +0.03. The same linearization on the manifold matches
  the full model to four decimals. The controller now keeps its own model
  state and takes what the plant reports as an output bias (integral action,
  the DMC way). Rate limits cut to a third.
- *The horizon could not see the inventory.* A reflux circuit's slowest mode
  is ~400 h; after a scrub cut the product rate read 160 % of the feed for
  hours while the recirculating inventory drained. The handles the plan ends
  on must now hold the floors at steady state too (DC gain).

Commissioning now ends with a trim: the controller, with full information,
takes the real plant from the model's design point to spec and every
scenario starts there. On the perturbed plant: purity 96.9 → 99.75 % in
10 h, recovery held at 95 %, no QP failures, reagent +17 % against the
design; the plant's true optimum (optimizer told the truth) is +1.4 %, so
about 15 % is what the model's error costs. On the matched plant the trim
moves nothing (reagent 2308 → 2306 mol/h). Feed switch under the new
controller on the perturbed plant: 1009 mol lost open loop, 77 under MPC,
nothing off spec. Everything is being regenerated on this footing; the 13:30
and 17:00 numbers below are superseded.

**17:00: breaking the inverse crime.** Until now the plant the controller
drives was the controller's own model, upsets and sensors apart — a
controller with perfect knowledge of the plant flatters itself. Added a
perturbed plant: same equations, different numbers from a seed (constants
±10 %, the chosen La+Ce one ±20 %, acid exponent 3.3 vs 3.0, stage
efficiencies 85–100 %, densities and hold-up a few percent off), unknown
to the controller, estimator and optimizer. Catalogue, sensor study and RL
to be rerun on it; those become the reported numbers, the matched plant the
reference. Still queued: the data-driven soft sensor from the plan.

**16:40: the drift case exposed one more blind spot.** The controller left a
drifting settler alone (0.385 → 0.365 m·h outside the window): its model had
the jackleg where it was commanded, so it expected the interface to fall
back on its own. Same disease as the flowmeter bias and the stopped feed:
the plant differs from the model in a way the controller is not told. Fix:
infer each settler's effective jackleg from where its interface sits (it is
at rest between control steps) and feed it forward. Excursion now 0.006
m·h. Controller catalogue and figures rerunning for the NdPr case; the
sensor study is not affected (its scenarios do not include the drift).

**13:30: the product is NdPr, not Nd.** A review caught it: NdPr (didymium) is
sold as one product, so the cut that matters is Ce | Pr (La and Ce out, Pr and Nd together
in), not Pr | Nd. Everything was built with the cut as a setting, so the code
does not change; every number does. Decision: the Ce | Pr cut becomes the main
case (`ndpr_*` run files, full Eneabba feed, Sm and heavier removed upstream);
the Pr | Nd results stay as the chapter where the model was checked against
the two published circuits (`prnd_*`). Weak point to state: the La+Ce
constant is chosen from an industrial separation-factor range, where the
Nd and Pr constants were fitted from measured tables. Interface centring went
into the controller before the reruns so one pass covers everything. NdPr
stage sweep running. Appendices A (model) and B (control) written, with
attribution for every borrowed equation and data table; no text or code was
copied from any source.

**12:20: sensor placement, judged fairly.** One analyser at stage 54 (end of
scrub) pins purity to ±0.01 points; the best second one (stage 58) adds
almost nothing; the patent's extraction + scrub pair is 13 % worse than the
single best. A late lab assay scores as well here because the metric does
not count delay; the closed-loop study does.

**12:05: the controller gets the feed flowmeter.** The pump-stop run under
full-state control came out worse than open loop (lowest purity 98.95 %,
12 QP holds, +20 % reagent). Cause: feed *flow* was not among the measured
disturbances the controller is given, only feed composition, so its model
kept feeding the circuit for two hours that the plant did not. A feed
flowmeter is a basic instrument, so it is now a given, and the recovery
ratio is floored at 5 % of the nominal feed so a stopped feed cannot blow
the QP up. Pump stop under control now: interfaces outside the window
0.73 → 0.084 m·h, no off-spec, +3 % reagent. Placement study rerunning
with every sensor set judged on the same estimator model (two disturbance
states), which the first pass did not do; sensor/actuator study resuming;
catalogue rerunning so all numbers come from one controller.

**11:50: near-ideal sweep done (k×20, 99.7 % stages).** 18 → 51 %, 20 → 62 %,
28 → 84 %, 36 → 93 %, 46 → 98 % recovery at 99 % purity; the sizes above 46
did not converge at that stiffness. The 95 % target is crossed near 40 stages
against 52 with 95 % stages. The sizing statement is now a bracket, 40–52,
set by stage efficiency; both figures are in the README.

**11:45: RL preliminary result.** PPO for one hour (13,920 steps) on the
18-stage circuit, episodes with a random feed switch plus random flowmeter
bias and a random slow mixer, same sensors and estimator as the MPC. Five
identical episodes scored both ways: purity held alike (99.45–99.47 % vs
99.45–99.49 %), Nd lost 1.6–4× more, reagent ~10 % more. The expected honest
finding; it goes in the README as such. Longer training on the plant circuit
is possible but not a priority.

**11:29: two robustness fixes from the long studies.** The sensor study died
twice: once on a NaN from an estimate with nothing in the product stage
(guarded: NaN-safe outputs, the controller holds its handles, the estimator
skips a senseless correction) and once in the plant integrator during the
pump stop under control, where empty stages reached D ~ 1e6 (a smooth cap
at 1e4, labelled chosen; baseline unchanged to four figures; plus a retry
ladder BDF → Radau → LSODA → sub-steps). The study is resumable and resumes
from its CSV. Sensor placement computed for an hour and died on a stale
attribute name in a line whose value was never used; it now caches as it
goes. Lesson written down: long runs save as they go, every time.

**10:43: acid floor settled; catalogue under the final controller done.** The
floor inside D does not move the operating point at all (3e-4, 1e-3, 3e-3 M:
recovery, purity, flows and reagent identical to 0.2 %). What the optimizer
sits on is its own lower bound on stage acidity, 1e-3 M (pH 3, chosen in
design.py). That bound is the operative number: plants do hold raffinate
pH near 3, but it is a choice and it is stated as one. The D floor can stay
at 1e-3 M as a numerical guard only.

**10:50–11:20: MPC made to work at plant size.** Re-linearizing every step
was correct but the QP was too big (24 horizon points × 64 interface rows,
8 moves × 64 jacklegs): >15 s a step. One jackleg move per step and
interfaces checked at five horizon points: 0.5 s a step, same behaviour.
Then the real finding: the plant baseline sat on the organic pump's
ceiling, so the controller had no room to answer a stronger feed and
doubled the scrub instead. Design now leaves 15 % pump headroom. Feed
strength +20 %: Nd lost 845 → 715 mol, purity and interfaces held, 33 %
more reagent to hold the 95 % recovery floor — a price question, reported
as such. Catalogue and sensor/actuator study running with this controller.
The ideal-stage sweep at k×1000 would not converge past 20 stages (too
stiff); rerunning at k×20 (99.7 % efficiency) later.

**Exact vs fast stages (10:10): a sensitivity that changes how the stage
count must be stated.** The exact equilibrium stage is the k→∞ limit of the
rate model, so the comparison is a convergence study (`06_*`). At the
designed operating point nothing much moves (purity 99.50 → 99.68 %,
recovery unchanged). But what the optimizer can get out of 18 stages goes
from **5 % recovery with 95 %-efficient stages to 60 % with ideal ones**.
Fractional extraction at β = 1.5 is that sensitive to per-stage efficiency.
So the "~52 stages for 95 % recovery" result is conditional on the 95 %
efficiency assumption (chosen, from Idaho's "three minutes to equilibrium");
the sweep gets an `--ideal` rerun to bracket it. This is the kind of thing to
state plainly: the stage count is a strong function of how well
each mixer-settler actually performs, and that is a measurable plant number.

**09:40–10:10: estimator and controller reworked after two honest failures.**
(1) The first Kalman filter made the estimate *worse* with more trust in the
readings. Cause: a 30 % feed change is not white noise on every state, and a
filter told so spreads the correction wrongly. Fix: disturbance states (feed
composition offset, organic flow offset) so the filter attributes what it
sees to the right cause — never more of them than the sensor set has
measurement rows. Result on the 18-stage circuit with two analysers: state
error 2.1 % → 0.2 %, feed estimate within 1–2 % of the true one, a +30 L/h
flowmeter bias estimated as +28. That estimate is handed to the MPC, which
is the Carester "flowmeter precision" problem closed. (2) The MPC fixed at
the operating point sent the moves the wrong way on a +20 % feed-strength
step (purity 99.5 → 95.7 %, whatever the tuning; the direction was wrong,
not the gain). Fix: re-linearize the circuit at the current estimate every
step (successive linearization), QP assembled numerically. Being verified.

**MPC working (09:20–09:35).** The nonlinear do-mpc controller was a dead
end: 1.4 s per IPOPT iteration, no convergence in 200, i.e. 90 s per step on
the smallest circuit. Replaced by the structure plants use: the nonlinear
steady-state optimizer sets the operating point and targets; a linear MPC
(the circuit's own equations linearized there, exact discretization, QP each
step) makes the moves. OSQP solves the plant-size QP (3.7 k variables, with
one slack per interface per step) in 0.02–2 s; qpOASES needed 20 s. Two
lessons on the way, both now in the code: weights must live within a few
decades of each other or the first-order QP solver fails (errors are now in
thousandths of purity, hundredths of recovery, cm of interface, fractions
of a rate limit); and purity and recovery must be **floors, not setpoints**:
as setpoints the controller pulled recovery *down* to 95 % to save reagent.
Feed switch on the plant circuit under MPC: purity held at 99.5 %, Nd lost
150 mol vs 162 open loop, reagent slightly lower. Modest, because 64 stages
absorb a feed switch on their own; the interrupt, the bias and the drift are
where it should earn its keep. Three controller tests green (20 in all).

**Stage-count study finished (09:05).** 54 optimizer solves, 40 min.
Nd recovery at 99 % purity against total stages: 18 → 15–18 %, 28 → 61–65 %,
36 → 80–82 %, 46 → 92 %, 58 → 97.5 %, 64 → 99 %, 78 → 99.6 %. The two
published anchors sit on the same curve: the Idaho 18-stage rig (high purity,
low recovery, as they reported) and the ECJTU 60-stage plant. The chosen
95 % target is crossed at about 52 stages. **Honest surprise:** with flows
optimized, the reflux scrub beats acid-only by only 2–4 points; the big gain
seen earlier came from fixing bad flows, not from the reflux. Every point
chose O/A at the 6.67 bound and scrub acid at the 2 M bound, so the flow
bounds are part of the answer and should be said so.

**Disturbance catalogue, first pass (08:30).** All eight upsets run
open loop on the plant-size baseline; figures in `figures/03_*.png`. Purity
moves little on a 64-stage circuit (worst: organic flowmeter +5 % → 98.4 %);
the feed interruption drops interfaces to 9 cm, well out of the window, for
about four hours. Two fixes before these numbers mean anything: the baseline
sat exactly on the 99 % spec, so any dip counted as a whole run off-spec
(design point moved to 99.5 %, spec stays 99 %), and the score now counts Nd
lost to the raffinate, which is what the interruption and the bias actually
cost. **Sensitivity checked later (10:43):** the acid floor in D does not shape the
operating point; the optimizer's pH 3 lower bound does. See below.

**First real result (08:05).** Operating points by optimization on the
model (src/design.py, IPOPT): the 18-stage Idaho circuit gives **18 % Nd
recovery at 99 % purity** (five different starts, same answer), the
ECJTU-size circuit (26 + 34 + 4 strip) gives **99 % recovery at 99 %
purity**. Both published anchors come out of the model as built. The full
sweep over circuit sizes is running (runs/stage_count.log). The baseline for
the disturbance work will be the ECJTU-size circuit at a reagent-minimized
operating point (recovery ≥ 95 %), so the flows are plant-like rather than
at the optimizer's bounds.

Two corrections made on the way: saponified extractant that finds no acid in
a stage now travels on with the organic (it was being lost), and D carries an
acid floor at 1e-3 M (below ~pH 3 the law stops applying; without the floor
the optimizer found "solutions" with a stage at zero acid extracting
everything, and the integrator failed on over-neutralized start-ups).

**Order change: stage-count study before the disturbance catalogue.** The
first operating-point scan on the 18-stage Idaho circuit showed that an
acid-only scrub cannot give purity and recovery together (99.5 % purity at
4 % recovery, or 95 % recovery at 75 % purity). The Idaho rig's own 99.7 %
run had a Nd-rich raffinate too; ECJTU runs 60 stages. A Fenske-type count
at β = 1.51 says about 23 stages at total reflux, so 40–60 with real flows.
The disturbances need a credible baseline, and that baseline comes out of the
stage-count study. Reflux scrub (strip product sent back) added as a setting
for the same reason. Simulator: 15 tests green.

**Parameters sourced (07:35–07:50).** What was found, and what was not:

- **Nd and Pr equilibrium:** fitted from the Idaho thesis single-metal tables
  (24 points, 20 used). With the textbook acid exponent of 3: K_Nd = 0.032,
  K_Pr = 0.021, separation factor **1.51**. The thesis measured 1.57–1.59 on
  mixed feeds; industry quotes 1.3–1.5 (Yin et al. 2018). The fit scatter is
  wide (K drifts with acidity, the thesis's own Fig. 16 shows it); a free
  exponent fits better but swings the separation factor 1.1–2.5, which the
  mixed-feed data do not support. Exponent is a setting, default 3.
  Reproducible: `notebooks/00_fit_equilibrium.py`.
- **Ce relative to Pr:** industrial β Pr/Ce 1.8–2.2 (Yin 2018) → 2.0. The
  La+Ce lump uses Ce's constant (conservative for the Pr product).
- **Sm relative to Nd:** nothing citable found. Carester Fig. 1 reads ≈14 off
  a log plot; handbooks are commonly quoted ≈6–7 (unverified). **Chosen 10.**
  Not load-bearing for the Pr|Nd cut.
- **Densities:** nothing citable for loaded phases. Chosen 0.85 g/mL organic
  with metal mass added; aqueous from water plus salt with an assumed molal
  volume. Only the difference and its drift matter, and both get swept.
- **Stage size and times:** Idaho Table 3 (120 mL mixer, 480 mL settler,
  3 min to equilibrium), scaled ×250 so throughput matches the ECJTU plant
  feed of 10 L/min.
- **ECJTU patent now read in full text:** 60 stages = 26 extraction + 34
  scrub, feed 10 L/min, two internal monitoring points, two flows
  manipulated, Elman (recurrent) neural network as the predictive model.
  Their words: "nonlinear, strong-coupling, severely lagged"; "on-site online
  detection has not been implemented yet"; "controlled by seasoned operating
  personnel based on their own experience".
- **Purity target 99 %** (Idaho "magnet applications"; commercial 99 % grade).

**Model designed** (ARCHITECTURE.md, "The model, in plain terms"): fast
rate-based stage first, jackleg interface with one-way coupling, three
sections with two aqueous paths (Idaho layout), saponification as an
equivalent base dose, the cut as a setting.

**Simulator running** (`src/circuit.py`, `src/simulate.py`, `src/config.py`,
`src/metrics.py`, `src/commission.py`; run files in `configs/`). Two things
learned on the way, both now in ARCHITECTURE.md:

- **Stage efficiency of a stirred stage is kτ/(1+kτ)**, not 1−e^(−kτ). The
  Kremser check showed it (stage 0 sat at exactly 600/601 of equilibrium).
  The nominal rate constant was giving 75 % stages; re-derived so "equilibrium
  in 3 min" means 95 %: k = 380 /h.
- **The Pr|Nd cut takes a didymium feed.** With the full monazite feed the
  La+Ce lump (68 %) loaded the organic four times more than Nd and purity
  stalled at 36 %. A plant does La/Ce and Nd/Sm cuts first; Eneabba's Pr:Nd
  is 26:74, the same didymium Idaho ran. Default feed changed accordingly;
  the full feed stays available for `cut = "LaCe"`.

- Started autonomous build.
