# Appendix B — the optimizer, the controller and the estimator

Formulations as implemented, in my notation. The methods are standard and
are named with their usual references; nothing here is copied from them.

**Status (October 19).** B.1 to B.9 are current for what they cover. B.1 gives
the design's purity target; the main case's controller aims higher (99.8 %,
`configs/ndpr_main.toml`). Pieces added later aren't written up here as
equations: the level loops under the MPC and the acidity floor
(ARCHITECTURE.md decisions 32 and 33), the guard and model registry (decisions
37 to 39), the feed-loss interlock and the learned estimator (PROGRESS.md,
phases 21 and 22). Some rows of the table in B.10 were overtaken by that work.
Rows 1 and 29 say so; row 6 still lists the interlock as a next step, and it
has since been built and run (PROGRESS.md, phase 21).

## B.1 Steady-state operating point (`src/design.py`)

Variables: the full state *x* and the free handles *v* (organic flow, scrub
flow, scrub acidity, saponified fraction). Two problems, solved in turn:

    maximize   recovery(x, v)
    subject to f(x, u(v)) = 0                      steady state of the model
               purity(x, v) ≥ purity_design        0.995 (spec 0.99 + margin)
               cᵃ, cᵒ ≥ 0,  h ≥ 1e-3 M,  v in its box

then, from that solution,

    minimize   reagent(v) = s·L_mono·Qᵒ + Qᵗ·hᵗ + Qˢ·hˢ         mol/h of base and acid
    subject to the same, plus recovery ≥ recovery_floor         0.95, or 90 % of the maximum

with *Q*ᵗ = *Q*_product,net + *Q*ˢ tied (reflux), *Q*ᵒ ≤ 85 % of the pump
maximum (headroom), and in every stage *j*

    Σᵢ cᵒᵢⱼ ≤ 0.85 · L_tot / 3                    organic loading at most 85 % of capacity

(a margin against saturation; without it the optimizer loads every
extraction and scrub stage to 95–99 %, where *D* ∝ *L*_free³ makes the
model too steep for any controller to work from). Solved with IPOPT (Wächter & Biegler, 2006) through
CasADi (Andersson et al., 2019) with exact first and second derivatives of
the model. Several starting points gave the same optimum on the circuits
checked. The patent cited in the report describes the same step as
"steady-state optimization" of the set values.

## B.2 Model state, output bias, linearization and prediction (`src/mpc.py`)

The controller keeps its own model state *x*_m, started at the model's
steady state for the initial handles and run forward each step with the
nonlinear model under the inputs it applied, by implicit Euler in six
sub-steps with the circuit's own Jacobian (a fixed cost every step; the
adaptive integrator could crawl for minutes after a pump stop, and a
controller has to come back on time). What the plant reports (the
true state in the full-information studies, the estimate of B.5 otherwise)
enters only through an output bias,

    b ← (1 − α) b + α (y(x_reported, u) − y(x_m, u)),      α = 0.5

added to every predicted output. This is the DMC structure (Cutler and
Ramaker, 1980): the model is used where it is right, on its own trajectory,
and the plant–model difference is carried as a constant the controller
plans around, which gives integral action. Linearizing at the reported
state instead was tried first; a few percent away from the model's own
steady states, the model's Jacobian has neither the slow mode of the metal
held up in the circuit nor the steady-state
gains of the circuit, and the plans built on it were wrong in sign.

Three things guard what enters the bias:

- **A product analyser outranks the estimate.** When a product-stream
  analyser has a fresh reading, its purity (and its recovery, if the reading
  is prompt) is taken as measured; the estimate stands in for the rest.
- **Clipping.** A reported purity is clipped to [0, 1] and a recovery to
  [0, 1.5]; the bias itself stays within ±0.2 on purity and ±0.5 on recovery.
  A stopped feed once made an estimate report a recovery of 2260 %.
- **Re-anchoring.** If a linearization comes out non-finite or with entries
  above 1e8 (1e6 for the affine term), the handles are held for that step and
  *x*_m is reset to the model's rest point.

Two offsets the plant reveals are fed forward like measured disturbances:
the scrub acidity read back on the scrub line (what the plant gets minus what
was commanded), and a jackleg offset per settler, inferred each step from
where the interface sits against where the model says the commanded jackleg
puts it (hydraulic drift).

At the model state *x*_c = *x*_m, handles *u*_c and measured disturbances
*d*_c (feed composition, feed acid, feed flow, estimated organic-flow
offset), the model is linearized with the affine term kept:

    ξ̇ = A ξ + B δu + E δd + f_c,       ξ = x − x_c,  δu = u − u_c,  δd = d − d_c
    y  = y_c + C ξ + D_u δu + D_d δd,   y = [purity, recovery, z₀ … z_{N−1}]

A, B, E, C, D_u, D_d and f_c come from CasADi's Jacobians of the circuit's
own equations. Discretization over the step *T*_s = 0.5 h is exact for
inputs held over the step, by the matrix exponential of the augmented
matrix [[A, B, E, f_c], [0, 0, 0, 0]].

Over a horizon of *H*_p = 24 steps with *H*_u = 8 moves (one move per
jackleg), the predicted outputs are affine in the moves Δu:

    Y = Y₀(x_m, u_prev, d) + 1 ⊗ b + M Δu

with *Y*₀ the response if the handles are held, *b* the bias, and *M* the
move-to-output map. Both are rebuilt numerically at every step (successive
linearization; see Rawlings, Mayne and Diehl, *Model Predictive Control:
Theory, Computation, and Design*, 2nd ed., for the family of methods).

The same linearization gives where the handles would leave the plant in the
long run: the model's rest point at the current handles and the
steady-state (DC) gain of a move,

    y_ss(u) ≈ y_c − C A⁻¹ f_c + b + (D_u − C A⁻¹ B)(u − u_c)

The metal held up in a reflux circuit (its inventory) settles over hundreds of hours, far past
any horizon; without this term the controller took credit for product that
was only the recirculating inventory draining.

## B.3 The quadratic program

Handles and their bounds (chosen): organic flow 600–4700 L/h (from
`[design]` in the parameter file), scrub flow 50–2000 L/h, scrub acidity
0.05–2 M, saponified fraction 0–0.6, each jackleg 0.30–0.48 m. With a reflux
scrub the strip flow follows the scrub flow (net product 300 L/h).

Variables: Δu, and slacks *s*_spec, *s*_target (purity, in thousandths),
*s*_rec (recovery, in hundredths), *s*_lo, *s*_hi (interfaces, in cm, at
five horizon points), *s*_ss,p and *s*_ss,r (purity and recovery at steady
state). Cost, with all weights of order 0.01–100 because every term is
scaled to its natural unit:

    Σ_k ‖Δu_k / rate‖²_W_move                      moves as fractions of their rate limits
      + w_hold · Σ_k ‖(u_k − u_prev) / rate‖² / H_p    holds against chatter between terms
      + w_spec · Σ s_spec²  + w_target · Σ s_target²  + w_rec · Σ s_rec²
      + w_iface · Σ (s_lo² + s_hi²)  + w_center · Σ ((z − z_set)[m] · 100)²
      + w_target · s_ss,p²  + w_rec · s_ss,r²
      + w_reagent · Σ_k (∇reagent · u_k) · T_s / 1000

Constraints:

    purity_k   + s_spec,k /1000 ≥ spec,        purity_k + s_target,k /1000 ≥ purity_target
    recovery_k + s_rec,k  /100  ≥ recovery_target
    z_lo ≤ z_k + s_lo,k /100,    z_k − s_hi,k /100 ≤ z_hi
    purity_ss(u_end) + s_ss,p /1000 ≥ purity_target,   recovery_ss(u_end) + s_ss,r /100 ≥ recovery_target
    u_min ≤ u_k ≤ u_max,   |Δu_k| ≤ rate,   all slacks ≥ 0

with *u*_end the handles the plan ends on. The weights, as named in
`MPCSettings`: w_purity_floor = 100 (written w_spec above), w_purity = 1
(w_target), w_recovery = 1 (w_rec), w_interface = 1 (w_iface),
w_center = 0.02, w_reagent = 0.01, w_hold = 0.02, and 1 on every move. The
centering term is per cm², like the interface slacks. It was fixed on
13 October (ARCHITECTURE.md, decision 26); before that a units slip made it do
nothing. Every result in the report is from the fixed code. Its weight was
swept from 0 to 0.02 (`notebooks/15_*`): the feed switch, drift and flowmeter
bias barely move, and the pump stop jumps around with no pattern (B.10 #6).
Rate limits per half-hour step
(chosen): organic 50 L/h, scrub 10 L/h, scrub acid 0.05 M, saponified
fraction 0.01, jackleg 1 cm; the plant circuit's long-run purity moves about
13 points per 100 L/h of organic, so the handles move gently. Targets are
floors: being above purity or recovery costs nothing. Solved with OSQP
(Stellato et al., 2020) through CasADi's conic interface, warm started, with
its step-size retuning on a fixed interval (`adaptive_rho_interval` = 25;
by default the interval follows OSQP's own setup time, which made the same
QP solve on one run and fail on the next). Every output constraint has a
slack, so the QP always has a solution; when OSQP still reports failure
(seen on plants far off spec, where the slack costs run into the millions),
the same QP goes to qpOASES (Ferreau et al., 2014), an active-set solver.
Only if that fails too are the handles held. Each step records which solver
answered (ARCHITECTURE.md, decision 30).

Two ways of ranking the spec above everything else are in the code, both
off by default, because neither helped (decision 30, REPORT.md 5.3):

- `ranked`: two QPs per step. The first has the same constraints and a cost
  on the spec slacks *s*_spec alone (plus 1e-6 on every variable, so it has
  one answer); the second is the QP above with each *s*_spec,k capped at the
  first one's value plus 0.05 thousandths (`ranked_tolerance`).
- `w_purity_floor_linear`: one QP, with a straight-line term
  w_lin · Σ s_spec added to the cost (an exact penalty), tried at 1000 and
  100.

## B.4 Sensors (`src/sensors.py`)

A reading of sensor *m* at time *t* is

    y_m(t) = C_m · x(t − delay_m) + offset_m + ν_m,     ν_m ~ N(0, σ_m²)

available only on its sample ticks. *C*_m picks a concentration, an
acidity, an interface height, or the linear density map (A.9); the feed
analyser reads the feed composition. Two instruments are taken as given
rather than studied: the feed flowmeter (its reading is the feed flow the
controller uses) and a pH probe on the scrub liquor line (B.2). The organic
flow isn't measured independently: a flowmeter that reads the command back
is why a flowmeter bias is invisible. Noise, delay and sampling per kind are
chosen and listed in the parameter table; the noise standard deviation is
never below 1e-4 in the reading's units.

## B.5 Estimator (`src/estimator.py`)

State augmented with disturbance states *d* (offsets on the feed
composition, most abundant components first, and on the organic flow), as
random walks — the disturbance-model approach to offset-free estimation
(Muske and Badgwell, 2002; Pannocchia and Rawlings, 2003):

    [x; d]_{k+1} = [[A_d, E_d], [0, I]] [x; d]_k + w_k,     y_k = [C, 0] [x; d]_k + ν_k

Never more disturbance states than the sensor set has measurement rows,
and fewer, down to none, until the Riccati equation can be solved. The
gain *K* is the steady-state Kalman gain from the discrete Riccati equation
for that pair, computed once on the model linearized at the operating point,
with process noise relative to each state's size there and the offsets
wandering by 0.5 % of their size per step (chosen).

Each step: predict with the **nonlinear** model under the commanded inputs
plus the estimated offsets (the feed flowmeter's reading included); correct
with *K* times the innovations, each innovation formed against the estimate
at the time its sample was taken (fixed-lag approximation for delayed
readings); gate every correction to 30 % of the state's size; keep
concentrations non-negative and below three times the largest at the
operating point, and acid above 1e-4 M; keep a feed offset
within 50 % of nominal and the flow offset within 15 % (chosen: a flowmeter
is not 50 % wrong, and a feed that really doubles is what a feed analyser is
for; let through, it saturated the controller's model). The prediction
uses the adaptive integrator, not the controller's implicit Euler: the gain
is built for the exact model, and where it is largest a cruder prediction
each step was amplified into a 15-point purity error. The adaptive
prediction gets a budget of 20,000 model evaluations per step (chosen, a few
hundred times what a normal step needs); past that the step falls back to
implicit Euler and is counted in `slow_predictions` (ARCHITECTURE.md,
decision 27). Some estimates made the adaptive integrator crawl, millions of
tiny steps for one half hour; in that case 4 of 80 steps fell back. A
controller has to come back on time. The plant's own simulation has no
budget. (The RL environment uses implicit Euler here too, for speed.) No reading's noise is taken below
1e-4 in its units, so a component the feed does not carry does not read as a
perfect zero and turn *R* singular. An interface reading updates the interface state directly
and takes no part in the Riccati gain: the model ties the interface tightly
to the densities, and a 2 mm reading through that coupling rewrote the
compositions. With no sensors the estimator is a model run and the offsets
stay zero.

What the controller receives: the state estimate, the feed estimate (when
there is no feed analyser) and the organic-flow offset, which enters the
prediction as feedforward through the model's own flow column.

## B.6 Transition cost (`src/metrics.py`)

Over a run of length *T*:

    off-spec       ∫ (product metal rate) · [purity < spec] dt                 mol
    lost excess    ∫ (product-side metal rate to the raffinate − its value at the start) dt    mol
    interface      ∫ Σ_j max(0, z_lo − z_j) + max(0, z_j − z_hi) dt            m·h
    reagent        ∫ (s·L_mono·Qᵒ + Qᵗ·hᵗ + Qˢ·hˢ) dt                          mol
    time to settle the last time purity leaves a ±0.002 band around its final value

The same five numbers score the open loop, the controller with any sensor
set, and the learned policy. Off-spec counts product made while purity is
below the 99 % spec, so two runs that both dip near it can differ by a few
hundred mol for a small difference in purity.

## B.7 Sensor placement (`notebooks/07_*`)

For a candidate sensor set, the steady-state estimation covariance *P* from
the Riccati equation above (two disturbance states for every set, so sets
are comparable), projected onto product purity with its gradient *g* at
the operating point: σ_purity = √(gᵀ P g). It counts precision and sampling,
not delay, and says so.

## B.8 The plant the controller drives (`src/mismatch.py`)

The controller, the estimator and the optimizer plan on the nominal
parameters. The plant they drive is built from a perturbed copy, drawn from
a seed (1) so that every run sees the same plant (`notebooks/14_*` draws
seeds 1 to 20 for the catalogue, ARCHITECTURE.md decision 29):

    K_i  →  K_i · exp(ε_i),      ε_i ~ N(0, 0.10²)   (0.20 for the chosen La+Ce constant)
    n    →  n + 0.3              acid exponent 3.3 in the plant against 3.0 in the model
    k_j  →  k · κ_j,             κ_j ~ U(0.85, 1)    transfer rate, per stage
    ρ⁰_org → ρ⁰_org (1 + ε),     ε ~ N(0, 0.02²);    φ_V likewise at 0.10
    V_mix → V_mix (1 + ε),       ε ~ N(0, 0.05²)

Same equations, different numbers, fixed for the run. On the plant circuit
this puts the plant's purity at 96.9 % on the handles the model designed for
99.5 %. Four plants can be chosen (`RunSpec.plant`, or a run file's
`[plant] mismatch` in the notebooks): the matched plant (the model itself,
the reference), this perturbed one, and the settler and entrainment plants
of Appendix A.12, which have mechanisms the model lacks.

## B.9 Commissioning trim (`src/commission.py`)

After start-up on the design handles, the controller is run on the plant
with full information for 30 h with no upset; the handles it ends on are
the plant's operating point, the plant is brought to rest there, and every
study starts from that point. On the matched plant nothing moves. On the
perturbed plant circuit the trim takes purity from 96.9 % to 99.75 % in
10 h with recovery held, at 17 % more reagent than the design; the plant's
own optimum, found by the optimizer told the true parameters, is 1.4 % more,
so the rest is what the model's error costs.

## B.10 How each control choice holds up

Each choice is marked **Why** when there's a measured result here or it's
standard practice, and **Judgement call** when it was picked because it
seemed reasonable or worked, without a study. Ranked by how much it matters
to the results. The code carries the same notes as `Why:` and
`Judgement call:` comments where each choice is made.

| # | Choice | Verdict | What would settle it |
| --- | --- | --- | --- |
| 1 | Reported numbers are on one perturbed plant (seed 1, B.8) | Partly settled. The catalogue was run on 20 seeds (ARCHITECTURE.md decision 29, REPORT.md 5.3): on the 15 plants that can make both targets the controller beats open loop on the feed switch and feed acid on all 15, the stronger feed on 14, the strong scrub acid on 14, the flowmeter bias on 8 (open loop clean on the other 7); the pump stop is worse on 5. Five plants can't make both targets at all. Corrected (ARCHITECTURE.md decisions 31 to 34): those five came from a generator that drew separation factors outside the published ranges; on plants drawn inside them, the main case (sized stages, margin, acidity floor, level loops) keeps all 20 on spec through the feed and acid upsets. Judgement call still: the sensor study, structural matrix and RL are on seed 1 | The sensor study and the structural matrix on several seeds |
| 2 | Linearize on the controller's own model state, plant in through an output bias (decision 21) | Why: measured. At the plant's state the gains had the wrong sign; on the model's trajectory they match the full model to four decimals. The DMC structure (Cutler and Ramaker) | Settled |
| 3 | Re-linearize every step (decisions 16, 18) | Why: measured. Fixed at the operating point, the moves went the wrong way on a +20 % feed | `mpc_fixed` turns off both re-linearization and the bias, so its gap to the MPC in REPORT.md 5.2 doesn't split the two. One run with each turned off alone |
| 4 | Hand-built linear QP, not a nonlinear MPC framework (decision 16) | Why: measured. do-mpc: about 1.4 s per IPOPT iteration, no convergence in 200, so 90 s a step at best on the 18-stage circuit. This QP: under a second | Settled for this study. A nonlinear MPC on the plant circuit, if a faster one is ever built, would show what the linearization costs |
| 5 | Steady-state floors on the handles the plan ends on (decision 21 (3)) | Why: measured. Without them the controller borrowed from the held-up metal. Judgement call: the long-run gain comes from one linearization | Long-run gains checked against full-model steady states after single moves |
| 6 | Recovery as product now over feed now, with no rule for loss of feed | Judgement call; the known failure. On seed 1 the controller now does about as well as open loop on the pump stop (nothing off spec, 27 mol lost; REPORT.md 5), but the result is fragile: in the centering-weight sweep the entrainment plant lost next to nothing at some weights and about 1,480 mol at another, and over 20 seeds the controller is worse than open loop on 5 of 15. Two simple fixes each hurt another case | An interlock as a plant has (freeze the handles below a feed-flow threshold; reset bias and model state from the plant when feed is back and settled), run on the whole catalogue and several seeds |
| 7 | Scoring: off spec all-or-nothing at 99 %, losses over 40 h only (B.6) | Judgement call. 215 against 0 mol can be a small purity difference. The ~400 h inventory mode isn't in a 40 h score | Purity shortfall integrated over time next to off spec; the change in held-up metal added to losses, or scoring at 200 h |
| 8 | Output bias in the MPC, disturbance states only in the estimator (B.2, B.5) | Why: offset-free at steady state, the industrial norm (Qin and Badgwell). Judgement call: a step-on-the-output bias follows an input-side gap slowly (Muske and Badgwell; Pannocchia and Rawlings). Bias filter 0.5 not swept | The feed switch with an input-disturbance model in the prediction; the filter at 0.2, 0.5 and 1.0 with a noisy sensor set |
| 9 | Estimator: random-walk offsets, no more than the readings support | Why: the disturbance-model approach (Muske and Badgwell; Pannocchia and Rawlings); measured, decision 18 | Settled |
| 10 | Estimator: one Kalman gain, computed at the operating point | Judgement call. The MPC re-linearizes; the filter doesn't. Far from the point the gain is for another model | Gain recomputed every few steps, purity error compared on the feed switch |
| 11 | Estimator guards: gate 30 %, caps 50 % feed and 15 % flow, concentrations under 3× | Why (that they exist): measured failures, decision 23 (4). Judgement call (the numbers): picked from those failures, not swept | Gate and caps at half and double on the sensor-set matrix |
| 12 | Interfaces set directly, outside the gain | Why: measured; a 2 mm reading rewrote the compositions (decision 23 (5)). Judgement call: what an interface says about density is thrown away | A small separate gain from interface to density, tried on the pump stop |
| 13 | Adaptive integrator in the estimator, implicit Euler in the controller; the estimator's prediction budgeted at 20,000 evaluations, then implicit Euler (B.5) | Why: measured both ways (decision 23 (4)); the budget because a crawling prediction hung runs (decision 27). Judgement call: the 20,000 | Count slow_predictions per run in the matrix |
| 14 | Feedforward: measured feed, scrub acid read back, jackleg offset inferred | Why: measured (decisions 18b, 23 (3)). Feedforward is the industrial norm. Measured cost: with the ECJTU pair plus a feed analyser it overreacts (REPORT.md 6). Judgement call: a pH probe at 0.05–2 M acid is assumed | What instrument a plant really uses on the scrub line |
| 15 | Product analyser outranks the estimate; recovery only from a prompt reading | Why: measured (decision 23 (2)); a measured controlled variable is used as measured | Settled |
| 16 | Soft output floors, hard handle bounds; targets as floors; spec weighted 100× the target | Why: keeps the QP feasible; range control is the industrial form (Qin and Badgwell). Judgement call: quadratic-only slack cost, so a floor gives way a little when the reagent term pulls. Ranking tried (decision 30): a two-stage QP with the spec first, and a linear term on the spec slacks at 1000 and 100. Neither helped: the two-stage form made plants that can make spec worse (seed 2 down to 62 % on the feed switch), the linear term changed little. Both kept as settings, off. On the 5 of 20 seeded plants that can't make both targets (seed 3 trims to 49 %) the gap is the model's gains, not the ranking (#29). Corrected: those plants were outside the published ranges (decision 31) | Settled for ranking |
| 17 | Reagent as a linear cost, weight 0.01; recovery held at any reagent cost | Judgement call. Not tied to prices (ARCHITECTURE.md, Open) | Prices for NdPr, caustic and acid |
| 18 | Horizons and timing: 0.5 h step, 24 steps, 8 moves, one jackleg move, interfaces at 5 points | Why for one jackleg move (interfaces settle in minutes). Judgement call for the rest; not compared | Hp 12 and 48, Hu 4 and 12, interfaces checked every step, on the catalogue |
| 19 | Weights, scaling and rate limits (B.3) | Why for scaling to natural units. Judgement call for the values; rate limits cut to a third after decision 21, not swept | Each at half and double on the catalogue |
| 20 | Clipping and re-anchoring (B.2) | Why (that they exist): measured; an estimate said recovery 2260 %. Judgement call: limits and the 1e8 threshold are orders of magnitude; re-anchoring goes to the rest point of the operating-point handles, not the current ones | Count clips and re-anchors per run in the matrix |
| 21 | Commissioning trim, 30 h with full information (B.9) | Why: plant practice, and every case starts from the same point. Judgement call: 30 h fixed, "handles stopped" not checked | Handle movement logged over the trim's last hours |
| 22 | Two layers, optimizer offline only (decision 16) | Why for the layout (Qin and Badgwell). Judgement call: no re-optimization when the feed changes for good | design.py re-run on the new feed, reagent compared with where the MPC settles |
| 23 | Sensor model: Gaussian noise, fixed delay, read on the 0.5 h grid | Judgement call. Delays under 0.25 h round to zero; numbers chosen (ARCHITECTURE.md, Open) | Vendor data; runs with analyser drift and dropped readings |
| 24 | Sensor placement by estimation covariance (B.7) | Why: the standard model-based ranking; checked on the perturbed plant: the same shape, best one stage later, numbers 40× larger | Settled as a first screen; delay is left to notebook 08 |
| 25 | Interface centering term, w_center 0.02 per cm² (B.3) | Fixed on 13 October and every closed-loop result rerun (decision 26), and again after decision 30. Weight swept 0 to 0.02 on six cases: feed switch, drift and flowmeter bias barely move (drift a little better with it); the pump stop jumps around at every weight | Settled for everything but the pump stop, which waits on #6 |
| 26 | RL comparison (REPORT.md 7) | Why: same plant, sensors, estimator, upsets and score. Judgement call: the policy has about 3× the MPC's rate limits and a 4000 L/h ceiling against 4700; one hour, one seed, five episodes | Retrain with the MPC's limits, several seeds, longer training |
| 27 | Structural-mismatch plants (decision 25) | Why: tests what a perturbed copy of the model can't. Judgement call: entrainment numbers chosen, not yet combined with the perturbed numbers or sensors | The same matrix with perturbed numbers on top and real sensor sets |
| 28 | OSQP on a fixed step-size schedule, qpOASES as a backup (B.3) | Why: measured. On plants far off spec OSQP called a QP that always has a solution infeasible, and the controller held still; it also wasn't repeatable there (decision 30). Judgement call: the interval 25 and the choice of qpOASES. On seed 17 both still fail on most steps of four upsets | Why seed 17 fails; a check that the two solvers agree where both answer |
| 29 | Fixed model gains: the controller's linearization comes from the nominal model, re-linearized but never corrected from plant data | Judgement call; the known gap on plants far from the model. On the 5 of 20 seeded plants that can't make both targets the gains are the right sign and the wrong size (seed 3: about fifteen times too small on purity per unit scrub flow), so the controller moves too timidly and doesn't reach spec in 40 h (decision 30, REPORT.md 5.3). Corrected (decisions 31 to 34): the five were outside the published ranges, and the "fifteen times" compared different operating points. On realistic plants the model's error does send the nominal-model MPC the wrong way on the first plant; stage sizing and the acidity floor fix that on the main case | Estimating the separation factors and stage efficiency online (Open) |

## References named

- A. Wächter, L. T. Biegler, *Math. Program.* 106 (2006) 25–57 — IPOPT.
- J. A. E. Andersson et al., *Math. Program. Comput.* 11 (2019) 1–36 — CasADi.
- B. Stellato et al., *Math. Program. Comput.* 12 (2020) 637–672 — OSQP.
- H. J. Ferreau et al., *Math. Program. Comput.* 6 (2014) 327–363 — qpOASES.
- J. B. Rawlings, D. Q. Mayne, M. M. Diehl, *Model Predictive Control:
  Theory, Computation, and Design*, 2nd ed., Nob Hill, 2017.
- C. R. Cutler, B. L. Ramaker, Dynamic matrix control, *Proc. JACC*, 1980.
- K. R. Muske, T. A. Badgwell, *J. Process Control* 12 (2002) 617–632.
- G. Pannocchia, J. B. Rawlings, *AIChE J.* 49 (2003) 426–437.
- S. J. Qin, T. A. Badgwell, A survey of industrial model predictive control
  technology, *Control Eng. Pract.* 11 (2003) 733–764.
