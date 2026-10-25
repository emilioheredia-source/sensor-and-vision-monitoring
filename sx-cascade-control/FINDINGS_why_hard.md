# Why the NdPr split is hard to control, and what the literature says about it

Working notes, started 2026-10-16. Model results are
from notebooks 19-24; the literature is listed under Sources below (page numbers are the
printed pages).

## The short version

1. The purity of the product depends on the ratio of organic to scrub flow raised to the power
   of the number of stages. A 1 % error in that ratio moves the impurity by roughly N %, where
   N is the number of purifying stages. That's why the window is a few percent wide. It's a
   property of any sharp countercurrent split, not of our model or controller.
2. Our design sat right on the edge of that window, because we chose the cheapest point that
   just meets both targets with the stages we had. The Chinese design method (Xu Guangxian's
   school) deliberately doesn't do that: it puts the scrub about 60 % above its minimum.
3. The minimum flows go as 1/(beta - 1). A plant whose separation factor is 1.8 instead of 2.0
   needs about 25 % more. That's exactly the 15-25 % more organic our hard plants needed.
4. Because of 1, the plant is ill-conditioned the way a high-purity distillation column is. A
   controller that inverts a model can be sent the wrong way by a small error in that model, and
   on the realistic plants the model's error is larger than that limit.
5. Plants deal with it by feedback on compositions inside the cascade (monitoring stages where
   the profile is steep, 5-15 stages from the outlets), trimming organic and scrub. That's the
   same thing our steady-state analysis found.

## 1. Why the window is narrow: purity goes as a power of the flow ratio

In a section where one element is being purified, the amount of impurity drops by the same
factor from one stage to the next. Cheng, Wu, Liao and Yan (J. Rare Earths 31 (2013) 169-173)
call this factor the adjacent stage impurity ratio. In the stages away from the feed and the
outlets it is

    extraction section   g = beta x S / W
    scrub section        g = beta x W / S

where S is the metal carried by the organic and W the metal carried back by the scrub aqueous
(eqs 14 and 19, pp. 171-172). Over N such stages the impurity drops by g to the power N. So:

  - both sections need g > 1, which means 1/beta < S/W < beta. The two sections pull the ratio
    in opposite directions; the window is between them.
  - the product's impurity goes as (S/W) to the power -N. With 10 purifying stages, a 1 % error
    in S/W changes the impurity by about 10 %. With 20, about 20 %.

That's the mechanism behind the narrow band we found (notebook 19: 1.8 L/h of scrub at 28
stages, a few percent at 44). It isn't a modelling artefact.

## 2. Why our design sat on the edge

Xu's theory (Hydrometallurgy of Rare Earths, 2018, chapter 3) gives the edges directly:

    minimum extraction amount   S_min ~ 1/(beta - 1) + f_A        (eq 3.83, p. 412)
    minimum scrub amount        W_min ~ 1/(beta - 1)              (independent of the feed)

and the stages needed go to infinity as the flows approach these minima (p. 425). The design
rule the book uses is W = k / (beta^k - 1) with k = 0.7 by default (eqs 3.162, 3.166,
pp. 438-441). At beta = 2 that puts W about 60 % above W_min (p. 454). Chapter 4 says the same
in practice: pick the point "a little right side of the lowest point" of the
stages x organic curve (pp. 560-562). The book notes its formulas are least accurate close to
the minima (p. 431), and it doesn't design there.

Our design step minimized reagent at a fixed stage count with purity and recovery as hard
limits. With the stages we had (8 + 20), that's the k -> 1 corner: the cheapest flows that the
installed stages can just about manage. Every real plant that separates a little worse than
the model is then off spec.

## 3. Why the hard plants needed 15-25 % more organic

W_min = 1/(beta - 1): at beta = 2.0 it's 1.00, at 1.8 it's 1.25, 25 % more (book, p. 412).
Our hard plants are the seeds with Pr/(La+Ce) of 1.77-1.96 against the model's 2.0. The
literature agrees with the size of the effect. Yang et al. (below) needed +17 % organic and
+23 % scrub for a feed composition change in their Pr/Nd example.

The book has no explicit margin for an uncertain separation factor. Its defences are: design
on the smaller of the extraction-section and scrub-section factors (p. 443), use the separation
factor measured over the stages on the real plant (p. 583), and re-simulate when the feed
changes. It also says one set of flows suits one feed composition "(a small range of variation
is tolerated)" and that otherwise "the quality of the product would not be ensured" (p. 556).
And the effective Pr/(La+Ce) factor changes with composition along the cascade (eqs 4.24-4.26,
p. 584): it isn't one number, which adds to the uncertainty.

## 4. Why a model-based controller struggles: ill-conditioning

Notebook 24 (sized plant). The long-run gain from (scrub flow, organic flow) to
(log impurity, log loss) at the design point:

    [[-27.6, 32.8],
     [ 14.9, -19.8]]       condition number 43, relative gain 9.6

The two flows push each output in opposite directions with almost the same strength, so their
difference (the ratio) has a huge effect and their common move (both up together) a small one.
That's the same structure as a high-purity distillation column run on reflux and boil-up
(Skogestad, Morari and Doyle, IEEE TAC 1988). Using ratio handles (scrub/organic, and organic
with the ratio held) makes it better (condition number 18, relative gain 2.4), but not good.

Note (Oct 22): the matrix above isn't the long-run gain. Notebook 24 solved each +-0.5 % step
from the design state with the default steady-state ladder, and Newton fails on those steps, so
each one returned the state after 250 h, still moving (|dx/dt| about 2e-4; the warning didn't
show). Solved to rest (800 h before the polish, |dx/dt| 1e-11, unchanged by 3,000 h more), at
the same design point (purity 99.466 %, recovery 95.003 % both ways):

                              as stored (Oct 16)            converged (Oct 22)
    G (log imp.; log loss)    [[-27.6, 32.8], [14.9, -19.8]]  [[-168.4, 202.0], [5.1, -8.1]]
    condition number          43                              204
    relative gain             9.6                             4.0
    ratio handles             18 and 2.4                      87 and 1.5

The conclusion stands and is stronger: the plant is ill-conditioned, more than the stored numbers
said, and ratio handles help. The model-error numbers below and the per-plant columns of
runs/conditioning_*.csv were computed the same way and haven't been rerun; treat them as
unconfirmed until notebook 24 is rerun with converged solves.

How wrong the model's gains are, as the relative error sigma_max((G_plant - G_model) G_model^-1)
(above 1 means a controller that inverts the model can move the plant the wrong way in some
direction):
  - at the design handles: 0.6-0.8 on the hard plants, 2-20 on the easy ones
  - at each plant's own design point (least reagent at 99.5 % and 95 %): above 1 on 17 of 19
    plants, up to 130 (corrected Oct 21 from "15 of 19"; runs/conditioning_ndpr_plant_sized.csv)

Note (Oct 24): notebook 24 rerun with every steady state solved to rest (--settled; written to
runs/conditioning_ndpr_plant_sized_settled.csv, the first file kept). Old and new, sized plant:

                                              as stored (Oct 16)       settled (Oct 24)
    model at the design point, condition no.  43.2                     203.4
    at the design handles, the ten plants
      that were 0.6-0.8 (seeds 2, 3, 6, 9,
      10, 11, 14, 16, 17, 20)                 0.58-0.77                0.91-1.01
    at the design handles, the other ten      1.5-20.7                 1.5-24.6
    at each plant's own design point          above 1 on 17 of 19,     above 1 on 19 of 19,
                                              0.79-130.8               3.1-28.0
    condition number there (plant)            6.4-95.2                 175.4-427.6

The conclusion holds and is stronger: at its own design point every plant's gains differ from the
model's by more than the limit an inverting controller can stand. The size of the error is smaller
than the stored "up to 130" said; that came from steps that hadn't settled. How sure: 53 of the
solves, on 13 of the 20 plants and most of them on the plants that separate easily, still moved
slightly after 4,000 h (|dx/dt| up to about 1e-4, the slow inventory mode), so those plants'
individual values are less certain. The six
plants where every solve settled (seeds 3, 6, 9, 10, 17, 20) give 17.2-28.0 at their design
points, all far above 1. The ndpr_plant run file (all solves settled) gives the same picture:
above 1 on 18 of 18 at the design points. The design-point condition number of the model matches
the Oct 22 note (203.4 against 204).
That's the closed-loop result of notebook 21 in numbers: the MPC with the nominal model moved
the organic the wrong way on all seven hard plants; with the true model it brought all seven to
spec.

The book quotes the same diagnosis from the Chinese control work: the process has "large input
and output; long time delays; nonlinear, multivariable and time-variant, nonminimum-phase... It
is hard to regulate it once from the deviates to the optimized state", and feedforward alone
isn't enough (p. 554). It also says reaching steady state takes weeks for La/Ce and years for
Pr/Nd circuits started from scratch, and warns of a "false steady state" (pp. 482, 492, 533).

## 5. What plants do: hold the profile at sensitive stages

  - The designer supplies the stage-by-stage profile. If routine assays show it has shifted
    because of organic, scrub or feed flow changes, the flows are adjusted to put it back,
    because "once the unqualified product comes out... it will take a very long time to adjust"
    (book p. 560).
  - Detecting points go where the composition gradient is steep (book p. 555), "between five
    and fifteen stages from outlet" (Yang, Xu, Lu and Ding, Chin. J. Chem. Eng. 23 (2015) 192,
    p. 193). The flows that are adjusted: organic (saponification rate), feed, scrub acid, strip
    acid (book p. 556, table 4.3). Online analysis by Am-241 EDXRF (book p. 553).
  - Yang et al. hold one composition in the extraction section and one in the scrub section with
    the extractant and scrub flows. Their evidence is a simulation without plant-model mismatch.

Our notebook 22 found the same structure at steady state, and across twenty plants: holding a
composition near the product end and one near the raffinate end with the two flows keeps 25 of
26 cases on spec without a correct model.

## 6. The acidity floor

The book doesn't give an acidity limit for the extraction section. It does say that emulsion
forms at the raffinate end when the aqueous runs short of metal and the pH rises, and that the
cure is to leave a little rare earth in the raffinate (p. 513). Our finding (the MPC
over-saponifies into a no-acid region where the chemistry model isn't valid) is consistent with
that; the pH 3 floor stays our own judgement call, from where the fitted data stop.

## What this means for the design (phase 2)

  - Design with a reflux margin, as Xu's method does (k ~ 0.7), instead of at the cheapest
    corner. That's the first and biggest lever.
  - Design on the low end of the separation factor (1.8, not 2.0), and size organic pumps and
    settlers for the flows that needs.
  - More stages let the same purity be met with the flow ratio further from its limits; they
    widen the window, at capital cost.
  - Instrument the two monitoring stages (one in each section, 5-15 stages from the outlets) and
    the raffinate pH; close the loop on them.

Phase 2 puts numbers on each of these against the twenty realistic plants.

## Sources

  - Hydrometallurgy of Rare Earths: Extraction and Separation (Elsevier, 2018), chapter 3 (theory
    of countercurrent extraction, pp. 391-532) and chapter 4 (process design, pp. 533-589).
  - Cheng F., Wu S., Liao C., Yan C., Adjacent stage impurity ratio in rare earth countercurrent
    extraction process, J. Rare Earths 31(2) (2013) 169-173, doi 10.1016/S1002-0721(12)60253-1
  - Cheng F. et al., Minimum amount of extracting solvent of AB/BC countercurrent extraction
    separation using organic feed, J. Rare Earths 32(5) (2014) 439-444,
    doi 10.1016/S1002-0721(14)60091-0
  - Yang H., Xu F., Lu R., Ding Y., Component content distribution profile control in rare earth
    countercurrent extraction process, Chin. J. Chem. Eng. 23 (2015) 192-198,
    doi 10.1016/j.cjche.2014.09.046
  - Skogestad S., Morari M., Doyle J., Robust control of ill-conditioned plants: high-purity
    distillation, IEEE Trans. Automatic Control 33(12) (1988) 1092-1105
  - Skogestad S., Dynamics and control of distillation columns: a tutorial introduction, Trans
    IChemE 75 Part A (1997) 539-562

## Phase 2 (2026-10-17): design, operating margin, and the sized plant

  - Margin is where you choose to operate, not only how you design. Retuned to the cheapest
    flows that meet the targets, every plant sits on its own edge again, whatever the design
    assumed (notebook 25). Operating at 99.8 % / 97 % instead of 99.5 % / 95 % triples the window
    against the spec (about 1 % to 3-4 % of the scrub flow; the worst plant 0.3 % to ~3 %) for
    about 15 % more reagent (notebook 26).
  - More stages pay for that margin: 14+38 stages operated with margin use about the same
    reagent as 8+20 at the edge. Bigger stages (195 L mixers) widen the window by about one more
    point.
  - Sizing the stages to their flows removes most of the difficulty. On the sized plant (130 L
    mixers, ~95 % stage efficiency) the MPC with the nominal model and the acidity floor brings all
    ten test plants inside spec, the seven that defeated it on the old plant included, at the edge
    and with margin (notebook 21, --config ndpr_plant_sized). On the old plant the stages ran at
    81-85 % of equilibrium, so the plant's unknown stage efficiencies were a large, uncertain
    number; at ~95 % a 15 % lower transfer rate costs about one point of efficiency.
  - Still fragile: the nominal-model MPC had 15-39 failed QP solves in 240 steps on those plants.
    It recovers, but that's a numerical weakness to look at.
  - As built, on the design's own flows, almost no plant is on spec in any design. Every
    perturbed plant shares one structural error (acid exponent 3.3 against the model's 3.0), so
    commissioning and retuning are always needed; a real plant is no different.

## Phases 3 and 4 (2026-10-17): a plant closer to the book, and identify-then-control

  - A plant built closer to the book (`plant_models.BookCircuit`, notebook 29): separation
    weaker toward the cascade ends and a little weaker in the scrub, an acid slope of 2.5, slower
    transfer in acid, slow drift. It can still make spec on every layout tried, but needs about
    25 % more organic and reagent than the controller's model predicts for the same design
    (8+20: 2,693 against 2,172 mol/h). Of the four mechanisms, the weaker separation toward the
    ends is the one that matters; the others barely move the best recovery on their own.
  - The controller with its ordinary model on twenty book plants (main case, notebook 14 with
    `--plant book`): better than open loop on the feed switch, stronger feed, feed acid and strong
    scrub acid on nearly every plant, on the flowmeter bias on most; small losses on the rest.
    The design optimizer often doesn't converge on the book plant, so its "can't make spec" list
    there isn't reliable; the controller ran those plants fine.
  - Identify, then control (notebook 31). On the first plant's seven hard plants, where the
    nominal-model MPC failed on all seven, six noisy instrument readings at the plant's resting
    state (XRF on product and raffinate, UV-Vis Pr and Nd at three extraction stages) were fitted
    to the physics model: two separation factors and a stage-rate factor, about 30 steady-state
    solves each. The fitted Pr/(La+Ce) factor came within about 0.05 of the truth on every hard
    plant, and the MPC on the fitted model brought all seven to spec or within a hair of it (seed 6
    rests at 94.9 % recovery). This is the case for measurement plus physics-based estimation, in
    numbers: a few readings and a fit do most of what knowing the true plant does.
  - Two bugs found on the way, both in new code: the book plant's acid correction made D fall as
    the acid ran out (several steady states, a stalled optimizer, a wrong "can't make spec on 8+20"),
    and the steady-state solver could return an unconverged state silently (now it warns; none of
    the 37,128 stored sweep points was affected).
  - Which instruments the identification needs (notebook 31, --sensors). On the seven hard
    plants, the same fit and controller with three instrument sets:
      XRF on product and raffinate only     Pr/(La+Ce) off by ~0.07 (biased low); 7 of 7 on spec
      UV-Vis at stages 2, 4, 6, ratios only, with an unknown 15 % gain error
                                            off by ~0.01, the best; 6 of 7 (the miss rests at 94.7 %)
      both                                  off by ~0.03; 6 of 7 (the miss rests at 94.9 %)
      nothing fitted (notebook 21)          0 of 7
    Cheap relative spectroscopy identifies the separation factor best, because the ratios cancel
    its calibration error and the profile's shape carries the information; the XRF plants already
    have is enough for the controller. One noise draw per plant; the stage-rate factor sometimes
    sits on its bound, so it stands in for what the model doesn't capture rather than measuring a
    true efficiency.
  - Tracking a drifting plant (notebook 32). The first plant, with its separation factor sliding
    from about 1.98 toward 1.8 over two weeks (an extractant degrading between make-ups). With the
    nominal model never updated, the MPC cut the organic as the plant got harder and lost it:
    purity down to 98.5 % and 98.2 %, recovery at the end 91 % and 77 %. Refitting the model once a
    day from the same instruments as notebook 31 (XRF on product and raffinate, UV-Vis at three
    stages, with noise) kept purity above 99.4 % and 99.2 % and recovery at or near 95 %; the
    daily fit followed the true separation factor within about 0.05, lagging on the last days of
    the faster slide, where recovery dipped to 94.4 %. A fit costs a couple of minutes of computing
    once a day.
  - Drift or fault (notebook 33, first plant seed 4, one run). The drifting plant with the daily
    refit, plus a fault on day 7: eight scrub mixers (stages 10 to 17) drop to 40 % of their
    transfer rate, as if fouled by crud. Each day the fit returns the Pr/(La+Ce) separation
    factor, the Nd/Pr factor and one stage-rate factor k for the whole cascade. The separation
    factor kept following the slow drift through the fault (1.80 fitted against 1.83 true on day
    8, 1.72 against 1.75 on day 14). k jumped on day 8 (0.84, 0.84, 1.33, then 1.65) and the rule
    "k more than 25 % away from the median of the last five days" flagged days 8 and 9, then went
    quiet as the median caught up. Three things to keep in mind. k moved up, not down: one
    cascade-wide factor can't describe eight slow stages, so it bends to fit the profile rather
    than measure an efficiency. From day 8 it sits on its upper bound (e^0.5 = 1.65), so the size
    of the jump is set by the bound. And the flag only works on the step; a fault that grew slowly
    would look like drift. With the refitted model the controller held recovery at 95.0 % but
    purity settled at 99.45 to 99.5 %, on or just under the 99.5 % spec, from day 8 on. Next step,
    if it's worth it: fit k per section (extraction, scrub, strip) so a fault also says where.
  - The feed-loss interlock on twenty plants (notebook 34, main case, the two-hour pump stop,
    80 h). The upset the controller handled worst: the seed study had it losing more metal than
    open loop on most plants (mean about 20 mol of extra loss against about 3 open loop, over
    that study's run length). With src/interlock.py (freeze the handles while the feed is off,
    re-sync the model and hand back two hours after it returns):
      MPC over level loops, no interlock     0/20 off spec; extra loss mean 41 mol; 8 failed solves
      with the interlock                     0/20 off spec; extra loss mean -8 mol (less than at
                                             the start); worst plant +31 mol; no failed solves
      interlock that also stops saponifying  1/20 off spec (seed 17, 2,034 mol); mean +37 mol
    Freezing and re-syncing is the fix; stopping the saponification as well is worse (why hasn't
    been looked into). Minimum purity during the
    stop dips to 99.0 % with the interlock (99.4 % without), for a short time; no product went
    off spec. Against the open-loop numbers from the seed study the interlock is better on 16 of
    20 plants and worse on 4 (seeds 2, 6, 10, 17, by 10 to 28 mol); the run lengths differ, so
    that comparison is rough.
    Note (Oct 18): the interlock's re-arm built the model's rest point from the design inputs
    (design feed, design strip flow) instead of the measured ones; found by notebook 36 and fixed
    (CircuitMPC.resync). A pump stop doesn't change the feed composition, so the effect here is
    the strip flow only; notebook 34 is being rerun on the fixed code.

## Phase 5 (2026-10-18): from a simulation toward a plant

  - Supervised model updates (notebook 37, src/model_registry.py). Notebook 33's run again (seed
    4, drift, eight scrub mixers fouled on day 7, the same instruments and noise draws), with each
    daily fit a proposal checked and reviewed before the controller uses it. Days 1 to 6 approved;
    from day 7 every fit rejected (the step check on day 7, then the stage-rate factor on its bound
    from day 8), a day before notebook 33's trend flag. The controller stayed on the day-6 model
    and ended at 99.42 % purity and 94.98 % recovery, against 99.45 % and 94.95 % with every fit put
    in use: the review kept a questionable model out and raised the alarm earlier, at no cost to
    the product over this fortnight. Its cost comes later: the plant keeps drifting while the model
    is held, so the fault has to be fixed or a model approved with eyes open.
    The first version also let the fit residual decide, with a limit chosen before looking (4 per
    reading). The residual ran 1 to 82 on normal days and 22 to 68 on fault days, so it separated
    nothing; that gate rejected every fit, and the drifting plant ended at 98.8 % purity and 88 %
    recovery, worse than with no review. Checks have to be calibrated on the plant's normal
    history. The residual is now shown to the reviewer and doesn't decide
    (runs/supervised_updates_*_first_gate.* keep the first run).
  - Deployment in steps (notebook 36, src/deployment.py). A guard between the controller and the
    plant: shadow (advice only), advisory (an operator applies it every 2 h), closed loop; moves
    limited to 2 % of the commissioned value per interval and 20 % in all; a watchdog that holds
    the handles if the controller fails and keeps it out until re-armed; a journal of every
    interval. On the first realistic plant and the feed switch (20 h): shadow gave exactly the open
    loop's result (18.0 mol lost) while logging what it would have done; closed loop inside the
    limits did as well as without them (-5.7 against -4.1 mol), with the organic flow sitting at
    the 20 % ceiling. The handover (shadow to 24 h, advisory to 48 h, closed after) ran without an
    alarm. The 20-plant comparison is queued.
    The watchdog test (the controller made to crash from 20 h to 23 h, re-armed at 30 h) found two
    real faults: the re-sync on re-arm used the design feed and strip flow, so after a feed switch
    the next QP had no solution; and a failed qpOASES solve left the solver unable to solve again
    ("previous QP is not solved"), turning one failure into a permanent one. Both fixed; the
    reference runs are unchanged to the last bit.
  - The plant over OPC UA (notebook 38, src/plant_server.py, src/historian.py). The plant as an
    OPC UA server with ISA-style tags (setpoints FIC-201, FIC-301, AIC-302, XIC-202; flows, scrub
    acid read-back, every analyser with its sample time, every interface), the settlers' level
    loops on the plant side as in a DCS, the guarded MPC as a client that sees only tags, and a
    historian subscribed to every tag. 80 h of the feed switch: 160 intervals, no difference at
    all from the same run in one process, in any handle or in the plant's final state; 182 s
    against 157 s; 7,678 samples of 51 tags in the historian. The figure is drawn from the
    historian only. Runs unencrypted (security mode None); a plant would use signed and encrypted
    sessions with certificates.
  - Where the instruments should go (notebook 39). On the model at the design point: the profile's
    leverage on the separation factors is at the La/Ce-NdPr front in the middle of the scrub
    (stages 13 to 20 of this 8+20+4 layout), up to about 60 % change in Pr per 1 % in the
    separation factor; notebook 31's optical points on extraction stages 2, 4, 6 sit where the
    profile barely moves. Pr and Nd there are tens to hundreds of mmol/L, readable by absorption.
    Lower bounds on the standard errors (Fisher information, XRF at the ends at 2 %, optical Pr and
    Nd at 1 % with an unknown common gain):
      XRF on the ends only                 the five unknowns can't be separated
      optical 2, 4, 6 (notebook 31)        separation factors ~1 %; scrub rate factor 0.55 (unidentifiable)
      optical 4 + scrub 10, 16             separation factors 0.2-0.4 %; scrub rate factor ~3 %
      optical 2, 6 + scrub 12 + strip 30   strip rate factor under 1 %
    Aqueous sampling is slightly better than organic in every layout. The three section rate
    factors leave nearly the same shape on the profile, so telling them apart rests on small
    differences: true in the linear model, to be tested with model mismatch (a per-section fit on
    notebook 33's fault). These are linear bounds at one point on one plant, a guide to where to
    look.
  - A neural net that reads the separation factors from the instruments (notebook 35, first plant,
    30 L mixers). Trained only on simulated plants: 1,200 drawn like the test plants but with
    other seeds, each at its design flows and up to seven random handle settings (9,116 rows),
    with fresh instrument noise every batch. A two-layer PyTorch net (64 units each) and a ridge
    regression as the baseline. On 171 held-out plants: rms error in log factor 0.006 and 0.005
    (net) against 0.023 and 0.014 (ridge). On notebook 31's ten plants, with exactly the noisy
    readings notebook 31's least-squares fit saw, mean absolute error in Pr/(La+Ce):
      hard plants   net 0.008   ridge 0.028   least squares 0.028
      easy plants   net 0.018   ridge 0.022   least squares 0.305
    and in Nd/Pr 0.004 / 0.006 (net) against 0.035 / 0.126 (least squares). The MPC on the net's
    model brought all seven hard plants to spec (99.50-99.51 % purity, 95.0 % recovery at 120 h).
    One estimate takes about half a millisecond, against minutes for the fit. Starting the
    least-squares fit from the net's answer didn't help: it went back to the same answer, so the
    bias is in fitting three numbers to a plant that differs in more ways than three, not in the
    starting point. The net learned from plants carrying every one of those differences.
    The caveat that goes with it: the net was trained on plants drawn the same way as the test
    plants. A real plant isn't drawn from this generator; the net is only as good as the
    simulator's spread is wide, and outside it (a fault, a drift the training never saw) it has
    no physics to fall back on. The physics fit has no such dependence. Used together, the net
    for a fast first answer and the physics fit and the review (notebook 37) as the check.

## Phase 6 (2026-10-18): the identification line on the main case, and the reruns

The identification studies (notebooks 31 to 39) had run on the first plant (30 L mixers), where
the nominal model failed on seven plants. Rerun on the main case (130 L mixers, margin, acidity
floor, the MPC over level loops, `--config ndpr_main`), where every plant already starts on spec.

  - Identify, then control (notebook 31, plants 1 to 10). All ten on spec at the end with the
    fitted model: purity 99.73 to 99.9995 %, recovery 97.0 to 98.2 % (target 97 %). The fit's
    bias is the same as on the first plant: the plants with a high Pr/(La+Ce) factor (2.2 to 2.4)
    come out 0.23 to 0.37 low; the rest within 0.04. The stage-rate factor sits on a bound on
    seven of ten. On this plant identification isn't what keeps it on spec; margin does.
  - Drift for two weeks (notebook 32, plants 11 and 4, separation factor sliding to about 1.74).
    The fixed model now holds too: 99.63 % purity and 97.0 % recovery at the end on both. Refitting
    daily ends at 99.79 % and 99.76 % on 6 % less organic (2,007 against 2,142 L/h; 2,051 against
    2,180). With margin, refitting buys efficiency, not rescue.
  - The fouled scrub mixers (notebook 33, plant 4). Barely visible: the fitted stage-rate factor
    stays at 0.94 to 0.97 through the fault, and the trend flag doesn't fire on it (its one flag,
    day 14, is the fit reaching its bound). The controller held spec; recovery went from 98.6 %
    to 97.0 % at the fault and stayed there. With every stage near equilibrium (130 L mixers),
    a 60 % slower transfer in eight stages changes the profile little, in line with notebook 39.
  - Supervised updates (notebook 37, the same run). Days 1 to 7 approved; days 8 and 9 held back
    (the separation-factor fit dropped by 0.10 at the fault, more than the step limit); days 10
    to 13 approved; day 14 held back (step and bound). End 99.76 % / 96.96 %, the same as with
    every fit used.
  - Where the instruments go (notebook 39 on the main case). The separation factors are pinned
    to 0.05 to 1 % with any optical layout; the section rate factors are much harder to see than
    on the first plant: the scrub rate factor to about 0.19 to 0.43 even with two scrub points
    (0.55 to 25 without). Same reason as above: near equilibrium, how fast the metal moves
    barely shows, and matters less.
  - The interlock on 20 plants, rerun after the re-sync fix (notebook 34; the earlier table is in
    runs/history/). The interlock as built: mean loss -7.8 mol (was -7.7), worst +23.1, none off
    spec, no failed solves, minimum purity 99.44 % (was 99.04 %). The version that also stops
    saponifying: none off spec (was one plant, 2,034 mol), mean +21.8 mol (was +36.5). So part
    of what made it look bad was the re-sync bug; it's still worse than not stopping.
  - Deployment modes on 20 plants, 80 h (notebook 36):
      feed switch       off-spec plants   off-spec mol   mean metal lost
        open loop              0               0            1,052
        MPC + interlock        5           4,707               -8
        guarded, closed        7          16,628               -2
        advisory, 2 h          2           1,296              -35
      stronger feed: every mode on spec; mean metal lost 141 open loop, 78 unguarded MPC,
        66 guarded, 64 advisory.
    The guard's 20 % ceiling is too tight for a switch to a feed twice as rich, which takes about
    38 % more organic: the guarded controller sat on it (140 clipped intervals out of 160 on
    average) and seven plants went off spec. Limits have to be sized from what the plant's real
    upsets need, the same lesson as the review checks. Advisory mode, smaller moves every 2 h,
    did best of the three on the feed switch; measured, not explained. And on the feed switch
    the MPC trades some off-spec product on a few plants for keeping metal out of the raffinate,
    where open loop holds purity and loses about 1,050 mol a plant.
  - The interlock against doing nothing, like for like (notebook 34 with open loop added: the same
    20 plants, the same pump stop, 80 h). Open loop over the level loops: mean -10.4 mol, none off
    spec, minimum purity 99.34 %. Against it, per plant: the MPC without the interlock is worse on
    19 of 20 (by 43.7 mol on average, up to 216.7); with the interlock it's about even, better on 9
    and worse on 11, by +2.6 mol on average (-20.5 to +40.1); the version that also stops
    saponifying is worse on 19 (+32.2). For a pump stop the best a controller can do is stay out of
    the way, and the interlock gets it there. The earlier "better on 16 of 20" compared against the
    seed study's open loop, a different run length on older code, and doesn't stand.
  - The learned estimator on the main case (notebook 35 --config ndpr_main). 600 training plants
    (3,069 rows: fewer handle settings converge per plant on this slower plant); on 85 held-out
    plants rms error in log factor 0.0043 and 0.0040 (net) against 0.018 and 0.013 (ridge). On
    plants 1 to 10, the same readings as notebook 31's fit, mean absolute error:
      Pr/(La+Ce)   net 0.038   ridge 0.040   least squares 0.148
      Nd/Pr        net 0.014   ridge 0.014   least squares 0.098
    About four and seven times more accurate than the physics fit, in half a millisecond to under
    two. Two differences from the first plant: the plain linear regression is nearly as good as
    the net here (this plant is more linear), and the net's two misses are the two plants with the
    highest Pr/(La+Ce) factor (2.33 and 2.41, read as 2.15 and 2.23), the edge of what it was
    trained on. The MPC on the net's model: seven plants on spec, settling at 99.66-99.87 % purity
    and 97.0-97.9 % recovery. Plants 1, 7 and 8 show recovery above 100 % at the end of 120 h
    (product outrunning feed while the large settler inventory drains can do that for a while)
    and also in the steady-state check of their final handles, which can't be right at rest; the
    run logged 16 unconverged steady states. Those three are unconfirmed, not on spec.
  - The rich feed as the base (configs/ndpr_main_yangibana.toml). A design on the Yangibana feed
    (about 41 % Pr and Nd) for the same 8+20+4 sized stages: organic 1,621 L/h (1,634 on Eneabba),
    scrub 142 L/h (218), reagent 1,862 mol/h (2,172), run the main case's way (margin, acidity
    floor, 120 h trim). All 20 plants can meet both targets. 20 plants, 80 h (notebook 14):
      switch to the lean feed   open loop: 14 of 20 off spec, 28,378 mol in all; metal lost
                                -86 mol on average (less than at the start)
                                controller: none off spec; +52 mol on average (worst 270)
      stronger feed             open loop +483 mol on average; controller +17
      no upset                  both on spec; open loop +1.9, controller -9.5
      no failed solves.
    The switch has two directions with opposite trades. Lean to rich (the Eneabba base): left
    alone the plant loses NdPr to the raffinate, and the controller keeps it at the cost of some
    off-spec product on a few plants. Rich to lean: the organic is suddenly oversized for the feed
    and pulls La and Ce into the product, so most plants go off spec; the controller cuts the
    organic and keeps every plant on spec for a small loss of metal.
  - The three unconfirmed plants of the learned-estimator test, run 400 h instead of 120 (notebook
    35 --hours 400 --seeds 1,7,8). Plant 1 is on spec: 99.77 % / 97.0 % at the end, settling at
    99.82 % / 97.01 %. Plants 7 and 8 still haven't settled: recovery at 400 h is still above 100 %
    (103.9 % and 102.7 %), the plant still drawing down its inventory, and the steady-state check
    of their final handles points below spec (94.7 % and 95.6 % purity) but logged unconverged
    solves, so it isn't reliable either. They are the two plants with the highest separation
    factors, the ones the net read 0.17-0.18 low. Open: whether the controller on the net's model
    takes these two off spec in the end. Settling it needs a much longer run or a direct solve of
    where the closed loop comes to rest.
  - The guard with its ceiling sized from the upset (notebook 36 sized: the unguarded controller
    took the organic up about 38 % on the feed switch, so 50 % instead of 20 %; 20 plants, 80 h):
      feed switch     guarded: 6 plants off spec, 5,104 mol (was 7 and 16,628 with 20 %; the
                      unguarded MPC 5 and 4,707); advisory: 3 plants, 1,811 mol (was 2, 1,296)
      stronger feed   all on spec in both modes, as before
    Sizing the ceiling from the upset took away nearly all of the guard's cost. The clipping left
    (75 of 160 intervals on average on the feed switch) is the 2 % per-interval move limit; the
    off-spec product left is the controller's own trade on this upset, which the unguarded
    controller makes too.

## Phase 7 (2026-10-21): questions from a review session

  - Why plants dip on the feed switch although the controller is told the feed changed (notebook
    41; the five plants of notebook 40 --full-state that went below spec, seeds 3, 10, 14, 16, 17;
    80 h each). Four ways: as run (the minima match notebook 40 exactly), the controller on the
    plant's true model, the move limits three times wider, the purity weights ten times heavier.
                       plants below spec   hours below spec   minimum purity
      as run                 5 of 5           6.5 to 13.0       98.68-98.98 %
      true model             0 of 5               0             99.22-99.47 %
      move limits x3         3 of 5             0 to 8.0        98.76-99.20 %
      purity weight x10      1 of 5        0 (9.0 on seed 10)   98.85-99.26 %
    The cause is the model. All five plants separate worse than the model believes (Pr/(La+Ce) of
    1.74 to 1.94 against 2.0). Seed 10, step by step: at the switch (5 h) the controller raises the
    organic at once (2,390 to 2,689 L/h by 8 h), lets the scrub fall (659 to 599 L/h) and cuts the
    scrub acid (0.89 to 0.59 mol/L). The scrub-to-organic ratio drops from 0.276 to 0.223. With the
    model's separation factor that ratio is enough; on this plant it isn't. Purity starts to sag
    about three hours after the switch as the La and Ce work through the scrub, and is below spec
    from 13 h to 25.5 h (minimum 98.68 % at 18 h). The output bias then turns it round: scrub back up, at its
    10 L/h-per-step limit on 29 of 160 steps, to 860 L/h, ratio 0.282. The controller on the true
    model never lets the ratio fall below 0.283 and ends at 0.318 (organic 3,078, scrub 980 L/h).
    So feedforward gets the direction right and the size of the organic move about right; the
    model's separation factor sets the ratio, which is the direction the plant is most sensitive
    to (condition number 43). Other handles: the scrub acid was cut hard on every plant (to 0.16 to
    0.34 mol/L); saponification moved little (0.443 to 0.418 on seed 10; the most, seed 17, 0.435
    to 0.376). Wider move limits shorten the dip but overshoot: recovery above 100 % at 80 h on
    three plants (the settlers' inventory drawn down by the bigger organic moves). The heavier
    purity weight removes the dip on four of five, for recovery ending at 97.2-97.5 % instead of
    97.0 %. Not every weak plant dips (seeds 2, 6 and 9, at 1.84, 1.87 and 1.83, don't). One caveat
    on the comparison: the true-model controller starts moving before the switch, because the
    trimmed starting point was found with the nominal model and isn't its own optimum.
  - Is the net better than a least-squares fit, or was the fit handicapped? (notebook 42, the
    first plant, notebook 31's ten plants and noise draws, the fit started from the net's answer.)
    Mean absolute error in Pr/(La+Ce), net 0.011:
      as notebook 35 (three unknowns, 60 evaluations, step 0.02)   0.107
      stage-rate factor held at 1 (two unknowns)                   0.107
      and 300 evaluations                                          0.107
      and step 0.005                                               0.108
      and the steady-state solve to 1e-11                          0.108
      three unknowns with that search                              0.109
    None of the suspects is it. On all ten plants the fit stops at a lower cost than the true
    separation factors give: the search does its job, and the model, fitted to these readings,
    prefers a wrong answer. Even at its best the fit leaves the readings 4 to 29 noise widths off
    on average (pure noise would give about 1): the plant differs from the model in ways two or
    three numbers can't absorb (acid exponent, stage efficiencies per stage, the other drawn
    numbers). The error is on the easy plants (0.29 on seeds 1, 7, 12; 0.03 on the seven hard
    ones). The net learned the mapping from plants carrying the same kinds of differences; that is
    its advantage here and its limit on a plant that differs in a way the generator doesn't draw.
  - The profile at the design point (notebook 01 now also writes runs/steady_profile_<config>.csv).
    Main case, model at the design handles: the NdPr share of the aqueous rare earths is 1.5 % in
    the raffinate stage, 16.5 % at the feed stage (7), flat at 16-17 % through scrub stages 8 to 13,
    then the front: 20 % at stage 16, 47 % at 20, 72 % at 22, 90 % at 24, 97 % at 26. About nine
    stages from 20 % to 94 %, the steepest steps 11 to 13 points per stage (stages 19 to 22). Pr peaks
    at 43.6 % at stage 24 (it is the lighter of the pair, so it gathers at the front); Nd climbs to
    78 % in the strip. The solve stopped at 250 h not fully settled (largest rate of change 7.9e-4
    per hour) at 99.47 % purity against the design's 99.50 %.

## Phase 8 (2026-10-22): a fair test of the physics fit, Tests 0 to 2 (notebook 43)

The plan (from the review session): the comparison of notebooks 35 and 42 may handicap the
least-squares fit, which adjusts three numbers (two separation factors, one shared stage rate)
while the perturbed plant also differs in acid exponent, stage efficiency per stage, densities,
salt volume and mixer hold-up. Main case (ndpr_main), the twelve readings of notebook 35 (XRF on
product and raffinate, UV-Vis Pr and Nd at stages 2, 4, 6), the same noise draws (seed + 5000),
plants 1 to 10. Everything below is new unless it says otherwise.

  - Test 0, the optical stages. Every notebook counts stages from 0 at the raffinate end
    (ARCHITECTURE.md, the circuit section; circuit.unpack). Stages 2, 4 and 6 are extraction
    stages; the study guide's 16 to 21 are scrub stages in the same frame. No frame mix-up. The
    extraction points do see a gradient (NdPr 3.7, 7.3 and 12.9 % of the aqueous rare earths), but
    the profile there barely answers the Pr/(La+Ce) factor: the ratio Pr2/Pr6 moves 2.6 % per 1 %
    in the factor, Pr at scrub stage 18 about 95 %, linearized (runs/sensor_sensitivity_ndpr_main.csv).
  - A solver finding on the way. The main case's slowest mode outlives the default steady-state
    ladder (250 h): at the model's own design point it stops at |dx/dt| 7.9e-4, and on a 0.5 %
    step from a settled state Newton fails and the ladder returns a state still moving (2.3e-4).
    800 h before the polish settles it (1e-11, unchanged by 3,000 h more). This changes notebook
    24's gain matrix (dated note in section 4: condition number 43 -> 204). It doesn't change the
    stage profile of phase 7 (identical to four decimals). Notebook 43 uses the 800 h solve
    throughout; notebooks 31 and 35 still use the default.
  - Test 1, sanity (plant = the model with only the fit's three numbers changed: each test plant's
    separation factors and its mean stage efficiency). The fit as notebook 31 runs it (from zero,
    60 evaluations, step 0.02) recovers them within noise on 7 of 10 plants (|error| / Fisher bound
    at most 1.6). On plants 1, 5 and 8, three of the four highest Pr/(La+Ce) factors (2.25-2.41), it stops
    0.29 to 0.44 low at a cost of 690 to 1,530 against 5 to 9 at the truth. Not a solver or tuning
    problem (converged solves, x_scale, tighter tolerances, dogbox, a smaller step and the rate
    factor held fixed all stop at the same point; started from the truth it stays there; the
    cost falls steadily along the straight line from the start to the truth): a second minimum
    near the nominal. Four starts on the Pr/(La+Ce) factor (0, +0.15, +0.3, -0.15), the lowest
    cost kept, recover all ten within noise (|z| at most 1.6). The shared rate factor itself is
    barely identifiable (Fisher bound 42-58 % in log) and ends on its upper bound on 6 of 10. On
    these plants the net is 0.22 to 0.28 low on the four highest factors (seed 1: 2.154 against
    2.374): every plant it trained on carries the acid-exponent offset, these don't.
    Closed loop, 120 h, MPC on each estimate: no difference between true model, fit, net and ridge
    in end purity (99.80-100 %); five plants start below spec at the design handles (minimum 95.1
    to 97.1 % for 12-17 h, identical for every estimator).
  - Test 2, which numbers to fit. Ten candidates; identifiability at the model's design point
    (Fisher information of the twelve noise-scaled readings, lower bound on the standard error
    with the candidate fitted alone, in units of its plausible spread), control importance as
    the change in the steady-state gains (|dG|/|G| over the four handles, at plus and minus one
    spread) and in closed loop (one candidate off by one spread, the controller on the nominal
    model against the controller on the right one, 120 h and at rest):
      candidate          alone   trades with        |dG|/|G|    closed loop, wrong vs right   decision
      Pr/(La+Ce)         0.005   -                  0.84-0.96   harder side: rests 97.9 % vs 99.5 %   keep
      Nd/Pr              0.006   -                  0.30-0.87   no harm                     keep
      acid exponent      0.027   common K (-1.000)  0.56-0.83   no harm                     keep
      common K factor    0.32    acid exponent      0.15-0.17   no harm                     fix
      shared/extr. rate  0.88    mixer hold-up (-1.000); Nd/Pr (-0.96 in notebook 31's set)
                                                    0.01-0.07   none                        fix
      mixer hold-up      1.37    extraction rate    0.01-0.07   none                        fix
      scrub, strip rate  none (no reading moves 1 % of its noise width)   0.00-0.01   none  fix
      organic density    none                       0           none                        fix
      salt molal volume  none                       0           none                        fix
    With two separation factors and the acid exponent the Fisher matrix has condition number 147
    and the worst correlation is -0.85 (Pr/(La+Ce) with the acid exponent); all three bounds are
    0.01-0.03 spreads. Notebook 31's set (shared rate instead of acid exponent): condition number
    5.4e5, the rate bound 7.5 spreads.
    The check, both sets fitted to the ten perturbed plants with the same four starts and the
    800 h solve, then the MPC on each fit for 120 h (runs/fair_fit_check_ndpr_main.csv):
                                    Pr/(La+Ce) error     Nd/Pr error      cost      fit time
                                    median   worst       median  worst    median
      factors + acid exponent       0.002    0.013       0.002   0.011    4         25-32 s
      factors + shared rate         0.058    0.368       0.057   0.237    474       29-179 s
      (net, notebook 35 main)       0.004    0.179       0.004   0.062    -         0.5-2 ms
      (ridge, notebook 35 main)     0.017    0.176       0.015   0.033    -
    The fitted acid offset is +0.26 to +0.34 against the plant's +0.3. On the perturbed plant the
    physics fit with the right third number is as good as the net on typical plants and much better
    on the worst one. Closed loop doesn't separate the sets: end purity 99.80-100 % and recovery
    97.0-99.9 % with both; at rest, the acid-exponent fit's controller leaves plants 7 and 8 at
    98.4 and 98.6 % purity (recovery 98.3 and 98.2 %), the shared-rate fit's at 100 %. The same
    happens with the exact model on easier plants in the importance runs (rest purity 95.9 % with
    the Pr/(La+Ce) factor a spread high): on easy plants the controller is still moving at 120 h,
    so the 120 h decision metric doesn't settle there. Longer runs or a direct rest-point solve
    of the closed loop would.
    Caveat that matters for Tests 3 and 4: every perturbed plant has exactly the same acid offset
    (+0.3, src/mismatch.py), and the other differences barely touch the readings near equilibrium.
    The chosen set fits the generator's main knobs, so this test can't say how the fit does on a
    plant that differs in a way the model doesn't have. That's Test 4.
  - Test 3, the fit given a fair chance (Oct 22, notebook 43 check --map, --uv). The chosen set
    as a MAP fit (a prior on each number at its plausible spread), four starts, 800 h solves; then
    with three more optical points in the scrub (stages 12, 16, 20), with and without the common
    K factor as a fourth number (the one candidate the scrub points make separable in the linear
    analysis: bound 0.31 spreads, worst correlation 0.84):
                                         Pr/(La+Ce) error    Nd/Pr error     worst corr.  cost
                                         median   worst      median  worst   at the fit   median
      A  12 readings, three numbers      0.002    0.012      0.002   0.010   0.80-0.96    5
      B  + scrub 12, 16, 20              0.003    0.076      0.003   0.038   0.83-0.96    8
      C  B + common K factor             0.003    0.074      0.003   0.037   0.92-1.00    8
    The prior hardly acts (A is the plain fit's result to the third decimal); the numbers are
    well pinned. The scrub points don't improve the factors here, since the twelve readings already
    pin them, and with them plant 1's fit stops in a wrong place again (cost 363 against about 9;
    four starts weren't enough). The common K factor isn't recovered in practice: true -0.09 to
    +0.09, fitted +0.004 to +0.045 (plant 1 +0.375), correlations near 1. The plant's other
    differences absorb it; the linear bound at the nominal point was optimistic. Closed loop is the
    same in A, B and C on every plant (end, minimum and rest purity and recovery within 0.05 points;
    most identical). So the fitted set stays at three: the two separation factors and the acid exponent.
    No fourth number is both seen by the readings and worth anything to the controller here.
  - Test 4, plants with physics the model doesn't have (Oct 23, notebook 44). Each of the ten main-
    case plants rebuilt four ways, on its own perturbed numbers and stage efficiencies: with
    entrainment between stages; with a mixer and three settler tanks per stage; drifting chemistry,
    read after 72 h on the design handles with the drift running (not at rest); and the book plant
    (Pr/(La+Ce) 2.1 mid-cascade, about 20 % weaker toward the ends, 8 % weaker in the scrub, acid
    slope 2.5, slower transfer in acid). Six estimators on the same twelve noisy readings: the fit
    as published (both factors and a shared stage rate, one start), the new fit (both factors and
    the acid exponent, MAP, four starts), and the ridge and net of notebook 35 trained on perturbed
    plants only, and retrained with 150 book and 150 entrainment plants added (899 plants; held-out
    rms error in log Pr/(La+Ce) 0.014 net, 0.049 ridge). Error in Pr/(La+Ce), median / worst:
                       old fit        new fit        ridge          net            ridge mixed    net mixed
      entrainment      0.058/0.369    0.002/0.011    0.018/0.173    0.004/0.178    0.043/0.188    0.009/0.158
      settler          0.058/0.356    0.002/0.011    0.017/0.080    0.004/0.031    0.039/0.132    0.009/0.056
      drifting         0.077/0.341    0.009/0.035    0.031/0.193    0.014/0.174    0.050/0.209    0.021/0.146
      book (vs 2.1)    0.239/0.261    0.240/0.265    0.295/0.338    0.271/0.291    0.170/0.225    0.019/0.055
    On entrainment, settler and the drifting plant read in motion, the new fit is the most accurate
    estimator, and it still finds the acid offset (0.26 to 0.37 against 0.3). Training the net on
    these plants doesn't close the gap. The book plant has no single true factor: every estimator
    except the net trained on book plants reads about 0.24 to 0.30 below the mid-cascade 2.1, an
    effective value for a cascade whose ends separate worse. The mixed net learned to answer 2.1,
    the label it was given, and its controller did worst: one book plant stayed below spec for the
    whole 120 h (93.4 % at the end); the perturbed-only ridge left no book plant below spec at rest.
    A number that matches the label isn't necessarily the number that controls well when the
    plant's physics differ from the model's; the closed loop is the judge.
    Closed loop, 120 h: end purity 99.6 to 99.9 % for every estimator on every plant type, except
    that one book plant. Every run starts below spec on some plants at the design handles (minimum
    mostly 92.6 to 93.0 % whatever the estimator; lower with the net on a settler plant, 90.7 %,
    and with the ridge on a drifting one, 92.0 %). At rest on the final handles, the new fit's
    controller leaves fewer plants below spec on the drifting and entrainment plants (1 and 2 of 10,
    against 6 to 7 for the ridge and nets); with the controller still moving at 120 h on easy
    plants, that metric is indicative only.
  - What the output bias should do when a refitted model is swapped in (Oct 23, notebook 46;
    Emilio's question). Notebook 32's drifting main-case plants (seeds 11 and 4, Pr/(La+Ce) sliding
    about 10 % in two weeks), refitted daily with the corrected set (MAP, the twelve readings), the
    controller rebuilt on the fitted model at the current handles with its state on the new model's
    rest point (decision 21: the model state stays on the model's own trajectory). The new
    controller's bias three ways: reset to zero (as notebooks 32 and 37 do), kept from the old
    controller, or matched so the new model plus bias equals what the plant reports at the swap.
                         purity end   min    hours below spec   recovery end
      never refitted     99.63        99.63  0                  96.97 / 97.02
      reset              99.76        99.75  0                  96.97 / 97.00   (seed 4 / 11)
      keep               99.76        99.75  0                  96.97 / 97.00
      match              99.76        99.75  0                  96.97 / 97.00
    The three ways differ by less than 0.006 points of purity at any moment and by up to about 1
    point of recovery for a while after a swap; the daily fit followed the true factor within 0.04.
    The bias itself stays small on this plant (purity within 0.1 %, recovery up to 2 %), so there's
    little for the choice to act on. The matched bias wasn't smaller after the swap than before
    (0.0082 against 0.0069 and 0.0094 against 0.0081, purity plus recovery), so "a good refit
    leaves a smaller bias" isn't a usable review check here: what's left in the bias is the plant
    being in motion, not model error. Refitting itself buys margin (99.76 % against 99.63 % purity
    at the end, as in phase 6). Matching the bias at the swap is the principled choice and costs
    nothing, but on this plant it doesn't change the result. It would matter where the model error
    is large at the swap, as on the first (30 L) plant; not tested there.
  - Why three numbers and not four, in plain terms (Oct 23). The only candidate for a fourth was the
    common K factor (all extraction constants moving together). With the twelve readings the fit
    can't tell it from the acid exponent: they change the readings the same way (correlation
    -1.000), so fitting both leaves a flat valley where the answer can sit anywhere. With three
    scrub optical points the linear analysis said they'd separate, but on the plants they didn't:
    the fitted common factor came out +0.004 to +0.045 whatever the true value (-0.09 to +0.09),
    correlations 0.92 to 1.00. The controller doesn't need it either (closed loop the same with
    three or four; one spread of error in it did no harm in Test 2). And it costs time: median fit
    46 s against 30 s (worst 358 s against 313 s). Each number the readings can't see adds a flat
    direction to the fit: slower and less reliable for no gain. (The search did get trapped once
    with the scrub points, plant 1, but with three numbers as well as four, so that isn't the
    reason.)
    What the third number is: the acid exponent n in D = K L^3 / h^n, how strongly extraction falls
    as the water gets more acidic. It's a property of the chemistry (extractant, diluent,
    saponification), not a handle: the scrub acid handle sets h, n says how much D answers to it.
    The model has 3 from the lab-data fit; the test plants have 3.3, and Qi's book gives about 2.4
    to 2.5 for saponified P507 (p. 246). Every prediction of what the scrub acid and the
    saponification will do goes through it, which is why it's worth calibrating on a real plant.

## Phase 9 (2026-10-23/24): does the estimate matter? (notebooks 47 to 50)

The worry (Emilio, in the review session): in Tests 1 and 2 every estimator gave about the same
closed-loop result at 120 h, the fit that was 0.29-0.44 off on Pr/(La+Ce) included. If that held,
feedback and margin would do the work and the model would only need to be roughly right. Checked
before anything is said about it.

  - Part A, the audit (notebook 47). The estimate does reach the controller: the factor in each
    of the 40 Test 1 controllers is the estimate's (plant 1: 2.374 true, 2.049 fit, 2.154 net, 2.560
    ridge). The trajectories differ: on plant 1 the badly-off fit holds the same purity on 13 %
    more organic and 21 % more scrub (1,846 against 1,635 L/h; 255 against 210), on plant 3 the
    handles differ by up to 157 L/h of organic along the way. Every run already starts on the
    design's handles, optimized on the nominal model (commission() only settles the plant there and
    sets the jacklegs). Starting bias: 0 with the true model, up to 0.3 points with the accurate
    estimates, 1.0-1.4 points of purity with the badly-off fit. Why the results looked the same:
    the badly-off estimates all fell on the easy plants (1, 5, 7, 8: 100 % purity on the design
    handles, so purity can't show a model error), every estimator was accurate on the five hard
    plants, the output bias settles any model with the right gain signs on the targets, and only
    the 120 h end values were compared, with reagent not measured. The true model was at least as
    good on end purity everywhere; 0.5-1 h more below spec on plants 2 and 3.
  - Part C, dose-response (notebook 48 C; plants 1-10, 400 h, the plant's own model with only the
    Pr/(La+Ce) factor wrong; + means the model thinks the plant separates better than it does):
      error   end purity    hours below spec   shortfall      base       acid      plants never
              (worst)       (mean)             pt.h (mean)    (vs true)  (vs true) on spec
      -0.4    99.92 %       10.4               21.8           +31 %      +40 %     0
      -0.2    99.90 %       10.4               21.4           +23 %      +20 %     0
      -0.1    99.88 %       10.3               21.2           +11 %      +10 %     0
       0      99.80 %       10.2               20.5           -          -         0
      +0.1    99.65 %       12.6               31.8           -1 %       -2 %      0
      +0.2    99.45 %       12.8               26.7           +3 %       -3 %      0
      +0.4    82.88 %       255.3              1,701.7        -19 %      -23 %     7
    (The 10 h below spec at zero error is the start: six plants (2, 3, 4, 6, 9, 10) sit at 92.9-97.5 % on the design
    handles.) The effect is lopsided. A model that thinks the plant separates worse than it does
    keeps the product on spec and pays in reagent, about 10 % per 0.1. A model that thinks it
    separates better is slower to spec, leaves some plants resting below spec on their final
    handles (down to 97.6 % at +0.1), and at +0.4 loses 7 of 10.
  - Part D, the crutches taken away (notebook 48 D_bias, D_switch, D_edge; Part D1 is already how
    every run starts). The output bias off: zero error unchanged; -0.2 holds purity but recovery
    ends at 85.8 %; +0.2 already loses 7 of 10 plants. A slow bias (0.1) behaves like the default.
    So the output bias is what makes moderate model error tolerable. The feed switch (80 h): an
    optimistic model adds time below spec (18.9 and 24.8 h at +0.1 and +0.2 against 14.4 h, worst
    end purity 99.31 and 99.06 %), a pessimistic one doesn't, which agrees with notebook 41 (the
    five plants that dipped all separate worse than the model believed). The edge design (99.5 /
    95 %, no margin): the same pattern; -0.4 costs +36 % base and +22 % acid, +0.4 loses 7 of 10.
  - In one sentence: model accuracy does change what the plant does. Through the output bias the
    controller absorbs an error of 0.1-0.2 in log Pr/(La+Ce), but a pessimistic error costs reagent
    (about 10 % per 0.1) and an optimistic one costs time below spec and can leave a plant drifting
    off it; at 0.4 optimistic, or 0.2 without the bias, the plant is lost.
  - Part E1, the feed switch's recovery (notebook 49; the five plants that dipped and three weak
    ones that didn't, nominal and true model, 240 h). Recovery as notebook 41 reads it (product
    over NdPr fed right now) drops to about 53 % for 4-5 h at the switch with any controller,
    the true model included: the feed's NdPr doubles within an hour and the product can't follow
    until that metal has worked through 32 stages. Metal actually lost is another matter: NdPr in
    the raffinate as a share of NdPr fed, from the switch, is below 5 % in every window from 24 h
    on, for both controllers (24 h: 1.0-4.5 %; 235 h: 0.8-2.5 %). The true model holds purity at
    99.22-99.47 % on the plants that dipped, above the 99 % spec but not 99.5 %, and buys it with
    a little more raffinate loss. Recovery keeps its definition: the plant runs near steady state
    almost all the time, and the switch is the exception.
  - Two controller weaknesses found on the way, not chased here. (1) Over-saponification: on
    plant 10 with the ridge's model (about 3 % optimistic), the controller raised saponification
    to 0.53 and cut scrub acid to 1.3 mol/L until the acid ran out in the extraction and scrub
    stages and the organic saturated; the plant simulation then stalled. The acidity floor is soft
    and covers the extraction stages only. (2) Easy plants drifting off spec: with the true model,
    plants 7 and 8 read 100 % purity and 102-103 % recovery after 400 h, handles settled, while at
    rest on those handles purity would be 95.6-96.3 % (the slow inventory mode); the controller
    doesn't act on it.
  - A simulator fix on the way: the plant integrator's last fallback (LSODA, no Jacobian, no step
    limit) could crawl for hours on a stalled state; two studies hung overnight on it. Each
    integration attempt now has an evaluation budget (simulate.MAX_RHS_CALLS); a stall fails fast
    and the run ends with a clear error. 342 tests pass, reference runs unchanged.
  - The easy-plant drift, mechanism confirmed (Oct 24, plant 8, true model, 150 h, logged every 10 h):
    the plant reads 100.00 % purity throughout and recovery climbs from 97.7 % to 100.2 %; the
    controller's steady-state estimate (y_ss, its linearized view of where purity settles) says
    100.00 % at every check; the plant at rest on the same handles falls from 100.00 % to 98.04 %
    (on to about 95.6 % by 400 h). The reagent term walks the organic and scrub down slowly (1,633 to
    1,617 L/h, 218 to 212) because purity looks saturated: near 100 % the purity barely answers a
    small move, so the linearized gain is about zero and the linear steady-state estimate can't
    see the drop coming. Recovery reading above 100 % is the visible symptom.
  - Reproducibility depends on the thread count (Oct 24). The commissioning trim (120 h of the
    MPC driving the plant onto its targets) lands on a different operating point with one math
    thread than with two, the project's setting in pixi.toml: plant 3, 2,385 against 2,134 L/h of
    organic, 668 against 622 L/h of scrub, both valid. Many handle settings meet the targets on this
    ill-conditioned plant, and rounding order alone sends the trim down a different path. Reference
    runs reproduce bit for bit only at a fixed thread count. Found because Part E2's first run
    (one thread per worker) re-trimmed the plants; it was rerun from the two-thread operating
    points notebook 41 used. Parts C and D start from a plain steady-state solve and aren't affected;
    notebook 49 reused notebook 41's trims. To do: pin the thread count in the code, not only in
    pixi.toml, or make the trim end on a defined point (the design optimizer's, say).
  - Part E2, the best any controller could do through the feed switch (notebook 50). One
    optimization over the whole switch on the plant's own equations and numbers, from notebook
    41's operating points (two-thread trims), 1 h steps over 60 h, the four handles within the
    MPC's bounds and rate limits (doubled for the 1 h step), the extraction stages' acid at or
    above 1e-3 mol/L, NdPr lost to the raffinate at most 5 % of the NdPr fed from the switch on,
    the lowest purity at any hour as high as possible; each answer replayed on the plant with the
    full integrator (what's quoted). Five plants that dipped (3, 10, 14, 16, 17) and three that
    didn't (2, 6, 9):
                                       lowest purity       NdPr lost
      switch known in advance          99.61-99.87 %       4.88-4.99 %
      known only when it happens       99.59-99.87 %       4.87-4.95 %
      no rate limits                   99.61-99.87 %       4.46-4.97 %
      6 h ramp instead of 1 h          99.60-99.87 %       4.72-4.96 %
    A trajectory that holds 99.5 % through the switch exists on every plant, in every variant, even
    with no warning; the loss cap binds (the optimizer spends the 5 %), the rate limits don't
    (0 or 1 handle-hours at a limit). So the dip below 99.5 % is a limit of the controller (its
    targets are a 99 % hard spec and a soft 99.8 %, and with the nominal model its ratio moves),
    not of the plant: the MPC on the true model reached 99.22-99.47 %, 0.1 to 0.5 points below
    what's possible. The best trajectories leave organic and scrub almost where they are and work
    the scrub acid and the saponification; the MPC on the nominal model raised the organic and let
    the scrub-to-organic ratio fall (notebook 41). Caveat: perfect knowledge of the plant and of
    the switch; a bound, not a controller. (A first run had re-trimmed the plants with one math
    thread, from operating points with more organic, plant 3 at 99.89 %; replaced by this one.)

## Phase 8, continued (written Oct 24): Tests 5 and 6

  - Test 5, the ridge against the physics fit, head to head (notebook 44 --kinds perturbed; the
    main case's perturbed plants, the same twelve readings and noise draws, closed loop 120 h).
    Plant 10 is missing: with the ridge's estimate the controller ran the acid out (phase 9) and
    the plant simulation stalled, so the plant's whole job failed; 9 of 10 plants. Error in
    Pr/(La+Ce), median / worst:
      fit as published 0.082 / 0.369     new fit 0.002 / 0.012     ridge 0.012 / 0.176
      net 0.004 / 0.179                  ridge mixed 0.044 / 0.190 net mixed 0.007 / 0.158
    The ridge isn't what made learning look good: it's three times the net's typical error and six
    times the new fit's, with the same worst miss as the net (the highest-factor plant, at the edge
    of training). The win the net had over the old fit came from the old fit's handicaps (Tests
    1 to 3), not from linear learning. Closed loop: end purity 99.64-99.80 % at worst for every
    estimator; at rest on the final handles, 0 plants below spec with the old fit, 2 with the new,
    5 with the ridge and the net (the controller still moving at 120 h on easy plants, phase 9).
  - Test 6, the smallest network that does the job (notebook 45; main case, notebook 35's data,
    features, training plants and schedule, widths 4 to 64, depths 1 to 3, three seeds each, CPU).
    Held-out rms error in log factor, Pr/(La+Ce) + Nd/Pr, mean over seeds:
                width 4   8      16     32      64
      depth 1   0.035     0.023  0.019  0.015   0.013
      depth 2   0.031     0.017  0.012  0.0087  0.0081 (notebook 35's size, 5,442 weights)
      depth 3   0.030     0.015  0.0095 0.0083  0.0076 (best, 9,602 weights)
    The smallest within 10 % of the best: width 32, depth 3, 2,754 weights, half of notebook 35's.
    Every size misses the same edge plant by about the same amount (0.07-0.08 in log on the ten
    test plants), and one estimate takes 0.02 to 0.05 ms on this PC. Closed loop on plants 1-10
    (120 h, each size's middle seed, and the ridge): no difference between sizes, end purity
    99.64-99.66 % at worst and recovery about 97.5 % for every one, down to 82 weights and the ridge
    (one ridge run, plant 10, failed as in Test 5). Accuracy separates the sizes; control on these
    plants doesn't. GPU timing not run: the installed PyTorch is a CPU build.
