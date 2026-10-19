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

How wrong the model's gains are, as the relative error sigma_max((G_plant - G_model) G_model^-1)
(above 1 means a controller that inverts the model can move the plant the wrong way in some
direction):
  - at the design handles: 0.6-0.8 on the hard plants, 2-20 on the easy ones
  - at each plant's own design point (least reagent at 99.5 % and 95 %): above 1 on 15 of 19
    plants, up to 130
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
