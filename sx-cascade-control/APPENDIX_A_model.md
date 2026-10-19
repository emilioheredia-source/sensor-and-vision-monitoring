# Appendix A — the model: assumptions, equations, parameters

This is the working behind the report. Notation is mine throughout; where an
equation follows a source, the source is named next to it. The parameter
table with every number and its origin is in `APPENDIX_A_parameters.md`,
rendered from the parameter file itself.

**Status (October 19).** The equations are current. The worked numbers (the
stage-efficiency table in A.5, the volumes, the checks at the `ndpr_plant`
point) are for the first plant, with 30 L mixers. The main case is now
`configs/ndpr_main.toml`, with 130 L mixers and 520 L settlers
(ARCHITECTURE.md decisions 28 and 34).

## A.1 Assumptions, stated

These are the assumptions of the controller's model, and of the matched and
perturbed plants built from it. The settler and entrainment plants (A.12)
drop assumptions 2 and 3 on purpose, to test the controller against physics
its model doesn't have.

1. **Isothermal.** No energy balance; temperature enters nowhere.
2. **Each phase in each stage is a stirred tank** with a fixed hold-up
   (mixer share plus settler share). The settler adds hold-up but no
   separate plug-flow delay.
3. **Flows are conserved per phase.** No entrainment, no crud, no phase
   loss. The aqueous leaving a stage equals the aqueous entering it.
4. **Four components**, lumped: La+Ce, Pr, Nd, Sm-and-heavier. Each lump
   has one extraction constant. Thorium and uranium are assumed removed
   upstream of solvent extraction.
5. **Chloride medium, acid-route chemistry**: the extractant is PC88A
   (P507), a phosphonic acid, and extraction releases protons (reaction
   A.3). Nitrate-route chemistry is not modelled.
6. **Finite-rate stages.** Transfer in a stage is proportional to the
   distance from equilibrium (A.5). The rate constant is set so that a stage
   reaches 95 % of equilibrium at the lab's three-minute mixer residence
   time, which is how the "three minutes to equilibrium" observation in Lyon
   (2016) is read. At the run files' flows the residence time is shorter and
   stages reach less (A.5).
7. **Saponification is an equivalent base dose** carried with the organic
   and spent where it meets acid (A.7); the sodium form of the extractant
   is not tracked.
8. **Densities are linear in loading** (A.9); the interface follows the
   hydrostatic balance with a first-order lag (A.8) and does not feed back
   into the hold-ups.
9. **The distribution law has a floor on acidity and a cap on D** (A.4).
   Below about pH 3 the law stops applying (hydrolysis, hydroxide
   precipitation); both regularizations are labelled chosen, and the
   operating point was shown not to move with the floor.
10. **Reflux scrub** takes the product stage's aqueous composition as the
    scrub liquor, at an acidity the operator sets; the product flow reported
    is net of the reflux.
11. **No third phase, no gelation, no solvent degradation**, no
    temperature or crud effects on the kinetics.

## A.2 Indices and states

- Components *i* ∈ {La+Ce, Pr, Nd, Sm+}, stages *j* = 0 … *N*−1 counted
  from the raffinate end.
- States per stage: aqueous concentrations *c*ᵃ*ᵢⱼ*, organic concentrations
  *c*ᵒ*ᵢⱼ* (mol/L), aqueous acid *h*ⱼ (mol/L), interface height *z*ⱼ (m).
- Inputs: organic flow *Q*ᵒ, feed flow *Q*ᶠ with composition *c*ᶠ and acid
  *h*ᶠ, scrub flow *Q*ˢ with acid *h*ˢ, strip flow *Q*ᵗ with acid *h*ᵗ,
  saponified fraction *s* of the extractant, jackleg heights *H*ⱼ, and a
  per-stage multiplier *κ*ⱼ on the transfer rate (1 nominal).
- Units: concentrations in mol/L, volumes *V* in L, flows *Q* in L/h, time in
  h, heights in m, densities in g/mL.
- The cut names the last component meant to leave in the raffinate. For the
  NdPr product it is La+Ce: Pr and Nd go to the product. Purity is the
  product-side share of the metal in the product stream, in moles; recovery
  is the product-side metal leaving in the product over the product-side
  metal fed.

## A.3 Chemistry

The extraction reaction, as written in Rollat, Bertuol and Lévêque (2024,
eq. 1) and in Lyon (2016, section 2.4), with the extractant as the dimer
H₂L₂:

    RE³⁺(aq) + 3 H₂L₂(org)  ⇌  RE(HL₂)₃(org) + 3 H⁺(aq)

Mass action on that reaction gives the distribution coefficient
*D*ᵢ = *c*ᵒᵢ / *c*ᵃᵢ as a function of free extractant and acidity. That
form is standard; Lyon (2016, eq. 4) writes it for this system and uses
it to reduce his batch data, and the constants here were fitted to his
Tables 4 and 5 by `notebooks/00_fit_equilibrium.py`.

## A.4 Distribution law, as implemented

    L_free,j = L_tot − 3 Σᵢ cᵒᵢⱼ                       free dimer, mol/L; L_tot = 0.5 M (1 M PC88A as monomer)
    D_raw,ij = Kᵢ · L_free,j³ / (hⱼⁿ + h_minⁿ)          n = 3; h_min = 1e-3 M (chosen)
    D_ij     = D_raw,ij / (1 + D_raw,ij / D_max)          D_max = 1e4 (chosen)

*K*ᵢ: Nd and Pr fitted (0.0321, 0.0212; separation factor 1.51 against
1.57–1.59 measured on mixed feeds in Lyon's Table 6). La+Ce set to
*K*_Pr / 2.0 from the industrial Pr/Ce separation factor range 1.8–2.2
quoted by Yin et al. (2018); Sm+ set to 10 × *K*_Nd (chosen). For the
NdPr case the La+Ce constant is the load-bearing chosen number.

Why the cube: three dimers per RE³⁺ is the stoichiometry of A.3, the
standard form for an acidic organophosphorus extractant (textbook).
Judgement call: *K* is conditional on the ionic strength of the Idaho
tests; the scrub here runs well above it (A.13, item 3).

## A.5 Stage balances (the "fast" model)

Transfer of component *i* into the organic in stage *j*, in mol/h:

    r_ij = κⱼ · k · V_mix · (1 − φⱼ) · (D_ij · cᵃᵢⱼ − cᵒᵢⱼ)

where *V*_mix · (1 − φⱼ) is the organic in the mixer (φⱼ below). A stirred
stage reaches the fraction *kτ*/(1+*kτ*) of equilibrium, with τ the mixer
residence time, *V*_mix over the total flow through the mixer. *k* is set
from the lab: 95 % at τ = 3 min gives *k* = (0.95/0.05)/0.05 h = 380 /h. The
plant mixer is the lab's scaled 250 times, which keeps three minutes at
600 L/h total; the run files' design flows are higher, so their stages reach
less:

| Run file | Extraction | Scrub | Strip |
| --- | --- | --- | --- |
| ndpr_plant | 81 % | 85 % | 83 % |
| ndpr_small | 85 % | 89 % | 87 % |
| prnd_plant | 70 % | 73 % | 72 % |

The exact equilibrium stage is the limit *k* → ∞ of the same equations;
`notebooks/06_*` measures the difference by raising *k* up to 1000 times.

Why the efficiency formula: the organic's residence time in the mixer is
*V*_mix(1−φ)/*Q*ᵒ = τ, so the rate law gives the stirred-tank result.
Judgement call: one *k* for all elements, in extraction and strip alike,
carried over from a 120 mL lab mixer (A.13, item 1).

Balances, with Σ_in over the streams entering the stage:

    V_aq,j · dcᵃᵢⱼ/dt = Σ_in Q·cᵢ − Q_aq,j · cᵃᵢⱼ − r_ij
    V_org,j · dcᵒᵢⱼ/dt = Σ_in Q·cᵢ − Qᵒ · cᵒᵢⱼ + r_ij

Hold-ups: *V*_aq,j = *V*_mix · φⱼ + *V*_set · *z*_set/*H* and
*V*_org,j = *V*_mix · (1−φⱼ) + *V*_set · (1 − *z*_set/*H*), with
φⱼ = *Q*_aq,j/(*Q*_aq,j + *Q*ᵒ) the aqueous share of the mixer, *H* the
settler height (0.5 m) and *z*_set the interface setpoint (half of *H*).
*V*_mix = 30 L and *V*_set = 120 L.

## A.6 Flows and topology

The circuit is a list of edges (phase, from, to). Organic enters stage 0
and leaves stage *N*−1, recycled to stage 0 with its composition. Two
aqueous paths: scrub liquor (last scrub stage) and feed (last extraction
stage) flow down to the raffinate at stage 0; strip acid (stage *N*−1)
flows down the strip section only and leaves as product. Because flows
are conserved, the aqueous outflow of a stage is the sum of the source
flows upstream of it on its path; this is a 0/1 matrix built once from
the edges. With reflux, net product = *Q*ᵗ − *Q*ˢ.

## A.7 Acid balance and saponification

    V_aq,j · dhⱼ/dt = Σ_in Q·h − Q_aq,j · hⱼ + 3 Σᵢ r_ij − doseⱼ

Three protons per metal ion transferred (A.3). The saponified extractant is
a base carried with the organic: *b*₀ = *s* · *L*_mono · *Q*ᵒ mol/h
(*L*_mono = 1 M, the extractant counted as monomer) enters
with the organic at stage 0, each stage spends doseⱼ = *b*ⱼ · *h*ⱼ/(*h*ⱼ+ε)
(ε = 1e-4 M makes it smooth), and *b*ⱼ₊₁ = *b*ⱼ − doseⱼ moves on with the
organic.

Why: a sodium-form extractant trading Na⁺ for RE³⁺ has the same net proton
balance as a base dose. Check: at the ndpr_plant point *s* = 0.4285, about
what an organic at the 85 % loading cap gives off (0.85 × 0.5 = 0.425).
Judgement call: 99 % of the base is spent in stage 0 there, and the
sodium it leaves (about 0.9 M in the raffinate) isn't tracked (A.13, item 4).

## A.8 Interface

A settler with a fixed organic weir at height *H*_w (0.45 m) and an aqueous
jackleg (the outlet pipe whose height the operator sets) at height *H*ⱼ holds
its interface, at rest, where the hydrostatic columns
balance — the continuous gravity decanter of the unit-operations
textbooks:

    z_rest,j = (ρ_aq,j · Hⱼ − ρ_org,j · H_w) / (ρ_aq,j − ρ_org,j)
    dzⱼ/dt   = (z_rest,j − zⱼ) / τ_z                  τ_z = 6 min (chosen)

The gain dz/d*H*ⱼ = ρ_aq/(ρ_aq − ρ_org) is about 4 to 7 at the ndpr_plant
point, which is the size a plant sees. Judgement call: the balance is taken
at zero flow, with no head over the weir that grows with organic flow
(A.13, item 7).

## A.9 Densities

    ρ_org,j = ρ₀ + Σᵢ Mᵢ · cᵒᵢⱼ / 1000                           metal mass added, volume unchanged
    ρ_aq,j  = ρ_w + Σᵢ (Mᵢ + 3 M_Cl − ρ_w · φ_V) · cᵃᵢⱼ / 1000      RECl₃ salt with an apparent molal volume φ_V

ρ₀ = 0.85 g/mL and φ_V = 15 mL/mol are chosen; no loaded-phase density data
were found. Only the difference and its drift with loading move the
interface.

## A.10 Checks the model passes

- **Kremser.** One component, constant *D* (acid exponent 0, floor and cap
  off), near-ideal stages: raffinate fraction equals (*E*−1)/(*E*^(N+1)−1)
  with *E* = *D*·(O/A), to 0.1 % (the textbook countercurrent result, e.g.
  Seader, Henley and Roper, *Separation Process Principles*).
- **Mass balance.** Metal in = raffinate + net product, per component, to
  1e-6; acid in − acid out = base actually spent, to 1e-6.
- **Interface.** The jackleg height that the formula says puts the interface
  at its setpoint does so in simulation to 1e-6 m.
- **CasADi equals numpy** on the same inputs.

## A.11 What is deliberately simple

The one-way interface coupling, the equivalent base dose, uniform stage
efficiency, lumped components and the chosen densities are all places where
a plant study would replace a choice with a measurement. Each is a setting
or a labelled number, so replacing it does not mean rebuilding.

## A.12 Plants with physics the model doesn't have

`src/plant_models.py` builds two plants from the same pieces, for the
robustness tests (ARCHITECTURE.md, decision 25). The controller's model stays
as above.

- **Settler.** Transfer happens in the mixer only (volume *V*_mix, split by
  φⱼ), and each phase then flows through the settler as three tanks in a row,
  each a third of that phase's settler hold-up, with no transfer. What leaves
  the stage is the last tank's contents. Total hold-up and steady state are
  the model's; only the timing differs.
- **Entrainment.** Each phase leaves its settler carrying droplets of the
  other: a volume fraction ε_ao(zⱼ) of aqueous in the organic overflow and
  ε_oa(zⱼ) of organic in the aqueous underflow. The droplets join their own
  phase in the next stage and come back with its flow, so between stages
  *j* and *k* (organic *j* → *k*, aqueous *k* → *j*) there is an exchange
  flow q = ε · Q that carries metal back the wrong way:

      ε_ao(z) = ε₀ + ε_max / (1 + exp(−(z − z_hi − δ)/w))
      ε_oa(z) = ε₀ + ε_max / (1 + exp(−(z_lo − δ − z)/w))

  with ε₀ = 5e-4, ε_max = 0.02, δ = 2.5 cm past the window edges z_lo, z_hi,
  and w = 1 cm (all chosen).

## A.13 What an SX engineer would question

What holds up first. The flowsheet is the standard fractional-extraction
layout (extraction, reflux scrub, strip, organic recycle). The reaction,
the three protons per RE³⁺ and the cube on free dimer are the textbook form
for PC88A. Cerium is Ce(III) in a chloride liquor, so using a trivalent
constant for it is right. Taking Sm and heavier out ahead of the Ce|Pr cut
is the usual order. The saponified fraction matches what an organic at the
loading cap gives off (A.7). The phase densities and the interface gain
come out at sizes a plant would recognise (A.8, A.9). A Fenske estimate
for this cut gives about 14 equilibrium stages at total reflux, and the
design has 28 extraction and scrub stages at 81-85 %, worth about 23
equilibrium stages: about 1.7 times the minimum, a normal margin. At the
ndpr_plant point the extraction runs near pH 1.5-1.75, the scrub near
pH 1.3-1.45, and the last scrub stage near pH 0.5.

What is open, ranked by how much it could move the results. A.1, A.11 and
the "Open" list in ARCHITECTURE.md already state several of these; the
items below add what a plant engineer would say about them.

1. **Stage size against the design flows.** A.5 and REPORT section 10 cover
   the mixer: 0.7 min against the lab's 3, so 81-85 % stages. The settler
   has the same problem: about 10 m³/h per m² in extraction, where settlers
   are usually designed for a few (copper SX practice is often quoted at
   3-5; not checked against a project source). A real settler that loaded
   would entrain heavily. Sized for the flows, both would be two to four
   times bigger: better stages (fewer of them, REPORT section 3) and a
   slower plant (every time constant in the control results). The first
   half is now done as a side study (REPORT section 5.4, ARCHITECTURE.md
   decision 28): 130 L mixers and 520 L settlers, sized for three minutes at
   the design flows, give 95-96 % stages, about 19 stages for 95 % recovery
   instead of 24, and a plant about 3.5 times slower. The main case now
   uses them (decision 34). *Settles it:* a pilot stage at design flow.
2. **The Pr/Ce separation factor** (A.4). Lower values than 1.8-2.2 also
   appear in the literature (not checked against a project source). The
   Fenske estimate goes from about 14 stages at 2.0 to about 20 at 1.6.
   Batch values also differ from what a loaded, concentrated scrub sees.
   *Settles it:* batch tests on the plant's own feed at scrub-section
   strength.
3. **Constants used outside the conditions they were fitted at.** *K* was
   fitted at 50 g/L feeds and 0.04-0.3 M acid. The scrub holds about
   0.8 M RECl₃, so its ionic strength is more than twice that of the
   fitted data. *K* is conditional, so it may shift;
   the free-exponent fit (n 3.9-4.5, ARCHITECTURE.md Open) points the same
   way. Near saturation the complexes can also change form, which is where
   the cube law is least sure. *Settles it:* isotherms at scrub-section
   strength and up to saturation.
4. **Saponification and the sodium it leaves** (A.1 item 7, A.7). At the
   design point 99 % of the base is spent in stage 0. That's where it would
   go in a plant too, but the mixer's local pH can run high enough for
   emulsions, and 43 % saponified is on the high side of common practice
   (not checked against a project source), where viscosity and gelling
   start to worry. The sodium reaches neither the densities nor the ionic
   strength. *Settles it:* the "full" saponification mode, and a lab check of
   the organic at 43 %.
5. **Reagent accounting.** The reflux scrub takes the product's metal at the
   operator's acid (decision 14). The product leaves the strip at about
   1.5 M acid and the scrub is set to 2.0 M, so the new acid is about 0.5 M,
   but the reagent count charges 2.0 M: about 360 of the 2,308 mol/h. The
   strip also runs with a lot of excess acid, which a plant would cut back.
   The separation isn't affected; the reagent numbers, the reagent-minimizing
   design and the reagent costs in the control results are. *Settles it:*
   charge only the acid added to the reflux, and let the optimizer move the
   strip acid.
6. **Hold-up the model doesn't have.** The stripped organic goes straight
   back to stage 0. A plant has an organic tank, often saponification
   contactors, and feed and product tanks. They add delay and slow the
   slow modes; the steady state stays the same. *Settles it:* the plant's
   tank volumes.
7. **Interface hydraulics** (A.1 item 8, A.8, A.9, A.11). Besides the one-way
   coupling: no weir head that grows with flow, so a change in organic
   flow doesn't move an interface by itself; free HCl and NaCl left out of
   the aqueous density (a few hundredths of g/mL); and τ_z chosen. These
   touch the interface results, not the separation. *Settles it:* a jackleg
   step test and a flow step test on a real settler.
8. **The perturbed plant's spreads** (decision 20). The Idaho fit scatter
   puts the standard error on ln *K* near 0.15-0.2, above the 10 % used, and
   the free-exponent fit is further from 3 than 3.3. The extractant
   concentration, which drifts in a plant and moves *D* as its cube, isn't
   perturbed. *Settles it:* plant-feed batch tests and organic titrations
   over time.
9. **Entrainment paths left out** (A.12). Only two-way exchanges are built.
   The one-way paths are the ones that hurt: scrub aqueous carried in the
   loaded organic into the strip puts La and Ce straight into the product,
   and organic leaving with the raffinate and product is lost. Entrainment
   also depends on throughput and crud, not on the interface alone.
   *Settles it:* add the one-way paths; on a plant, assay the product
   against the scrub liquor.
10. **Slow plant-life effects** (A.1 items 1, 3, 11). Temperature (season,
    saponification heat), impurities from a monazite leach (iron, aluminium,
    thorium residues, silica, for instance), crud, organic losses and make-up,
    and third phase or gelling. Iron(III) in particular loads strongly on
    acidic organophosphorus extractants and strips poorly in HCl (textbook),
    so capacity creeps down until the organic is cleaned. None of this moves
    a 60-hour run much; all of it matters over months. *Settles it:* a full
    feed assay and a log of organic capacity on a running plant.
11. **Acidity as instruments see it.** *h* is a concentration. A pH probe
    reads activity, and in 0.8 M RECl₃ or 2 M HCl that differs from −log *h*;
    below pH 0 a probe is no use and plants titrate free acid. The sensor
    study's acid readings assume the probe reads *h*. *Settles it:* calibrate
    the probe in the plant's own liquors, or model a titrator.
12. **Feed and targets.** Carester Table 1 is a deposit's make-up, not a
    leach liquor's; 50 g/L of metal is dilute next to many plant feeds; many
    monazite flowsheets take most of the cerium out before SX; and 95 %
    recovery is low for the element pair that carries most of the value.
    Each changes the design point more than the method. *Settles it:* the
    real leach liquor assay and the plant's own targets.

## Sources named in this appendix

- K. L. Lyon, *Separation of Adjacent Rare Earth Elements Using Solvent
  Extraction*, MSc thesis, University of Idaho, 2016 (equilibrium data
  tables 4–6; equipment table 3; the reaction and its mass-action form).
- A. Rollat, C. Bertuol, A. Lévêque, *Improvements in Rare Earth Separation*,
  Proc. 63rd Conference of Metallurgists (COM 2024), pp. 1529–1532
  (feed compositions, Table 1; the extraction reaction, eq. 1).
- Yin, Pei, Peng, Zhang, Srinivasakannan, *Hydrometallurgy* 175 (2018)
  64–69 (industrial separation factors with saponified P507).
- H. Yang et al., US patent 11,214,850 B2 (2022) (60-stage circuit, feed
  flow, measuring points).
- J. D. Seader, E. J. Henley, D. K. Roper, *Separation Process Principles*
  (Kremser equation; gravity decanter).
