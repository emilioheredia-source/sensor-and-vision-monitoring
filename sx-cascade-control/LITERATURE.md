# What this project takes from the literature

Rare earth solvent extraction is hard to control, and other people have worked on the same
problems, in rare earth plants and in distillation, its closest well-studied relative. This file
lists what worked for them and where this project uses it.

## Learning the plant's numbers while it runs

**Huang, Patwardhan and Biegler**, *Offset-free nonlinear model predictive control based on moving
horizon estimation for an air separation unit*, DYCOPS 2010. They paired a nonlinear MPC with an
estimator that looks back over a short window of readings. When they knew which of the model's
numbers were uncertain, estimating those numbers on the running plant removed the model error faster
than adding correction terms, and they recommend doing so when the uncertain numbers are known.

Here the uncertain numbers are known: the two separation factors and the acid exponent (notebook 43).
Every two hours the controller fits those three numbers to the last two hours of readings and
rebuilds itself on the new model (notebook 51). On the plant where a slightly wrong model drove the
saponification too high, the first refit already had the numbers right, and from then on the
controller moved the handles the way it does with the plant's own model. On all ten test plants it
removed what the wrong model added: no stall, and about the same time below spec as with the plant's
own model.

**Kuntz and Rawlings**, *Offset-free model predictive control: stability under plant-model mismatch*,
TWCCC report 2024-04. Estimating parameters together with the state converges only when the data
carry enough information. Here the steady profile along the stages pins the three numbers without
moving the plant (notebook 43, the identification test), which is why a short window is enough.

## Where to put the correction

**Pannocchia and Rawlings**, *Robustness of MPC and disturbance models for multivariable
ill-conditioned processes*, TWMCC report 2001-02. Industrial MPC corrects model error with a
constant added to its predicted outputs. On an ill-conditioned plant, they showed a correction
placed where the error enters, at the inputs, is more robust to model error and follows slow
disturbances faster. This plant is ill-conditioned (FINDINGS section 4). The controller here still
uses the output correction; refitting the numbers that are wrong (above) puts the fix where the
error is. An input correction was screened on this plant's own gains before building it (notebook
63): with an accurate model it was about ten times better, as the paper found, but it works through
the model's inverse, and with this model's errors it went unstable on most plants. So it's not built.

## Tying the saponification to the metal

**Yang, Xu, Lu and Ding**, *Component content distribution profile control in rare earth
countercurrent extraction process*, Chin. J. Chem. Eng. 23 (2015). Their Pr/Nd process sets the flows
from countercurrent extraction theory and trims them; the saponification is tied to the scrub acid,
the organic flow and the feed metal by a balance, and they note that a small change in any one of
these can make the extraction impossible.

Here the balance is counted from what the plant shows: three times the metal the loaded organic
carries out, plus the acid coming in with the feed and the scrub, less what the raffinate keeps at
its acid floor, sets the most saponification the plant can use. The controller may not go above it
(notebook 52). The balance uses none of the model's fitted numbers, so it holds when the model is
wrong. On the plant above, with the wrong model and no refit, the extraction acid stayed above
13 mmol/L instead of running out. On all ten test plants, with any of the models, the extraction acid
never went under its floor, at no cost in time below spec; over 600 hours the cap holds it at the
floor itself on some plants.

They also place monitoring stages five to fifteen stages from each outlet, where composition moves
fastest, with targets taken from the design. This project's optical points go in the same place, at
the front in the middle of the scrub (FINDINGS phase 5).

## Reading the profile

**Mejdell and Skogestad**, *Estimation of distillation compositions from multiple temperature
measurements using partial-least-squares regression*, Ind. Eng. Chem. Res. 30 (1991). Several stage
readings estimate product composition much better than one, and are less sensitive to where the
sensor sits. Working in logarithms of composition was the single biggest help against the process's
nonlinearity. This project's gain analysis already works in log impurity and log loss (notebook 24),
and several optical stages are read, not one.

**Chew, Jones and Wilson**, *Distillation column control using the whole temperature profile*,
IChemE Symposium Series 152 (2006). They estimate the feed composition from the measured profile and
recompute the best reachable steady state at every step; the profile shows a disturbance near the
feed long before it reaches the products. The estimator here works the same way: from the profile
readings back to the plant's numbers. The controller also looks ahead the way they do: every 12 hours
it solves its own model to rest on the handles held now and holds its long-run purity and recovery
floors there (notebook 55). Two of the ten test plants had been heading for about 94.5 % purity while
the reading said 100 %; with the check they settle at 99.97 to 99.99 %, at the same time below spec
and nearly the same reagent.

## Rare earth practice

**Zhang, Zhu, Xiong, Yang and Nie**, *Constrained model predictive control based on event
triggering for the rare earth extraction process*, ACS Omega 8 (2023), and **Yang et al.**, *Digital
twin key technology on rare earth process*, Sci. Rep. (2022). In industry the reagent dosage is still
set by offline analysis and the operators' experience, and the line takes a long time to settle
after a change. Operators judge composition by the solution's colour, and camera-based soft sensors
are being built on that. The MPC in the first paper moves the extractant and scrub flows only when
the product leaves a band.

## pH

**Kordosky** (1979), as summarized by 911Metallurgist, *pH control in solvent extraction circuits*:
pH probes in the mixer give about +-0.1 pH control on a commercial circuit, with standard probes in
an aqueous-continuous mix. The controller here reads the scrub acid back on the line; probes on the
extraction and scrub stages are the next measurement to add.

