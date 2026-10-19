"""Which instruments are worth their cost? Noise, calibration error and price, scored on the model.

The question. The sensor study (notebooks 07 and 08) compared a handful of hand-picked sensor
sets, by noise only, with no costs. A plant asks a different question: for a given spend, which
instruments, where, tell me product purity best, and early enough to act? And can a cheap
instrument that's precise but not accurate (a UV-Vis spectrometer) earn its place?

The method, all on the model at its design point (no simulation, a few minutes):

  - A draft catalogue of real instrument types (DRAFT_CATALOGUE below): what each reads, its
    noise, its calibration error (a gain error and an offset), how often it reads, how late the
    result comes, and a rough relative cost (about k$ installed). First guesses, to be corrected.

  - Calibration error is drift, not noise: it changes over days, not from one reading to the next,
    and averaging doesn't remove it. Each source of it gets its own state in the Kalman filter, a
    slow drift that wanders back and forth with a one-week time constant (weekly recalibration):
        b_next = a b + w,   a = exp(-step / 168 h),   stationary sd = the catalogue's error
    Which readings share a source is physics:
        XRF       each element's calibration curve is its own: one source per element
        UV-Vis    window fouling and lamp ageing scale every band of one spectrum alike: one gain
                  source per unit, so the Nd/Pr band ratio doesn't see it. Units calibrated against
                  the same standards share most of their baseline offset (UVVIS_SHARED_OFFSET), so a
                  difference between two units doesn't see that part
        lab ICP   one source per element, small
    The filter then works out for itself what a difference or a ratio is worth: a UV-Vis gets used
    for changes, and the lab and XRF anchor the level. Nothing is hand-built for it.

  - The plant's own unknowns are the same for every set: the estimator's five offsets (feed La+Ce,
    Pr, Nd, Sm, organic flow) with its noise, drifting with the same one-week time constant (about
    6.5 % of nominal). Unlike the estimator, which carries no more offsets than it has readings, a
    set that can't see the feed isn't excused from it here.

  - A late reading (the lab: 4 h from sample to result) is a chain of delay states: what the
    filter gets now is the plant as it was then. Reading less often than every step makes the
    noise count for more (R x interval / step).

  - The score: the standard deviation of product purity H hours from now, forecast on the model
    from today's estimate, in thousandths:
        score_H = sqrt(g_H' P g_H),   g_H = (A^H)' g,   P from the Riccati equation
    H = 0 is how well purity is known now. H = 8 h and 24 h are what the controller needs: a change
    that's on its way to the product but hasn't arrived. The forecast's other error, from upsets
    that haven't happened yet, is the same whatever the instruments, so it's left out.

  - A greedy search starts from the lab assay every plant has, adds one at a time the instrument
    that cuts the score most per unit cost, and stops when nothing cuts it by 1 %. Its path is
    compared with sets plants run, and the scrub front (where a shift in the purity/recovery trade
    shows first) is looked at on its own.

Run:  pixi run python notebooks/18_sensor_economics.py
Writes runs/sensor_economics_ndpr_plant.csv and figures/18_sensor_economics_ndpr_plant.png
"""
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Inputs                                          # noqa: E402
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.estimator import Estimator                                     # noqa: E402
from src.metrics import heavy_mask                                      # noqa: E402
from src.plotting import FIGURES, INK, INK2, MUTED, SLOT, section_bands, style   # noqa: E402
from src.sensors import Sensor, SensorSet                               # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = "ndpr_plant"
T_STEP = 0.5                                       # h, the controller's and estimator's step
DRIFT_TIME_H = 168.0                               # chosen: calibration and offsets drift over a week
HORIZONS = (0.0, 8.0, 24.0)                        # h ahead: now, a shift, a day

# ---- 0. the draft catalogue (all chosen, to be corrected) ----
# noise: one standard deviation per reading, relative to the reading unless abs=True
# gain: calibration error as a fraction of the reading; offset: in the reading's units (mol/L, g/mL)
# shared_gain: one gain error for all of the instrument's readings (one spectrum) instead of one each
# interval: h between readings; delay: h from sample to result; cost: rough installed, about k$
DRAFT_CATALOGUE = {
    "xrf":      dict(noise=0.02, gain=0.02, offset=0.0, shared_gain=False, interval=0.25, delay=0.0,
                     cost=200.0, what="on-stream XRF, one point: La+Ce, Pr, Nd, Sm"),
    "uvvis":    dict(noise=0.005, gain=0.05, offset=0.002, shared_gain=True, interval=0.05, delay=0.0,
                     cost=40.0, what="UV-Vis flow cell: Pr and Nd only (La, Ce(III) don't absorb in the visible)"),
    "density":  dict(noise=0.002, gain=0.0, offset=0.002, shared_gain=False, interval=0.05, delay=0.0,
                     cost=20.0, abs=True, what="vibrating-tube density meter, g/mL: total metal by proxy"),
    "titrator": dict(noise=0.02, gain=0.02, offset=0.0, shared_gain=False, interval=0.1, delay=0.0,
                     cost=40.0, what="online free-acid titrator"),
    "lab":      dict(noise=0.005, gain=0.005, offset=0.0, shared_gain=False, interval=2.0, delay=4.0,
                     cost=30.0, what="lab ICP on the product, sampled every 2 h, result 4 h later"),
}
UVVIS_SHARED_OFFSET = 0.9          # chosen: share of a UV-Vis unit's offset variance common to all units


@dataclass
class Instrument:
    """One instrument: its rows of the measurement map, noise per row, calibration sources, cost.

    sources: (key, vector) pairs. The vector is how one standard deviation of that source shifts
    each of the instrument's readings. Two instruments that list the same key share the source."""
    name: str
    kind: str
    cost: float
    C: np.ndarray                  # (m, NX + ND): readings as a linear map of [state; offsets]
    sd: np.ndarray                 # (m,) noise per reading at the operating point
    interval: float                # h between readings
    delay: float                   # h from sample to result
    sources: list = field(default_factory=list)


# ---- 1. the plant at its design point, the purity gradient, the estimator's model ----
rc = load_config(CONFIG)
circ = build_circuit(rc)
p, cmap = circ.p, circ.cmap
N, n, NX = circ.N, circ.n, circ.nx
PRODUCT, RAFF, FEED_STAGE = cmap.product_stage, 0, cmap.n_ext - 1
x0, u0 = commission(circ, nominal_inputs(rc, circ))
x_sym, u_sym = ca.SX.sym("x", NX), ca.SX.sym("u", Inputs.size(n, N))
out = circ._out(x=x_sym, u=u_sym)
heavy = [i for i, m in enumerate(heavy_mask(p, rc.cut)) if m]
purity_expr = ca.sum1(out["c_prod"][heavy]) / ca.sum1(out["c_prod"])
g = np.array(ca.Function("g", [x_sym, u_sym], [ca.jacobian(purity_expr, x_sym)])(x0, u0.numpy())).ravel()

sensors = SensorSet([Sensor("composition_aq", stage=RAFF)], circ)
est = Estimator(circ, sensors, x0, u0, T_STEP)
ND = est.nd
A_DRIFT = np.exp(-T_STEP / DRIFT_TIME_H)
# The plant model every set faces: the estimator's, with all its offsets drifting over a week
# instead of wandering without limit (so an offset no instrument sees has a finite spread).
A_PLANT = est.A_aug.copy()
A_PLANT[NX:, NX:] = A_DRIFT * np.eye(ND)
Q_PLANT = est.Q_aug.copy()
G_AUG = np.concatenate([g, np.zeros(ND)])
FEED_START = Inputs.layout(n, N)["c_feed"].start
feed_cols = {int(c) - FEED_START: k for k, c in enumerate(est._d_cols) if FEED_START <= c < FEED_START + n}
c_feed = np.asarray(u0.c_feed, float)
names = list(p.names)
present = [i for i in range(n) if c_feed[i] > 0]
SEEN_BY = {"uvvis": [names.index("Pr"), names.index("Nd")]}            # what each kind can read
c_aq = np.array([np.asarray(v, float).ravel() for v in circ.unpack(x0)[0]])     # (N, n) mol/L


# ---- 2. building instruments from the catalogue ----
def make(kind: str, stage=None) -> Instrument:
    """An instrument of catalogue `kind` at `stage` (None: on the feed; the lab is on the product)."""
    spec = DRAFT_CATALOGUE[kind]
    comps = SEEN_BY.get(kind, present)
    if kind == "lab":
        stage = PRODUCT
    if stage is None:                                         # feed analyser: reads the feed offsets
        C = np.zeros((len(comps), NX + ND))
        for r, i in enumerate(comps):
            C[r, NX + feed_cols[i]] = 1.0
        return _finish(f"{kind} feed", kind, spec, C, c_feed[comps])
    if kind == "density":
        s = Sensor("density_aq", stage=stage)
        value = np.atleast_1d(sensors.true_value(s, x0, u0.numpy()))
    elif kind == "titrator":
        s = Sensor("acid", stage=stage)
        value = np.atleast_1d(sensors.true_value(s, x0, u0.numpy()))
    else:
        s = Sensor("composition_aq", stage=stage, components=[names[i] for i in comps])
        value = c_aq[stage, comps]
    C = np.hstack([sensors.rows(s), np.zeros((len(value), ND))])
    return _finish(f"{kind} {stage}", kind, spec, C, value)


def _finish(name: str, kind: str, spec: dict, C: np.ndarray, value: np.ndarray) -> Instrument:
    """Noise and calibration sources from the catalogue's numbers and the readings' values."""
    value = np.abs(np.asarray(value, float))
    m = len(value)
    sd = np.full(m, spec["noise"]) if spec.get("abs") else np.maximum(spec["noise"] * value, 1e-4)
    sources = []
    if spec["gain"] > 0:
        if spec["shared_gain"]:                               # one spectrum: one gain for every band
            sources.append((f"{name} gain", spec["gain"] * value))
        else:                                                 # one calibration per reading
            sources += [(f"{name} gain {r}", spec["gain"] * value[r] * np.eye(m)[r]) for r in range(m)]
    if spec["offset"] > 0:
        if kind == "uvvis":                                   # part common to every unit, part its own
            sources.append(("uvvis offset, all units", np.sqrt(UVVIS_SHARED_OFFSET) * spec["offset"] * np.ones(m)))
            sources.append((f"{name} offset", np.sqrt(1 - UVVIS_SHARED_OFFSET) * spec["offset"] * np.ones(m)))
        else:
            sources += [(f"{name} offset {r}", spec["offset"] * np.eye(m)[r]) for r in range(m)]
    return Instrument(name, kind, spec["cost"], C, sd, spec["interval"], spec["delay"], sources)


# ---- 3. the score of a set of instruments ----
def score(instruments: list, horizons=HORIZONS, calibration: bool = True) -> list:
    """Standard deviation of product purity at each horizon (h ahead), in thousandths.
    calibration=False leaves the calibration error out (noise only), to show how much it costs.
    inf if the Riccati equation has no solution for the set."""
    A, Q, C, R, G = _system(instruments, calibration)
    try:
        P = solve_discrete_are(A.T, C.T, Q, R)
    except (np.linalg.LinAlgError, ValueError):
        return [np.inf] * len(horizons)
    result = []
    for h in horizons:
        g_h = np.linalg.matrix_power(A.T, int(round(h / T_STEP))) @ G     # purity h hours on
        result.append(1e3 * float(np.sqrt(max(g_h @ P @ g_h, 0.0))))
    return result


def _system(instruments: list, calibration: bool):
    """The filter's model for this set: [plant state; offsets | calibration sources | delay copies].

    Calibration sources drift (A = a, Q = 1 - a^2, unit variance); each reading adds its vector
    times the source. A late instrument of m readings and d steps adds m x d states: the first copy
    takes C_i x each step, each next copy the one before, and the reading is the last copy."""
    n0 = NX + ND
    keys = list(dict.fromkeys(k for i in instruments for k, _ in i.sources))
    ns = len(keys)
    n_delay = sum(len(i.sd) * int(round(i.delay / T_STEP)) for i in instruments)
    nt = n0 + ns + n_delay
    A, Q = np.zeros((nt, nt)), np.zeros((nt, nt))
    A[:n0, :n0], Q[:n0, :n0] = A_PLANT, Q_PLANT
    A[n0:n0 + ns, n0:n0 + ns] = A_DRIFT * np.eye(ns)
    Q[n0:n0 + ns, n0:n0 + ns] = (1 - A_DRIFT ** 2) * np.eye(ns)
    Q[n0 + ns:, n0 + ns:] = 1e-14 * np.eye(n_delay)            # a hair of noise keeps the equation well posed
    rows, at = [], n0 + ns
    for inst in instruments:
        m, d = len(inst.sd), int(round(inst.delay / T_STEP))
        row = np.zeros((m, nt))
        if d == 0:
            row[:, :n0] = inst.C
        else:
            A[at:at + m, :n0] = inst.C                         # first copy: the reading as of now
            for k in range(1, d):                              # each next copy: the one before, a step later
                A[at + k * m:at + (k + 1) * m, at + (k - 1) * m:at + k * m] = np.eye(m)
            row[:, at + (d - 1) * m:at + d * m] = np.eye(m)    # the reading: the last copy
            at += m * d
        if calibration:
            for key, vec in inst.sources:
                row[:, n0 + keys.index(key)] = vec
        rows.append(row)
    C = np.vstack(rows)
    sd = np.concatenate([i.sd * np.sqrt(max(i.interval, 1e-3) / T_STEP) for i in instruments])
    R = np.diag(np.maximum(sd, 1e-6) ** 2)
    G = np.concatenate([G_AUG, np.zeros(ns + n_delay)])
    return A, Q, C, R, G


# ---- 4. the candidates and the greedy search ----
STAGES = sorted({RAFF, 2, 4, 6, FEED_STAGE, *range(cmap.n_ext + 1, PRODUCT, 2), PRODUCT - 1, PRODUCT})
candidates = ([make("xrf", j) for j in STAGES] + [make("xrf")]
              + [make("uvvis", j) for j in STAGES] + [make("uvvis")]
              + [make("density", j) for j in STAGES]
              + [make("titrator", j) for j in (RAFF, FEED_STAGE, PRODUCT)] + [make("lab")])
LAB = next(c for c in candidates if c.kind == "lab")
rows = []                                          # (part, what, cost, horizon, score, noise only)


def greedy(h: float, max_added: int = 8) -> list:
    """Start from the lab; add the instrument with the largest share of the score cut per unit cost,
    until nothing cuts it by 1 %. Returns [(names, cost, score)]."""
    chosen = [LAB]
    current = score(chosen, (h,))[0]
    path = [([LAB.name], LAB.cost, current)]
    while len(chosen) <= max_added:
        best, best_gain, best_score = None, -np.inf, None
        for c in candidates:
            if c in chosen:
                continue
            s = score(chosen + [c], (h,))[0]
            gain = np.log(current / s) / c.cost if np.isfinite(s) else -np.inf
            if gain > best_gain:
                best, best_gain, best_score = c, gain, s
        if best is None or best_score >= current * 0.99:
            break
        chosen.append(best)
        current = best_score
        path.append(([c.name for c in chosen], sum(c.cost for c in chosen), current))
    return path


paths = {}
for h in HORIZONS:
    paths[h] = greedy(h)
    print(f"\ngreedy, purity {h:.0f} h ahead (thousandths):", flush=True)
    for k, (chosen_names, cost, s) in enumerate(paths[h]):
        added = chosen_names[-1] if k else chosen_names[0] + " (start)"
        print(f"  + {added:16s} cost {cost:5.0f}   {s:7.3f}")
        rows.append(("greedy", f"{k}: + {added}", cost, h, s, np.nan))

# ---- 5. sets plants run, for comparison ----
mid_ext, mid_scrub = cmap.n_ext // 2, cmap.n_ext + cmap.n_scrub // 2
typical = {
    "lab only": [make("lab")],
    "lab + XRF on raffinate, product": [make("lab"), make("xrf", RAFF), make("xrf", PRODUCT)],
    "lab + XRF on feed, raffinate, product": [make("lab"), make("xrf"), make("xrf", RAFF), make("xrf", PRODUCT)],
    "lab + XRF mid-extraction, mid-scrub (patent pair)": [make("lab"), make("xrf", mid_ext), make("xrf", mid_scrub)],
    "lab + UV-Vis on feed, raffinate, product": [make("lab"), make("uvvis"), make("uvvis", RAFF),
                                                 make("uvvis", PRODUCT)],
}


def report(label: str, insts: list, part: str, cost=None) -> None:
    """Print and keep one set's scores, with and without calibration error."""
    cost = sum(i.cost for i in insts) if cost is None else cost
    full, noise = score(insts), score(insts, calibration=False)
    print(f"  {label:52s} {cost:5.0f}   " + "   ".join(f"{a:6.3f} ({b:5.3f})" for a, b in zip(full, noise)))
    rows.extend((part, label, cost, h, a, b) for h, a, b in zip(HORIZONS, full, noise))


print("\nsets plants run: cost, then purity sd in thousandths now / 8 h / 24 h ahead (noise only):")
for name, insts in typical.items():
    report(name, insts, "typical")

# ---- 6. the scrub front ----
# Long-run change in every stage's aqueous Pr for a small rise in scrub flow (the handle that
# trades recovery for purity): dx_ss = -J^-1 (dF/du) du at the design point. The design sits on a
# steep edge: on the full model, -1 L/h of scrub (of 239) costs 9.8 thousandths of purity in the
# long run and +1 L/h gains 2.9, so this linear slope is a local one.
J = circ.jac(x0, u0.numpy())
col = Inputs.layout(n, N)["Q_scrub"].start
dx = -np.linalg.solve(J, circ.jac_u(x0, u0.numpy())[:, col])                  # per +1 L/h of scrub
d_caq = dx[: N * n].reshape(N, n)
i_pr = names.index("Pr")
noise_pr = DRAFT_CATALOGUE["uvvis"]["noise"] * np.maximum(c_aq[:, i_pr], 1e-6)
signal = np.abs(d_caq[:, i_pr]) / noise_pr                       # how many noise sds one move shows
front = int(np.argmax(signal[:PRODUCT]))
print(f"\nscrub front (scrub flow {float(u0.Q_scrub):.0f} L/h): per +1 L/h, aqueous Pr moves most, relative to "
      f"UV-Vis noise, at stage {front} ({signal[front]:.0f} noise sds); purity moves {1e3 * g @ dx:+.2f} thousandths")
base = typical["lab + XRF on raffinate, product"]
print("lab + XRF raffinate/product, plus front instruments:")
report("(nothing added)", base, "front", cost=0.0)
for label, extra in [(f"+ UV-Vis {front}", [make("uvvis", front)]),
                     (f"+ UV-Vis {front - 2} and {front + 2}", [make("uvvis", front - 2), make("uvvis", front + 2)]),
                     (f"+ UV-Vis {front} and {PRODUCT}", [make("uvvis", front), make("uvvis", PRODUCT)]),
                     (f"+ density {front}", [make("density", front)]),
                     (f"+ XRF {front}", [make("xrf", front)])]:
    report(label, base + extra, "front", cost=sum(i.cost for i in extra))

# ---- 7. write the table and the figure ----
out_csv = HERE / "runs" / f"sensor_economics_{CONFIG}.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["part", "what", "cost", "ahead_h", "purity_sd_thousandths", "noise_only"])
    w.writerows(rows)
print("wrote", out_csv)

style()
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
for ax, h in zip(axes[:2], (0.0, 24.0)):
    path = paths[h]
    ax.plot([q[1] for q in path], [q[2] for q in path], "o-", color=SLOT[0], markersize=4,
            label="greedy: most certainty per unit cost")
    for q in path:
        ax.annotate(q[0][-1], (q[1], q[2]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2)
    for k, (name, insts) in enumerate(typical.items()):
        ax.plot([sum(i.cost for i in insts)], [score(insts, (h,))[0]], "s", color=SLOT[1 + k % 5], markersize=5,
                label=name)
    ax.set_xlabel("rough installed cost, about k$ (draft)")
    ax.set_yscale("log")
    ax.set_title(f"Purity {'now' if h == 0 else f'{h:.0f} h ahead'}: certainty per spend", loc="left",
                 color=INK, fontsize=9)
axes[0].set_ylabel("purity sd (noise + calibration), thousandths")
axes[0].legend(frameon=False, fontsize=6, loc="upper right")
ax2 = axes[2]
ax2.plot(range(N), signal, color=SLOT[0])
ax2.axvline(front, color=MUTED, ls=":", lw=0.8)
section_bands(ax2, cmap)
ax2.set_xlabel("stage (0 = raffinate end)")
ax2.set_ylabel("Pr change per +1 L/h scrub, in UV-Vis noise sds")
ax2.set_yscale("log")
ax2.set_title("Where the scrub front moves", loc="left", color=INK, fontsize=9)
fig.savefig(FIGURES / f"18_sensor_economics_{CONFIG}.png")
print("wrote", FIGURES / f"18_sensor_economics_{CONFIG}.png")
