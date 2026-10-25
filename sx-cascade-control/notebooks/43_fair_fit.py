"""A fair test of the physics fit against the learned estimator: Tests 1 to 3 of the plan.

The question (Oct 21, Emilio, in a review session). Notebooks 35 and 42 compare a least-squares fit of
three numbers (the Pr/(La+Ce) and Nd/Pr separation factors and one shared stage-rate factor) with a
net trained on simulated plants. The perturbed plant differs from the model in more than those
three numbers, so the fit may be handicapped.

Test 1, sanity. The plant is the model with only the fit's three numbers changed (the separation
factors of test plant `seed`, and its mean stage efficiency as the shared rate factor); everything
else nominal. The fit (notebook 31's: from zero, bounds +-0.5, step 0.02, 60 evaluations) should
recover them to within instrument noise. Its error is compared with the Fisher lower bound on its
standard error at the truth: |error| / bound should look like a standard normal. The net and the
ridge (notebook 35, trained on the main case) are run for reference, and the MPC (main case, over
level loops) is built on each estimate and run 120 h on that plant, with the true model as the
reference. --multistart adds three more starts on the Pr/(La+Ce) factor.

Test 2, which numbers to fit. Candidates (study_tools.SPREAD): both separation factors, a common
factor on every extraction constant, the acid exponent, the stage rate per section, the organic
density, the mixer hold-up and the salt's molal volume.
  identify   at the model's design point: the readings' sensitivity to each candidate, scaled by
             the instrument noise; the Fisher information, its eigenvalues and condition number,
             the correlation matrix, the lower bound on each standard error against the
             candidate's plausible spread, and the same for a few candidate sets.
  gains      the steady-state gains from the four handles to log impurity and log loss, with the
             candidate at nominal and at plus and minus its spread.
  loop       the plant = the model with that one candidate at plus or minus its spread; the MPC on
             the nominal model and on the right one run it 120 h from the design handles.
  check      a fitted set on the perturbed plants 1-10 (notebook 35's readings and noise draws),
             four starts, settled solves; the MPC on each fit, 120 h.
Test 3 is `check` with --map (a prior at each number's plausible spread) and --uv (more optical
stages, appended so the first twelve readings and their noise draws stay notebook 35's).

Run:  pixi run python notebooks/43_fair_fit.py test1 [--no-loop] [--multistart]
      pixi run python notebooks/43_fair_fit.py identify [--uv 2,4,6,12,16,20] | gains | loop
      pixi run python notebooks/43_fair_fit.py check --sets "beta_pl,beta_np,dn_acid;beta_pl,beta_np,k_all" [--map] [--uv ...]
Writes runs/fair_fit_<test>_<config>[_uv...][_map].csv (and .npz for identify).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.simulate import steady_state                                   # noqa: E402
from study_tools import (SPREAD, PlantCase, closed_loop, extended_model, notebook_35,   # noqa: E402,F401
                         purity_recovery, run_jobs, separation_factors, settle, workers, write_rows)

CONFIG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "ndpr_main"
SEEDS = tuple(range(1, 11)) if CONFIG == "ndpr_main" else (2, 3, 6, 10, 14, 16, 20, 1, 7, 12)
WORKERS = workers()
MULTISTART = "--multistart" in sys.argv
UV = tuple(int(v) for v in sys.argv[sys.argv.index("--uv") + 1].split(",")) if "--uv" in sys.argv else (2, 4, 6)
TAG = "" if UV == (2, 4, 6) else "_uv" + "-".join(map(str, UV))
MAP = "--map" in sys.argv
CANDIDATES = tuple(SPREAD)
BOUNDS = {"dn_acid": (-1.0, 1.0)}                # every other fitted number +-0.5 in log, as notebook 31
STARTS = (0.0, 0.15, 0.3, -0.15)                # Pr/(La+Ce) starts: one start falls into a second minimum
CANDIDATE_SETS = (("beta_pl", "beta_np", "dn_acid"), ("beta_pl", "beta_np", "k_ext"),
                  ("beta_pl", "beta_np", "dn_acid", "common_K"), ("beta_pl", "beta_np", "dn_acid", "k_ext"),
                  ("beta_pl", "beta_np", "dn_acid", "k_scrub"), ("beta_pl", "beta_np", "dn_acid", "k_ext", "k_scrub"),
                  ("beta_pl", "beta_np", "dn_acid", "k_ext", "k_scrub", "k_strip"))


def nb35():
    """Notebook 35 on this run file (its estimators, plants and helpers)."""
    return notebook_35(CONFIG)


def instruments(n35):
    """Notebook 31's readings and noise (through notebook 35), on this run's optical stages."""
    m = n35.nb31()
    m.UV_STAGES = UV
    return m


def noisy_readings(m, case: PlantCase, seed: int):
    """The plant's readings at rest with notebook 31's noise draw for this plant: (readings, noise sd)."""
    names = list(case.plant.p.names)
    y_true = m.readings(case.plant, case.x_rest, case.u_design.replace(k_factor=case.plant_k), names)
    sd = m.noise_sd(y_true)
    return y_true + np.random.default_rng(seed + 5000).normal(0.0, sd), sd


# ---- Test 1 -----------------------------------------------------------------------------------------
def test1_one(seed: int) -> dict:
    """Test 1 on one plant: the fit, its Fisher bound, the learned estimators, and (unless --no-loop)
    the MPC on each estimate. Returns the row for runs/fair_fit_test1_*.csv."""
    warnings.filterwarnings("ignore")
    n35 = nb35()
    m = instruments(n35)
    case, theta_true, u0 = sanity_case(n35, m, seed)
    y, sd = noisy_readings(m, case, seed)
    residuals, state = fit_residuals(m, case, y, sd)
    t0 = time.time()
    fit = sanity_fit(residuals)
    fit_s, fit_evals, fit_unconverged = time.time() - t0, state["evals"], state["unconverged"]   # the fit's alone
    bound = fisher_bound(residuals, theta_true, len(y))
    cost_truth = 0.5 * float(np.sum(residuals(theta_true) ** 2))
    est = learned_estimates(n35, y, n35.handles_of(case.u_design, u0))
    row = test1_row(seed, case, {"theta_true": theta_true, "fit": fit, "bound": bound}, est)
    row.update({"cost_truth": cost_truth, "fit_s": round(fit_s, 1), "fit_evals": fit_evals,
                "fit_unconverged": fit_unconverged})
    if "--no-loop" not in sys.argv:
        row.update(test1_loops(m, case, {"true": theta_true, "fit": fit.x, "net": [*est["net"][:2], 0.0],
                                         "ridge": [*est["ridge"][:2], 0.0]}))
    return row


def sanity_case(n35, m, seed: int):
    """The model with only the fit's three numbers changed to plant `seed`'s, at rest on the design
    handles. Returns (PlantCase, the true three numbers, the design inputs)."""
    rc, pert, pert_k = n35.plant_for(seed)
    b = n35.true_log_betas(pert, build_circuit(rc))
    theta_true = np.array([b[0], b[1], float(np.log(np.mean(pert_k)))])
    plant, _, _ = m.model_with(rc, theta_true)
    u0 = nominal_inputs(rc, plant).replace(k_factor=np.ones(plant.N))
    x_rest, u_design = commission(plant, u0)
    return PlantCase(rc, plant, np.ones(plant.N), settle(plant, u_design, x0=x_rest), u_design), theta_true, u0


def fit_residuals(m, case: PlantCase, y, sd):
    """Notebook 31's residual function on this plant's readings, solved as notebooks 31 and 35 solve it
    (default steady-state ladder, warm-started from the last solve). Returns (residuals, state), state
    counting evaluations and unconverged solves."""
    names = list(case.plant.p.names)
    state = {"x": case.plant.observable(case.x_rest), "evals": 0, "unconverged": 0}

    def residuals(theta):
        """Noise-scaled reading errors of the model with these three log numbers."""
        mod, _, _ = m.model_with(case.rc, theta)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            x = steady_state(mod, case.u_design, x0=state["x"])
        state["unconverged"] += sum("not converged" in str(c.message) for c in caught)
        state["x"], state["evals"] = x, state["evals"] + 1
        return (m.readings(mod, x, case.u_design, names) - y) / sd
    return residuals, state


def sanity_fit(residuals):
    """Notebook 31's fit from zero; with --multistart, three more starts on Pr/(La+Ce), lowest cost kept."""
    from scipy.optimize import least_squares
    fit = least_squares(residuals, np.zeros(3), bounds=(-0.5, 0.5), diff_step=0.02, max_nfev=60)
    if MULTISTART:
        for s0 in STARTS[1:]:
            f = least_squares(residuals, np.array([s0, 0.0, 0.0]), bounds=(-0.5, 0.5), diff_step=0.02, max_nfev=60)
            fit = f if f.cost < fit.cost else fit
    return fit


def fisher_bound(residuals, theta, n_readings: int) -> np.ndarray:
    """Lower bound on each number's standard error at theta: (J'J)^-1 of the noise-scaled readings,
    J by central differences (step 0.01)."""
    J = np.zeros((n_readings, len(theta)))
    for j in range(len(theta)):
        d = np.zeros(len(theta))
        d[j] = 0.01
        J[:, j] = (residuals(theta + d) - residuals(theta - d)) / 0.02
    return np.sqrt(np.diag(np.linalg.inv(J.T @ J)))


def learned_estimates(n35, y, handles) -> dict:
    """The net's and the ridge's log factors from these readings, and the net's time, ms."""
    net_estimate, ridge_estimate = n35.load_estimators()
    t1 = time.perf_counter()
    net = net_estimate(y, handles)
    return {"net": net, "net_ms": 1000 * (time.perf_counter() - t1), "ridge": ridge_estimate(y, handles)}


def test1_row(seed: int, case: PlantCase, result: dict, est: dict) -> dict:
    """Test 1's numbers for one plant: true, fitted and learned factors, and the fit's errors in bounds.
    result: the true three numbers, the fit and its Fisher bound."""
    theta_true, fit, bound = result["theta_true"], result["fit"], result["bound"]
    b0 = separation_factors(build_circuit(case.rc))
    row = {"seed": seed, "beta_pl_true": b0[0] * np.exp(theta_true[0]), "beta_np_true": b0[1] * np.exp(theta_true[1]),
           "k_true": np.exp(theta_true[2]), "beta_pl_fit": b0[0] * np.exp(fit.x[0]),
           "beta_np_fit": b0[1] * np.exp(fit.x[1]), "k_fit": np.exp(fit.x[2]), "cost_fit": float(fit.cost),
           "net_ms": round(est["net_ms"], 3)}
    for i, key in enumerate(("pl", "np", "k")):
        row[f"z_{key}"] = (fit.x[i] - theta_true[i]) / bound[i]
        row[f"bound_{key}_pct"] = 100 * bound[i]
    for name in ("net", "ridge"):
        row[f"beta_pl_{name}"] = b0[0] * np.exp(est[name][0])
        row[f"beta_np_{name}"] = b0[1] * np.exp(est[name][1])
    return row


def test1_loops(m, case: PlantCase, thetas: dict) -> dict:
    """The MPC on each estimate's model, 120 h on the plant; columns <estimate>_<measure>."""
    out, ones = {}, np.ones(case.plant.N)
    for name, theta in thetas.items():
        mod, _, _ = m.model_with(case.rc, theta)
        out.update({f"{name}_{k}": v for k, v in closed_loop(case, mod, ones).items()})
    return out


def test1():
    """Test 1 on every plant, written row by row."""
    out = HERE / "runs" / (f"fair_fit_test1_{CONFIG}" + ("_multistart" if MULTISTART else "")
                           + ("_noloop" if "--no-loop" in sys.argv else "") + ".csv")
    rows = []
    for _, r in run_jobs(test1_one, SEEDS, WORKERS):
        rows.append(r)
        print(f"seed {r['seed']:2d}: Pr/LaCe true {r['beta_pl_true']:.3f} fit {r['beta_pl_fit']:.3f} (z {r['z_pl']:+.2f}) "
              f"net {r['beta_pl_net']:.3f} ridge {r['beta_pl_ridge']:.3f} | Nd/Pr true {r['beta_np_true']:.3f} fit "
              f"{r['beta_np_fit']:.3f} (z {r['z_np']:+.2f}) | cost fit {r['cost_fit']:.1f} truth {r['cost_truth']:.1f} | "
              f"{r['fit_evals']} evals {r['fit_s']} s", flush=True)
        write_rows(out, rows, first=("seed",))
    print("wrote", out.name)


# ---- Test 2: identifiability ---------------------------------------------------------------------
def sensitivity_matrix(rc, m) -> np.ndarray:
    """J: each reading's change per unit of each candidate, in noise widths, at the model's design
    point, by central differences (step 0.05 for the acid exponent, 0.01 in log for the rest)."""
    model = build_circuit(rc)
    u_nom = commission(model, nominal_inputs(rc, model))[1]

    def readings_at(theta: dict, x0=None):
        """The model's readings at rest with these candidate numbers changed: (readings, state)."""
        circ, kf = extended_model(rc, theta)
        u = u_nom.replace(k_factor=kf)
        x = settle(circ, u, x0=x0)
        return m.readings(circ, x, u, list(circ.p.names)), x
    y0, x0 = readings_at({})
    sd = m.noise_sd(y0)
    J = np.zeros((len(y0), len(CANDIDATES)))
    for j, c in enumerate(CANDIDATES):
        h = 0.05 if c == "dn_acid" else 0.01
        J[:, j] = (readings_at({c: h}, x0)[0] - readings_at({c: -h}, x0)[0]) / (2 * h) / sd
        print(f"{c:8s} largest |d reading / d param| in noise widths: {np.abs(J[:, j]).max():9.2f}", flush=True)
    return J


def fisher_summary(J: np.ndarray) -> dict:
    """The Fisher information of the candidates the readings see, and what follows from it.

    A candidate is seen if one plausible spread moves some reading by 1 % of its noise width.
    Returns F (per unit), live (seen candidates' indices), ev (eigenvalues per spread), corr and the
    rows of the identify table (bound alone and with every seen candidate fitted together)."""
    F = J.T @ J
    live = [i for i, c in enumerate(CANDIDATES) if np.abs(J[:, i]).max() * SPREAD[c] > 0.01]
    S = np.diag([SPREAD[c] for c in CANDIDATES])
    ev = np.linalg.eigvalsh((S @ F @ S)[np.ix_(live, live)])
    cov = np.linalg.pinv(F[np.ix_(live, live)])
    sdp = np.sqrt(np.clip(np.diag(cov), 0, None))
    rows = [{"param": CANDIDATES[i], "spread": SPREAD[CANDIDATES[i]], "bound_all": sdp[k],
             "bound_alone": 1 / np.sqrt(F[i, i]), "max_sens_noise_widths": np.abs(J[:, i]).max()}
            for k, i in enumerate(live)]
    rows += [{"param": c, "spread": SPREAD[c], "bound_all": np.inf, "bound_alone": np.inf,
              "max_sens_noise_widths": np.abs(J[:, i]).max()} for i, c in enumerate(CANDIDATES) if i not in live]
    return {"F": F, "live": live, "ev": ev, "corr": cov / np.outer(sdp, sdp), "rows": rows}


def set_summary(F: np.ndarray, fit_set: tuple) -> str:
    """One candidate set fitted together: each bound in spreads, condition number, worst correlation."""
    ix = [CANDIDATES.index(c) for c in fit_set]
    Fs = F[np.ix_(ix, ix)]
    if np.linalg.matrix_rank(Fs) < len(fit_set):
        return f"  {'+'.join(fit_set):45s} singular"
    C = np.linalg.inv(Fs)
    sdv = np.sqrt(np.diag(C))
    worst = np.max(np.abs((C / np.outer(sdv, sdv))[np.triu_indices(len(fit_set), 1)]))
    ev = np.linalg.eigvalsh(Fs)
    return (f"  {'+'.join(fit_set):45s} " + " ".join(f"{c} {v / SPREAD[c]:.2f}" for c, v in zip(fit_set, sdv))
            + f" | cond {ev.max() / ev.min():.3g} | worst corr {worst:.3f}")


def print_identify(s: dict) -> None:
    """The identify results, as printed to the log."""
    print("\nno effect on any reading:", [c for i, c in enumerate(CANDIDATES) if i not in s["live"]])
    print("eigenvalues of the Fisher information per spread (live):", np.array2string(s["ev"], precision=3))
    print(f"condition number {s['ev'].max() / max(s['ev'].min(), 1e-300):.3g}")
    print("lower bound on each standard error, all live candidates fitted together, against its spread:")
    for r in s["rows"]:
        print(f"  {r['param']:8s} spread {r['spread']:.3f}  alone {r['bound_alone']:.4f}  together {r['bound_all']:.4f}"
              f"  ({r['bound_all'] / r['spread']:.2f} spreads)")
    labels = [CANDIDATES[i] for i in s["live"]]
    print("correlations (live):\n          " + " ".join(f"{c:>8s}" for c in labels))
    for k, c in enumerate(labels):
        print(f"{c:9s} " + " ".join(f"{v:8.3f}" for v in s["corr"][k]))
    print("candidate sets (lower bound / spread for each; condition number; worst correlation):")
    for fit_set in CANDIDATE_SETS:
        print(set_summary(s["F"], fit_set))


def identify():
    """Test 2's identifiability at the model's design point; writes the table and the matrices."""
    J = sensitivity_matrix(load_config(CONFIG), instruments(nb35()))
    s = fisher_summary(J)
    out = HERE / "runs" / f"fair_fit_identify_{CONFIG}{TAG}.csv"
    write_rows(out, s["rows"], first=("param", "spread", "bound_all", "bound_alone", "max_sens_noise_widths"))
    np.savez(HERE / "runs" / f"fair_fit_identify_{CONFIG}{TAG}.npz", J=J, corr=s["corr"], live=np.array(s["live"]),
             names=np.array(CANDIDATES), eig_per_spread=s["ev"])
    print_identify(s)
    print("wrote", out.name)


# ---- Test 2: what each candidate does to the controller ----------------------------------------
def gains_matrix(rc, theta: dict):
    """d (log10 impurity, log10 loss) / d ln(Q_org, Q_scrub, h_scrub, saponified_fraction) at steady
    state, with these candidate numbers changed, and the outputs at the design handles."""
    circ, kf = extended_model(rc, theta)
    model = build_circuit(rc)
    u0 = commission(model, nominal_inputs(rc, model))[1].replace(k_factor=kf)
    x0 = settle(circ, u0)

    def y(u):
        """log10 impurity and log10 loss at rest on inputs u."""
        p_, r_ = purity_recovery(circ, settle(circ, u, x0=x0), u.numpy(), rc.cut)
        return np.array([np.log10(max(1 - p_, 1e-12)), np.log10(max(1 - r_, 1e-12))])

    G = np.column_stack([(y(moved(u0, h, np.exp(0.005))) - y(moved(u0, h, np.exp(-0.005)))) / 0.01
                         for h in ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")])
    return G, y(u0)


def moved(u0, handle: str, factor: float):
    """u0 with one handle multiplied by factor; the strip flow follows the scrub flow (net product fixed)."""
    change = {handle: float(getattr(u0, handle)) * factor}
    if handle == "Q_scrub":
        change["Q_strip"] = float(u0.Q_strip) - float(u0.Q_scrub) + change["Q_scrub"]
    return u0.replace(**change)


def gains_one(job):
    """One candidate at plus or minus its spread (None: nominal): (candidate, sign, G, outputs)."""
    c, sign = job
    warnings.filterwarnings("ignore")
    G, y = gains_matrix(load_config(CONFIG), {} if c is None else {c: sign * SPREAD[c]})
    return c, sign, G, y


def gains():
    """The gain change each candidate makes, against the nominal gains; written to a table."""
    jobs = [(None, 0)] + [(c, s) for c in CANDIDATES for s in (1, -1)]
    res = {job: r for job, r in run_jobs(gains_one, jobs, WORKERS)}
    _, _, G0, y0 = res[(None, 0)]
    print("nominal G (rows log impurity, log loss; cols ln organic, scrub, scrub acid, saponification):\n", G0.round(2))
    rows = [{"param": c, "sign": s, **gain_change((G, y), (G0, y0))} for c, s, G, y in (res[j] for j in jobs[1:])]
    for r in rows:
        print(f"{r['param']:8s} {r['sign']:+d} spread: |dG|/|G| {r['rel_change_G']:.3f}  model error (flows) "
              f"{r['model_error_flows']:.3f}  shift log impurity {r['d_log_impurity']:+.3f}, log loss {r['d_log_loss']:+.3f}")
    out = HERE / "runs" / f"fair_fit_gains_{CONFIG}.csv"
    write_rows(out, rows, first=("param", "sign"))
    print("wrote", out.name)


def gain_change(changed: tuple, nominal: tuple) -> dict:
    """How far one candidate's gains are from the nominal ones: relative change over all four handles,
    notebook 24's model-error measure on the two flows, and the shift in the outputs. Each argument is
    (G, outputs) as gains_matrix returns them."""
    (G, y), (G0, y0) = changed, nominal
    E = (G[:, :2] - G0[:, :2]) @ np.linalg.inv(G0[:, :2])
    return {"rel_change_G": np.linalg.norm(G - G0) / np.linalg.norm(G0),
            "model_error_flows": np.linalg.svd(E, compute_uv=False)[0],
            "d_log_impurity": y[0] - y0[0], "d_log_loss": y[1] - y0[1]}


def loop_one(job) -> dict:
    """One candidate at plus or minus its spread in the plant: the MPC on the nominal model (that number
    wrong) and on the plant's own (right), 120 h each. Returns one row, nom_* and true_* columns."""
    c, sign = job
    warnings.filterwarnings("ignore")
    rc = load_config(CONFIG)
    plant, plant_k = extended_model(rc, {} if c is None else {c: sign * SPREAD[c]})
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=plant_k))
    case = PlantCase(rc, plant, plant_k, settle(plant, u_design, x0=x_rest), u_design.replace(k_factor=np.ones(plant.N)))
    t0 = time.time()
    nom = closed_loop(case, build_circuit(rc), np.ones(plant.N))
    true = closed_loop(case, plant, plant_k)
    return {"param": c or "none", "sign": sign, **{f"nom_{k}": v for k, v in nom.items()},
            **{f"true_{k}": v for k, v in true.items()}, "seconds": round(time.time() - t0)}


def loop():
    """loop_one for every candidate, both signs, and the nominal plant; written row by row."""
    jobs = [(None, 0)] + [(c, s) for c in CANDIDATES for s in (1, -1)]
    out = HERE / "runs" / f"fair_fit_loop_{CONFIG}.csv"
    rows = []
    for _, r in run_jobs(loop_one, jobs, WORKERS):
        rows.append(r)
        print(f"{r['param']:8s} {r['sign']:+d}: " + " | ".join(
            f"{w} model: purity end {100 * r[w + '_purity_end']:.2f} %, recovery end {100 * r[w + '_recovery_end']:.2f} %, "
            f"at rest {100 * r[w + '_rest_purity']:.2f} %" for w in ("nom", "true")), flush=True)
        write_rows(out, rows, first=("param", "sign"))
    print("wrote", out.name)


# ---- Tests 2 and 3: a fitted set, in closed loop ------------------------------------------------
def check_one(job) -> dict:
    """One perturbed plant: fit the set to notebook 35's readings (four starts, settled solves, a
    prior with --map), then the MPC on the fit, 120 h. Returns one row."""
    seed, fit_set = job
    warnings.filterwarnings("ignore")
    n35 = nb35()
    m = instruments(n35)
    rc, plant, plant_k = n35.plant_for(seed)
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=plant_k))
    case = PlantCase(rc, plant, plant_k, settle(plant, u_design, x0=x_rest), u_design.replace(k_factor=np.ones(plant.N)))
    residuals, state = set_residuals(m, case, fit_set, noisy_readings(m, case, seed))
    t0 = time.time()
    fit = multistart_fit(residuals, fit_set)
    row = check_row(seed, case, fit_set, fit)
    row.update({"evals": state["evals"], "fit_s": round(time.time() - t0)})
    fitted, model_k = extended_model(rc, dict(zip(fit_set, fit.x)))
    row.update(closed_loop(case, fitted, model_k))
    return row


def set_residuals(m, case: PlantCase, fit_set: tuple, readings: tuple):
    """The residual function for a fitted set on (readings, noise sd) (settled solves, warm-started;
    prior rows with --map). Returns (residuals, state counting evaluations)."""
    y, sd = readings
    names = list(case.plant.p.names)
    state = {"x": case.plant.observable(case.x_rest), "evals": 0}
    prior_sd = np.array([SPREAD[c] for c in fit_set])

    def residuals(theta):
        """Noise-scaled reading errors (and theta over its prior spread, with --map)."""
        circ, kf = extended_model(case.rc, dict(zip(fit_set, theta)))
        u = case.u_design.replace(k_factor=kf)
        x = settle(circ, u, x0=state["x"])
        state["x"], state["evals"] = x, state["evals"] + 1
        r = (m.readings(circ, x, u, names) - y) / sd
        return np.concatenate([r, np.asarray(theta) / prior_sd]) if MAP else r
    return residuals, state


def fit_bounds(fit_set: tuple) -> tuple:
    """(lower, upper) bound for each fitted number."""
    return ([BOUNDS.get(c, (-0.5, 0.5))[0] for c in fit_set], [BOUNDS.get(c, (-0.5, 0.5))[1] for c in fit_set])


def multistart_fit(residuals, fit_set: tuple):
    """The fit from each start in STARTS on Pr/(La+Ce) (others at zero), the lowest cost kept. Why:
    from the nominal alone the search falls into a second minimum on the high-factor plants (Test 1)."""
    from scipy.optimize import least_squares
    fit = None
    for s0 in STARTS:
        start = np.zeros(len(fit_set))
        if "beta_pl" in fit_set:
            start[fit_set.index("beta_pl")] = s0
        f = least_squares(residuals, start, bounds=fit_bounds(fit_set), diff_step=0.02, max_nfev=100)
        fit = f if fit is None or f.cost < fit.cost else fit
    return fit


def fit_uncertainty(fit, n: int) -> tuple:
    """Standard errors, worst correlation and Jacobian condition number where the fit stopped, from
    (J'J)^-1 of the solver's Jacobian (prior rows included); inf when it's singular."""
    try:
        JtJ = fit.jac.T @ fit.jac
        cov = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        return np.full(n, np.inf), 1.0, np.inf
    sdf = np.sqrt(np.diag(cov))
    worst = float(np.max(np.abs((cov / np.outer(sdf, sdf))[np.triu_indices(n, 1)]))) if n > 1 else 0.0
    ev = np.linalg.eigvalsh(JtJ)
    return sdf, worst, float(np.sqrt(ev.max() / max(ev.min(), 1e-300)))


def check_row(seed: int, case: PlantCase, fit_set: tuple, fit) -> dict:
    """The fitted and true numbers for one plant, the fit's cost and uncertainty."""
    model = build_circuit(case.rc)
    th = dict(zip(fit_set, fit.x))
    b0 = separation_factors(model)
    b_true = np.log(separation_factors(case.plant) / b0)
    lo, hi = fit_bounds(fit_set)
    sdf, worst, jac_cond = fit_uncertainty(fit, len(fit_set))
    pr = list(model.p.names).index("Pr")
    return {"seed": seed, "set": "+".join(fit_set) + ("_map" if MAP else "") + TAG,
            "beta_pl_true": b0[0] * np.exp(b_true[0]), "beta_pl_fit": b0[0] * np.exp(th.get("beta_pl", 0.0)),
            "beta_np_true": b0[1] * np.exp(b_true[1]), "beta_np_fit": b0[1] * np.exp(th.get("beta_np", 0.0)),
            "others": " ".join(f"{c}={v:+.3f}" for c, v in th.items() if not c.startswith("beta")),
            "common_K_true": float(np.log(case.plant.p.K[pr] / model.p.K[pr])),
            "dn_acid_true": float(case.plant.p.n_acid - model.p.n_acid),
            "common_K_fit": float(th.get("common_K", 0.0)), "dn_acid_fit": float(th.get("dn_acid", 0.0)),
            "on_bound": " ".join(c for c, v, a, b in zip(fit_set, fit.x, lo, hi) if min(v - a, b - v) < 1e-3),
            "cost_fit": float(fit.cost), "sd": " ".join(f"{c}={v:.4f}" for c, v in zip(fit_set, sdf)),
            "worst_corr": worst, "jac_cond": jac_cond}


def check():
    """check_one for every plant and every set given with --sets; written row by row."""
    sets = [tuple(a.split(",")) for a in sys.argv[sys.argv.index("--sets") + 1].split(";")]
    out = HERE / "runs" / f"fair_fit_check_{CONFIG}{TAG}{'_map' if MAP else ''}.csv"
    rows = []
    for _, r in run_jobs(check_one, [(seed, s) for s in sets for seed in SEEDS], WORKERS):
        rows.append(r)
        print(f"{r['set']:28s} seed {r['seed']:2d}: Pr/LaCe true {r['beta_pl_true']:.3f} fit {r['beta_pl_fit']:.3f} | "
              f"Nd/Pr true {r['beta_np_true']:.3f} fit {r['beta_np_fit']:.3f} | {r['others']} | {r['evals']} evals "
              f"{r['fit_s']} s | purity end {100 * r['purity_end']:.2f}, recovery end {100 * r['recovery_end']:.2f}", flush=True)
        write_rows(out, rows, first=("seed", "set"))
    print("wrote", out.name)


if __name__ == "__main__":
    {"test1": test1, "identify": identify, "gains": gains, "loop": loop, "check": check}[
        [a for a in sys.argv[1:] if a in ("test1", "identify", "gains", "loop", "check")][0]]()
