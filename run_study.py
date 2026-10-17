"""
Run every simulation of the study (Table 4) and cache each one in results/sims/.

    python run_study.py --study all --jobs 2
    python run_study.py --study orientation        # or kt | contrast | convergence | oat | mc

Already-computed simulations are loaded from results/sims/ rather than re-run. Then run
analyze.py to build the tables, figures and dataset files.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from cpfe_lattice.config import get_params
from cpfe_lattice import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.join(HERE, "results", "sims")


def tasks(study):
    p = get_params()
    T = []
    B, S, KT0 = p["bounds"], p["seeds"], p["kt_reference"]
    if study in ("orientation", "all"):
        T += [(b, s, th, KT0, gr, {}, "") for b in B for s in S for th in p["build_orientations_deg"] for gr in (1.0, p["gs_ratio"])]
    if study in ("meanref", "all"):
        T += [(b, s, th, KT0, p["gs_ratio"], {"__mean_ref": True}, "_meanref") for b in B for s in S for th in p["build_orientations_deg"]]
    if study in ("kt", "all"):
        T += [(b, s, 0, kt, gr, {}, "") for b in B for s in S for kt in p["kt_sweep"] for gr in (1.0, p["gs_ratio"])]
    if study in ("contrast", "all"):
        T += [(b, s, 0, KT0, gr, {}, "") for b in B for s in S for gr in p["gs_ratio_sweep"]]
    if study in ("convergence", "all"):
        conf = [(64, 13), (125, 17), (216, 20), (343, 23), (216, 16), (216, 24)]
        for b in B:
            for ng, g in conf:
                for gr in (1.0, p["gs_ratio"]):
                    T.append((b, 0, 0, KT0, gr, {"n_grains": ng, "grid": g}, "_conv_n%d_g%d" % (ng, g)))
            for nq in (8, 24):
                for gr in (1.0, p["gs_ratio"]):
                    T.append((b, 0, 0, KT0, gr, {"steps_per_quarter_cycle": nq}, "_conv_q%d" % nq))
            for nc in (4, 12):
                for gr in (1.0, p["gs_ratio"]):
                    T.append((b, 0, 0, KT0, gr, {"n_cycles": nc}, "_conv_c%d" % nc))
    if study in ("oat", "all"):
        for b in B:
            for name in ("g0", "h0", "g_inf", "m", "k_fip", "gs_ratio"):
                for f in (0.8, 1.2):
                    v = p[name] * f
                    for s in (S[:2] if b == "sachs" else S[:1]):
                        for gr in (1.0, p["gs_ratio"]):
                            o = {name: v}
                            g2 = gr
                            if name == "gs_ratio":
                                if gr == 1.0:
                                    continue
                                g2 = v
                            T.append((b, s, 0, KT0, g2, o, "_oat_%s_%.1f" % (name, f)))
    if study in ("mc", "all"):
        rng = np.random.default_rng(p["mc_seed"])
        for i in range(40):
            z = rng.standard_normal(len(p["mc_parameters"]))
            while np.any(np.abs(z) > p["mc_truncation_sd"]):
                bad = np.abs(z) > p["mc_truncation_sd"]; z[bad] = rng.standard_normal(bad.sum())
            o = {nm: p[nm] * (1 + p["mc_rel_sd"] * zi) for nm, zi in zip(p["mc_parameters"], z)}
            for b in B:
                if b == "taylor" and i >= 8:           # Taylor ensemble limited to 8 samples (cost)
                    continue
                for gr in (1.0, o["gs_ratio"]):
                    T.append((b, 0, 0, KT0, gr, o, "_mc%03d" % i))
    # unique, Taylor (expensive) first, larger strains first
    seen, U = set(), []
    for t in T:
        k = pipeline.sim_key(t[0], t[1], t[2], t[3], t[4], t[6])
        if k not in seen:
            seen.add(k); U.append(t)
    U.sort(key=lambda t: (t[0] != "taylor", -t[3]))
    return U


def work(t):
    b, s, th, kt, gr, o, tag = t
    o = dict(o); mref = o.pop("__mean_ref", False)
    p = get_params(o)
    t0 = time.time()
    k = pipeline.sim_key(b, s, th, kt, gr, tag)
    done = os.path.exists(os.path.join(SIMS, k + ".npz"))
    pipeline.simulate(p, b, s, th, kt, gr, outdir=SIMS, tag=tag, mean_ref=mref)
    return k, done, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="all")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    T = tasks(a.study)
    print("%d simulations" % len(T), flush=True)
    os.makedirs(SIMS, exist_ok=True)
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, (k, done, dt) in enumerate(pool.imap_unordered(work, T), 1):
            print("[%4d/%d] %-48s %s %6.1fs  (elapsed %.0f min)" % (i, len(T), k, "cached" if done else "ran", dt,
                                                                  (time.time() - t0) / 60), flush=True)
