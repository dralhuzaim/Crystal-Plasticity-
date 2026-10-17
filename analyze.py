"""
Build the dataset, tables and figures of the manuscript from results/sims/ (run run_study.py first).

    python analyze.py

Outputs
  results/dataset/simulations.csv     one row per simulation (all metrics), the archived dataset
  results/dataset/tension_curves.csv  calibration curves (copied from calibrate.py output)
  results/tables/*.csv                Tables 3, 5-9 of the manuscript
  results/figures/*.png               Figures 1-12 of the manuscript
  results/numbers.json                every number quoted in the manuscript text
"""
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from cpfe_lattice.config import get_params, STUDY, PROVENANCE
from cpfe_lattice import pipeline, fip

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SIMS, DS, TB, FG = (os.path.join(RES, d) for d in ("sims", "dataset", "tables", "figures"))
for d in (DS, TB, FG):
    os.makedirs(d, exist_ok=True)
P = get_params()
B = P["b"]
NEXT = P["n_extreme_per_realisation"]
COL = {"taylor": "#1f4e9c", "sachs": "#c0392b"}
LAB = {"taylor": "Iso-strain (Taylor) bound", "sachs": "Iso-stress (Sachs) bound"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
KEY = re.compile(r"(?P<bound>taylor|sachs)_s(?P<seed>\d+)_th(?P<theta>\d+)_kt(?P<kt>[\d.]+)_gr(?P<gr>[\d.]+)(?P<tag>.*)\.npz")


def load():
    rows, data = [], {}
    for f in sorted(glob.glob(os.path.join(SIMS, "*.npz"))):
        m = KEY.match(os.path.basename(f))
        d = np.load(f)
        r = {"key": os.path.basename(f)[:-4], "bound": m["bound"], "seed": int(m["seed"]), "theta_deg": int(m["theta"]),
             "Kt": float(m["kt"]), "gs_ratio": float(m["gr"]), "tag": m["tag"].lstrip("_"),
             "eps_hotspot": float(m["kt"]) * P["eps_nominal"]}
        g = fip.fit_gumbel(d["grain_maxima"][:NEXT])
        r.update({"gumbel_mu": g["mu"], "gumbel_beta": g["beta"], "ad_pvalue": g["ad_pvalue"],
                  "Ac_pct": float(d["Ac_pct"]), "f_high": float(d["f_high"]), "enrichment": float(d["enrichment"]),
                  "chi_crit": float(d["chi_crit"]), "chi_mean_top5": float(d["chi_mean_top5"]),
                  "mean_fip": float(d["mean_fip"]),
                  "stabilisation_pct": 100 * float(d["mean_fip"] / d["mean_fip_prev_cycle"] - 1),
                  "stress_amp_MPa": float(d["stress_amp_mean"])})
        rows.append(r); data[r["key"]] = d
    return pd.DataFrame(rows), data


def pair_eta(df, data, sel, b=B):
    """Per-seed eta and pooled eta for segregated (gs_ratio > 1) vs segregation-free runs matching sel."""
    out = []
    for bound in P["bounds"]:
        q = df[(df.bound == bound) & sel(df)]
        for (th, kt, gr), g in q[q.gs_ratio > 1.0].groupby(["theta_deg", "Kt", "gs_ratio"]):
            per, es, eh = [], [], []
            for _, r in g.iterrows():
                h = q[(q.gs_ratio == 1.0) & (q.seed == r.seed) & (q.theta_deg == th) & (q.Kt == kt)]
                if h.empty:
                    continue
                h = h.iloc[0]
                per.append(1 - pipeline.life_ratio(r.gumbel_mu, h.gumbel_mu, b))
                es.append(data[r.key]["grain_maxima"][:NEXT]); eh.append(data[h.key]["grain_maxima"][:NEXT])
            if not per:
                continue
            gs_, gh_ = fip.fit_gumbel(np.concatenate(es)), fip.fit_gumbel(np.concatenate(eh))
            emp = np.array([1 - pipeline.life_ratio(np.mean(a), np.mean(c), b) for a, c in zip(es, eh)])
            per = np.array(per)
            n = len(per)
            ci = stats.t.ppf(0.975, n - 1) * per.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            out.append({"bound": bound, "theta_deg": th, "Kt": kt, "gs_ratio": gr, "n": n,
                        "eta_mean_pct": 100 * per.mean(), "eta_sd_pct": 100 * per.std(ddof=1) if n > 1 else np.nan,
                        "eta_ci95_pct": 100 * ci,
                        "eta_pooled_pct": 100 * (1 - pipeline.life_ratio(gs_["mu"], gh_["mu"], b)),
                        "life_ratio_pooled": pipeline.life_ratio(gs_["mu"], gh_["mu"], b),
                        "p_value_eta0": float(stats.ttest_1samp(per, 0.0).pvalue) if n > 1 else np.nan,
                        "Ac_pct": g.Ac_pct.mean(), "f_high_pct": 100 * g.f_high.mean(), "enrichment": g.enrichment.mean(),
                        "chi_mean_top5": g.chi_mean_top5.mean(),
                        "TN": fip.scatter_band(gs_["mu"], gs_["beta"], b), "ad_pvalue": gs_["ad_pvalue"],
                        "gumbel_mu_seg": gs_["mu"], "gumbel_beta_seg": gs_["beta"], "gumbel_mu_hom": gh_["mu"],
                        "eta_empirical_mean_pct": 100 * emp.mean(),
                        "eta_empirical_ci95_pct": 100 * stats.t.ppf(0.975, n - 1) * emp.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan})
    return pd.DataFrame(out)


def main():
    df, data = load()
    df.to_csv(os.path.join(DS, "simulations.csv"), index=False)
    if os.path.exists(os.path.join(RES, "tension_curves.csv")):
        shutil.copy(os.path.join(RES, "tension_curves.csv"), os.path.join(DS, "tension_curves.csv"))
    main_ = lambda d: d.tag == ""
    N = {"n_simulations": int(len(df))}

    # ---------------- Table 5: build orientation at Kt = 2 -------------------------------------
    T5 = pair_eta(df, data, lambda d: main_(d) & (d.Kt == P["kt_reference"]) & (d.gs_ratio.isin([1.0, P["gs_ratio"]])))
    T5.to_csv(os.path.join(TB, "table5_orientation.csv"), index=False)
    # ---------------- heterogeneity-only effect: segregated vs mean-strength-matched homogeneous ----
    het = []
    for bound in P["bounds"]:
        for th in P["build_orientations_deg"]:
            per = []
            gm_ = []
            for s in P["seeds"]:
                ks = pipeline.sim_key(bound, s, th, P["kt_reference"], P["gs_ratio"])
                km = pipeline.sim_key(bound, s, th, P["kt_reference"], P["gs_ratio"], "_meanref")
                if ks in data and km in data:
                    ms = fip.fit_gumbel(data[ks]["grain_maxima"][:NEXT])["mu"]; mm = fip.fit_gumbel(data[km]["grain_maxima"][:NEXT])["mu"]
                    per.append(1 - pipeline.life_ratio(ms, mm, B))
                    if "g_mean" in data[km].files: gm_.append(float(data[km]["g_mean"]))
            if len(per) > 1:
                per = np.array(per); n = len(per)
                het.append({"bound": bound, "theta_deg": th, "n": n, "eta_het_mean_pct": 100 * per.mean(),
                            "eta_het_ci95_pct": 100 * stats.t.ppf(0.975, n - 1) * per.std(ddof=1) / np.sqrt(n),
                            "p_value": float(stats.ttest_1samp(per, 0.0).pvalue),
                            "g_mean_over_g0": (np.mean(gm_) / P["g0"]) if gm_ else np.nan})
    TH = pd.DataFrame(het); TH.to_csv(os.path.join(TB, "table5b_heterogeneity.csv"), index=False)
    # ---------------- Table 6: Kt sweep at theta = 0 --------------------------------------------
    T6 = pair_eta(df, data, lambda d: main_(d) & (d.theta_deg == 0) & (d.gs_ratio.isin([1.0, P["gs_ratio"]])))
    T6.to_csv(os.path.join(TB, "table6_kt.csv"), index=False)
    # ---------------- Table 7: slip-resistance contrast at theta = 0, Kt = 2 --------------------
    T7 = pair_eta(df, data, lambda d: main_(d) & (d.theta_deg == 0) & (d.Kt == P["kt_reference"]))
    T7.to_csv(os.path.join(TB, "table7_contrast.csv"), index=False)
    # ---------------- b sweep (post-processing) -------------------------------------------------
    rows = []
    for b in P["b_sweep"]:
        t = pair_eta(df, data, lambda d: main_(d) & (d.theta_deg == 0) & (d.Kt == P["kt_reference"]) & d.gs_ratio.isin([1.0, P["gs_ratio"]]), b=b)
        for _, r in t.iterrows():
            rows.append({"b": b, "bound": r.bound, "eta_mean_pct": r.eta_mean_pct, "eta_ci95_pct": r.eta_ci95_pct})
    Tb = pd.DataFrame(rows); Tb.to_csv(os.path.join(TB, "table_b_sweep.csv"), index=False)

    # ---------------- Table 8: convergence ------------------------------------------------------
    conv = []
    for bound in P["bounds"]:
        ref = None
        for tag, lab in [("", "reference: 216 grains, 20^3, 12 steps/quarter, 6 cycles")] + \
                [(t, t) for t in sorted(df[df.tag.str.startswith("conv")].tag.unique())]:
            q = df[(df.bound == bound) & (df.seed == 0) & (df.theta_deg == 0) & (df.Kt == P["kt_reference"]) & (df.tag == tag)]
            s = q[q.gs_ratio == P["gs_ratio"]]; h = q[q.gs_ratio == 1.0]
            if s.empty or h.empty:
                continue
            eta = 100 * (1 - pipeline.life_ratio(s.iloc[0].gumbel_mu, h.iloc[0].gumbel_mu, B))
            conv.append({"bound": bound, "configuration": lab, "P_ext_seg": s.iloc[0].gumbel_mu, "P_ext_hom": h.iloc[0].gumbel_mu,
                         "eta_pct": eta, "stabilisation_pct": s.iloc[0].stabilisation_pct})
    T8 = pd.DataFrame(conv)
    for bound in P["bounds"]:
        m = T8.bound == bound
        if m.any():
            r0 = T8[m & T8.configuration.str.startswith("reference")].P_ext_seg.values[0]
            T8.loc[m, "P_ext_change_pct"] = 100 * (T8.loc[m, "P_ext_seg"] / r0 - 1)
    T8.to_csv(os.path.join(TB, "table8_convergence.csv"), index=False)

    # ---------------- Table 3: one-at-a-time sensitivity on eta ----------------------------------
    oat = []
    for bound in P["bounds"]:
        base = T5[(T5.bound == bound) & (T5.theta_deg == 0)]
        q0 = df[(df.bound == bound) & main_(df) & (df.theta_deg == 0) & (df.Kt == P["kt_reference"])]
        for tag in sorted(df[df.tag.str.startswith("oat")].tag.unique()):
            q = df[(df.bound == bound) & (df.tag == tag)]
            parts = tag.split("_")
            name, f = "_".join(parts[1:-1]), float(parts[-1])
            etas = []
            for s in sorted(q.seed.unique()):
                seg = q[(q.seed == s) & (q.gs_ratio > 1.0)]
                hom = q[(q.seed == s) & (q.gs_ratio == 1.0)]
                if hom.empty:                                   # gs_ratio perturbation reuses the baseline hom run
                    hom = q0[(q0.seed == s) & (q0.gs_ratio == 1.0)]
                if seg.empty or hom.empty:
                    continue
                etas.append(100 * (1 - pipeline.life_ratio(seg.iloc[0].gumbel_mu, hom.iloc[0].gumbel_mu, B)))
            seeds = sorted(q.seed.unique())
            ref = []
            for s in seeds:
                seg = q0[(q0.seed == s) & (q0.gs_ratio == P["gs_ratio"])]; hom = q0[(q0.seed == s) & (q0.gs_ratio == 1.0)]
                ref.append(100 * (1 - pipeline.life_ratio(seg.iloc[0].gumbel_mu, hom.iloc[0].gumbel_mu, B)))
            if etas:
                oat.append({"bound": bound, "parameter": name, "factor": f, "eta_pct": np.mean(etas),
                            "eta_ref_same_seeds_pct": np.mean(ref), "delta_eta_pct_points": np.mean(etas) - np.mean(ref)})
    for b in (P["b"] * 0.8, P["b"] * 1.2):
        t = pair_eta(df, data, lambda d: main_(d) & (d.theta_deg == 0) & (d.Kt == P["kt_reference"]) & d.gs_ratio.isin([1.0, P["gs_ratio"]]), b=b)
        for _, r in t.iterrows():
            base = T5[(T5.bound == r.bound) & (T5.theta_deg == 0)].eta_mean_pct.values[0]
            oat.append({"bound": r.bound, "parameter": "b", "factor": b / P["b"], "eta_pct": r.eta_mean_pct,
                        "eta_ref_same_seeds_pct": base, "delta_eta_pct_points": r.eta_mean_pct - base})
    T3 = pd.DataFrame(oat); T3.to_csv(os.path.join(TB, "table3_oat.csv"), index=False)

    # ---------------- Table 9: Monte Carlo ------------------------------------------------------
    mc = []
    for bound in P["bounds"]:
        q = df[(df.bound == bound) & df.tag.str.startswith("mc")]
        for tag, g in q.groupby("tag"):
            s, h = g[g.gs_ratio > 1.0], g[g.gs_ratio == 1.0]
            if s.empty or h.empty:
                continue
            mc.append({"bound": bound, "sample": tag, "eta_pct": 100 * (1 - pipeline.life_ratio(s.iloc[0].gumbel_mu, h.iloc[0].gumbel_mu, B))})
    MC = pd.DataFrame(mc); MC.to_csv(os.path.join(DS, "monte_carlo_samples.csv"), index=False)
    T9 = MC.groupby("bound").eta_pct.agg(n="count", mean="mean", sd="std", p5=lambda x: np.percentile(x, 5),
                                         p95=lambda x: np.percentile(x, 95), frac_negative=lambda x: np.mean(x < 0)).reset_index() if len(MC) else pd.DataFrame()
    T9.to_csv(os.path.join(TB, "table9_monte_carlo.csv"), index=False)

    # ---------------- provenance table ----------------------------------------------------------
    pd.DataFrame([{"parameter": k, "value": STUDY[k], "status": v[0], "source": v[1]} for k, v in PROVENANCE.items()]) \
        .to_csv(os.path.join(TB, "table4_parameters.csv"), index=False)

    # ---------------- numbers quoted in the text ------------------------------------------------
    cal = json.load(open(os.path.join(RES, "calibration.json")))
    N["calibration"] = cal
    for bound in P["bounds"]:
        t = T5[T5.bound == bound]
        N[bound] = {"eta_orientation_pct": {int(r.theta_deg): [round(r.eta_mean_pct, 2), round(r.eta_ci95_pct, 2), r.p_value_eta0] for _, r in t.iterrows()},
                    "eta_kt_pct": {float(r.Kt): [round(r.eta_mean_pct, 2), round(r.eta_ci95_pct, 2)] for _, r in T6[T6.bound == bound].iterrows()},
                    "eta_contrast_pct": {float(r.gs_ratio): [round(r.eta_mean_pct, 2), round(r.eta_ci95_pct, 2)] for _, r in T7[T7.bound == bound].iterrows()},
                    "Ac_pct": [round(t.Ac_pct.min(), 1), round(t.Ac_pct.max(), 1)],
                    "enrichment": [round(t.enrichment.min(), 2), round(t.enrichment.max(), 2)],
                    "chi_mean_top5": [round(t.chi_mean_top5.min(), 3), round(t.chi_mean_top5.max(), 3)],
                    "TN": [round(t.TN.min(), 2), round(t.TN.max(), 2)],
                    "ad_pvalue_min": round(t.ad_pvalue.min(), 3),
                    "ad_reject_count": int((pd.concat([T5[T5.bound == bound], T6[T6.bound == bound], T7[T7.bound == bound]]).drop_duplicates(["theta_deg", "Kt", "gs_ratio"]).ad_pvalue < 0.05).sum()),
                    "ad_case_count": int(len(pd.concat([T5[T5.bound == bound], T6[T6.bound == bound], T7[T7.bound == bound]]).drop_duplicates(["theta_deg", "Kt", "gs_ratio"]))),
                    "eta_empirical_orientation_pct": {int(r.theta_deg): [round(r.eta_empirical_mean_pct, 2), round(r.eta_empirical_ci95_pct, 2)] for _, r in t.iterrows()},
                    "eta_empirical_kt_pct": {float(r.Kt): [round(r.eta_empirical_mean_pct, 2), round(r.eta_empirical_ci95_pct, 2)] for _, r in T6[T6.bound == bound].iterrows()},
                    "enrichment_all": [round(pd.concat([T5, T6, T7])[lambda x: x.bound == bound].enrichment.min(), 2), round(pd.concat([T5, T6, T7])[lambda x: x.bound == bound].enrichment.max(), 2)],
                    "stabilisation_pct_max": round(df[(df.bound == bound) & main_(df)].stabilisation_pct.abs().max(), 2),
                    "stress_amp_MPa_Kt2": round(df[(df.bound == bound) & main_(df) & (df.Kt == 2.0) & (df.theta_deg == 0) & (df.gs_ratio == 1.0)].stress_amp_MPa.mean(), 0)}
    N["f_high_pct"] = round(100 * df.f_high.mean(), 2)
    chis = [pipeline.build_rve(P, s)[2] for s in P["seeds"]]
    N["chi_mean_volume"] = round(float(np.mean([c.mean() for c in chis])), 3)
    N["chi_p10_volume"] = round(float(np.mean([np.percentile(c, 10) for c in chis])), 3)
    for bound in P["bounds"]:
        if TH.empty:
            continue
        t = TH[TH.bound == bound]
        N[bound]["eta_het_orientation_pct"] = {int(r.theta_deg): [round(r.eta_het_mean_pct, 2), round(r.eta_het_ci95_pct, 2), r.p_value] for _, r in t.iterrows()}
        N[bound]["g_mean_over_g0"] = round(float(t.g_mean_over_g0.mean()), 4) if len(t) else None
    json.dump(N, open(os.path.join(RES, "numbers.json"), "w"), indent=2, default=float)

    figures(df, data, T5, T6, T7, T8, MC, cal, TH)
    print(json.dumps(N, indent=1, default=float)[:4000])


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FG, name), dpi=300); plt.close(fig)


def figures(df, data, T5, T6, T7, T8, MC, cal, TH):
    # Fig 1: workflow
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = ["Process scenario\npulsed-laser DED IN718\ncolumnar grains", "RVE generation (Module A)\nVoronoi + <001> fibre\n216 grains, 20³ points",
             "Segregation field\nScheil-type χ(x)\nEqs. (6)–(7)", "Cyclic crystal plasticity (Module B)\niso-strain and iso-stress\nEqs. (1)–(5), (8)",
             "Band-averaged FIP\nGumbel extremes\nEqs. (10)–(12)", "Hotspot strain Kt·εn, life ratio η\nvs segregation-free and\nmean-strength references"]
    pos = [(0.03, 0.56), (0.36, 0.56), (0.69, 0.56), (0.69, 0.12), (0.36, 0.12), (0.03, 0.12)]
    for (x, y), t in zip(pos, boxes):
        ax.add_patch(FancyBboxPatch((x, y), 0.28, 0.3, boxstyle="round,pad=0.01", fc="#eef2f8", ec="#1f4e9c", lw=1))
        ax.text(x + 0.14, y + 0.15, t, ha="center", va="center", fontsize=7.5)
    arr = dict(arrowstyle="->", lw=1.2)
    ax.annotate("", xy=(0.36, 0.71), xytext=(0.315, 0.71), arrowprops=arr)
    ax.annotate("", xy=(0.69, 0.71), xytext=(0.645, 0.71), arrowprops=arr)
    ax.annotate("", xy=(0.83, 0.43), xytext=(0.83, 0.555), arrowprops=arr)
    ax.annotate("", xy=(0.645, 0.27), xytext=(0.69, 0.27), arrowprops=arr)
    ax.annotate("", xy=(0.315, 0.27), xytext=(0.36, 0.27), arrowprops=arr)
    ax.text(0.5, 0.02, "Every quantity is computed by the open-source package.", ha="center", fontsize=7, style="italic")
    save(fig, "fig01_workflow.png")

    # Fig 2: computed RVE and segregation field (seed 0)
    gm, eul, chi, vox = pipeline.build_rve(P, 0)
    rng = np.random.default_rng(1); colors = rng.random((gm.max() + 1, 3)) * 0.6 + 0.3
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    k = gm.shape[0] // 2
    axs[0].imshow(colors[gm[:, k, :]].transpose(1, 0, 2), origin="lower", extent=[0, P["rve_edge_um"]] * 2)
    axs[0].set_title("Grains (x–z section)", fontsize=8); axs[0].set_xlabel("x (µm)"); axs[0].set_ylabel("z = build direction (µm)")
    im = axs[1].imshow(chi[:, k, :].T, origin="lower", cmap="magma", vmin=0, vmax=P["chi_max"], extent=[0, P["rve_edge_um"]] * 2)
    axs[1].set_title("Segregation indicator χ", fontsize=8); axs[1].set_xlabel("x (µm)"); plt.colorbar(im, ax=axs[1], fraction=0.046)
    axs[2].hist(chi.ravel(), bins=40, color="#7a3b8f"); axs[2].axvline(P["chi_high"], color="k", ls="--", lw=0.8)
    axs[2].set_yscale("log"); axs[2].set_xlabel("χ"); axs[2].set_ylabel("Material points")
    axs[2].set_title("χ > %.2f: %.1f %% of volume" % (P["chi_high"], 100 * np.mean(chi > P["chi_high"])), fontsize=8)
    save(fig, "fig02_rve_segregation.png")

    # Fig 3: bounds schematic
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.4))
    for ax, (t, txt) in zip(axs, [("Iso-strain (Taylor)", "every point: same strain\nhard χ-rich point carries more stress\nslips less"),
                                  ("Iso-stress (Sachs)", "every point: same stress\nsoft matrix point strains more\nhard point strains less")]):
        ax.axis("off"); ax.set_title(t, fontsize=9)
        for j, (c, lab) in enumerate([("#e7d3ef", "matrix\nχ≈0"), ("#7a3b8f", "interdendritic\nχ≈0.85")]):
            ax.add_patch(FancyBboxPatch((0.15 + 0.4 * j, 0.35), 0.28, 0.4, boxstyle="round,pad=0.01", fc=c, ec="k"))
            ax.text(0.29 + 0.4 * j, 0.55, lab, ha="center", va="center", fontsize=7, color="k" if j == 0 else "w")
        ax.text(0.5, 0.12, txt, ha="center", va="center", fontsize=7)
    save(fig, "fig03_bounds.png")

    # Fig 4: spatial map of top-5 % FIP sites vs chi (theta 0, Kt 2, seed 0)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, bound in zip(axs, P["bounds"]):
        r = pipeline.simulate(P, bound, 0, 0, P["kt_reference"], P["gs_ratio"], keep_fields=True)
        pb, ch = r["p_bar"], r["chi"]
        thr = np.quantile(pb, 1 - P["top_fraction"])
        ax.imshow(ch[:, k, :].T, origin="lower", cmap="Greys", vmin=0, vmax=P["chi_max"], extent=[0, P["rve_edge_um"]] * 2)
        yy, xx = np.nonzero((pb[:, k, :] >= thr).T)
        ax.scatter((xx + 0.5) * vox, (yy + 0.5) * vox, s=9, c=COL[bound], marker="s", label="top-5 % P̄FIP")
        ax.set_title(LAB[bound] + "\nAc = %.1f %%" % r["Ac_pct"], fontsize=8); ax.set_xlabel("x (µm)")
        ax.legend(fontsize=6, loc="upper right")
    axs[0].set_ylabel("z (µm)")
    save(fig, "fig04_crack_sites.png")

    # Fig 5: eta vs theta
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for bound in P["bounds"]:
        t = T5[T5.bound == bound].sort_values("theta_deg")
        ax.errorbar(t.theta_deg - 2, t.eta_mean_pct, yerr=t.eta_ci95_pct, marker="o", capsize=3, color=COL[bound],
                    label=LAB[bound].split(" (")[0] + ": vs χ = 0 reference")
        if len(TH):
            h = TH[TH.bound == bound].sort_values("theta_deg")
            ax.errorbar(h.theta_deg + 2, h.eta_het_mean_pct, yerr=h.eta_het_ci95_pct, marker="D", ls="--", mfc="white", capsize=3,
                        color=COL[bound], label=LAB[bound].split(" (")[0] + ": vs mean-strength reference")
    ax.axhline(0, color="0.5", lw=0.7); ax.set_xticks(P["build_orientations_deg"])
    ax.set_xlabel("Build orientation θ (°)"); ax.set_ylabel("Life reduction η (%)"); ax.legend(fontsize=6, frameon=False)
    save(fig, "fig05_eta_orientation.png")

    # Fig 6: eta vs Kt
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for bound in P["bounds"]:
        t = T6[T6.bound == bound].sort_values("Kt")
        ax.errorbar(t.Kt, t.eta_mean_pct, yerr=t.eta_ci95_pct, marker="s", capsize=3, color=COL[bound], label=LAB[bound])
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xlabel("Kt  (hotspot strain εh = Kt × 0.4 %)"); ax.set_ylabel("Life reduction η (%)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig06_eta_kt.png")

    # Fig 7: eta vs gs/g0
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for bound in P["bounds"]:
        t = T7[T7.bound == bound].sort_values("gs_ratio")
        ax.errorbar(t.gs_ratio, t.eta_mean_pct, yerr=t.eta_ci95_pct, marker="^", capsize=3, color=COL[bound], label=LAB[bound])
    ax.axhline(0, color="0.5", lw=0.7); ax.set_xlabel("Slip-resistance contrast gs/g0"); ax.set_ylabel("Life reduction η (%)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig07_eta_contrast.png")

    # Fig 8: hysteresis
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, bound in zip(axs, P["bounds"]):
        d = data[pipeline.sim_key(bound, 0, 0, P["kt_reference"], P["gs_ratio"])]
        ax.plot(100 * d["loop_mean"][:, 0], d["loop_mean"][:, 1], color="k", lw=1, label="RVE average")
        ax.plot(100 * d["loop_hi"][:, 0], d["loop_hi"][:, 1], color="#7a3b8f", lw=1, ls="--", label="max-χ point (χ = %.2f)" % d["chi_hi"])
        ax.plot(100 * d["loop_lo"][:, 0], d["loop_lo"][:, 1], color="#e08e0b", lw=1, ls=":", label="min-χ point (χ = %.2f)" % d["chi_lo"])
        ax.set_title(LAB[bound], fontsize=8); ax.set_xlabel("Macroscopic axial strain (%)")
        ax.legend(fontsize=6, frameon=False)
    axs[0].set_ylabel("Axial stress (MPa)")
    save(fig, "fig08_hysteresis.png")

    # Fig 9: Gumbel fits
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, bound in zip(axs, P["bounds"]):
        for gr, ls in ((1.0, ":"), (P["gs_ratio"], "-")):
            x = np.concatenate([data[pipeline.sim_key(bound, s, 0, P["kt_reference"], gr)]["grain_maxima"][:NEXT] for s in P["seeds"]])
            g = fip.fit_gumbel(x)
            ax.hist(x, bins=25, density=True, alpha=0.3, color=COL[bound] if gr > 1 else "0.5")
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, stats.gumbel_r.pdf(xs, g["mu"], g["beta"]), ls, color=COL[bound] if gr > 1 else "0.3",
                    label="%s: μ = %.4f, β = %.4f" % ("gs/g0 = %.2f" % gr if gr > 1 else "segregation-free", g["mu"], g["beta"]))
        ax.set_title(LAB[bound], fontsize=8); ax.set_xlabel("Grain-maximum band-averaged FIP"); ax.legend(fontsize=6, frameon=False)
    axs[0].set_ylabel("Probability density")
    save(fig, "fig09_gumbel.png")

    # Fig 10: chi at top-5 % sites vs volume
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    t = T5[T5.theta_deg == 0]
    xs = np.arange(len(t))
    ax.bar(xs - 0.2, t.f_high_pct, 0.4, color="0.6", label="χ > 0.65 volume fraction (%)")
    ax.bar(xs + 0.2, t.Ac_pct, 0.4, color=[COL[b] for b in t.bound], label="Ac: top-5 % sites with χ > 0.65 (%)")
    ax.set_xticks(xs); ax.set_xticklabels([LAB[b].split(" (")[0] for b in t.bound], fontsize=7)
    ax.set_ylabel("%"); ax.set_ylim(0, 1.45 * max(t.Ac_pct.max(), t.f_high_pct.max())); ax.legend(fontsize=6, frameon=False, loc="upper center")
    save(fig, "fig10_colocation.png")

    # Fig 11: convergence
    if len(T8):
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        for bound in P["bounds"]:
            t = T8[T8.bound == bound].reset_index(drop=True)
            ax.plot(range(len(t)), t.P_ext_change_pct, "o-", color=COL[bound], label=LAB[bound])
        ax.set_xticks(range(len(t))); ax.set_xticklabels(["ref" if c.startswith("reference") else c.replace("conv_", "") for c in t.configuration], fontsize=6, rotation=45)
        ax.set_xlabel("Configuration (n = grains, g = points per edge, q = increments per quarter cycle, c = cycles)", fontsize=6)
        ax.axhline(0, color="0.5", lw=0.7); ax.set_ylabel("Change in P_ext vs reference (%)"); ax.legend(fontsize=6, frameon=False)
        save(fig, "fig11_convergence.png")

    # Fig 12: calibration
    tc = pd.read_csv(os.path.join(RES, "tension_curves.csv"))
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for bound, g in tc.groupby("bound"):
        ax.plot(100 * g.strain, g.stress_MPa * np.exp(-g.strain), color=COL[bound], label=LAB[bound] + " (engineering)")
    ax.axhline(cal["target"]["YS_MPa"], color="k", ls=":", lw=0.8); ax.axhline(cal["target"]["UTS_MPa"], color="k", ls="--", lw=0.8)
    ax.text(1, cal["target"]["UTS_MPa"] + 10, "UTS, Song et al.", fontsize=6); ax.text(15, cal["target"]["YS_MPa"] + 10, "YS, Song et al.", fontsize=6)
    ax.set_xlabel("Strain (%)"); ax.set_ylabel("Stress (MPa)"); ax.legend(fontsize=6, frameon=False, loc="lower right")
    save(fig, "fig12_calibration.png")


if __name__ == "__main__":
    main()
