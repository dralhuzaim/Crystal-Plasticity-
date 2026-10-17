"""
Apply the paper's post-processing (Eqs. 10-13) to per-element fields from any solver, e.g. a
full-field DAMASK run of the same RVEs. Each realisation needs a segregated and a segregation-free
.npz file with: grain_map (g,g,g) int, chi (g,g,g), volume (g,g,g), dgamma_range (g^3,12),
sigma_n_max (g^3,12) and ws (g^3,), all over the last cycle.

    python run_from_fields.py --seg runs/seg_s*.npz --hom runs/hom_s*.npz
"""
import argparse
import glob

import numpy as np

from cpfe_lattice.config import get_params
from cpfe_lattice import fip, lattice, crack


def extremes(F, p):
    P, _ = fip.fatemi_socie_fip(F["dgamma_range"], F["sigma_n_max"], p["sigma_y_ref"], p["k_fip"])
    pb = fip.band_average(P, F["volume"], F["grain_map"], p["band_window_vox"])
    return pb, np.sort(fip.grain_maxima(pb, F["grain_map"]))[::-1][: p["n_extreme_per_realisation"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", nargs="+", required=True)
    ap.add_argument("--hom", nargs="+", required=True)
    a = ap.parse_args()
    p = get_params()
    seg = sorted(sum((glob.glob(x) for x in a.seg), [])); hom = sorted(sum((glob.glob(x) for x in a.hom), []))
    assert len(seg) == len(hom) > 0, "need matching segregated / segregation-free files"
    etas, ac = [], []
    for fs, fh in zip(seg, hom):
        Fs, Fh = fip.fields_from_npz(fs), fip.fields_from_npz(fh)
        pb, es = extremes(Fs, p); _, eh = extremes(Fh, p)
        etas.append(lattice.life_reduction(fip.fit_gumbel(es)["mu"], fip.fit_gumbel(eh)["mu"], p["b"]))
        ac.append(crack.colocation_metric(pb, Fs["chi"], p["top_fraction"], p["chi_high"]))
    e = 100 * np.array(etas)
    print("eta = %.2f %% (sd %.2f, n = %d)   Ac = %.1f %%" % (e.mean(), e.std(ddof=1) if len(e) > 1 else 0, len(e), np.mean(ac)))


if __name__ == "__main__":
    main()
