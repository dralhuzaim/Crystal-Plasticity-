"""
Identify the matrix slip resistance g0 and saturation g_inf from as-deposited DED IN718 tensile data.

Target (Song et al., Sci. Rep. 13 (2023) 19235, as-fabricated wire-arc DED IN718, room temperature):
  0.2 % offset yield strength 427.7 MPa, ultimate tensile strength 787.4 MPa.
Model: Taylor (iso-strain) polycrystal, 400 random orientations, uniaxial tension at 1e-3 s^-1.
The UTS is the maximum engineering stress sigma*exp(-eps) (Considere), with the small-strain
stress taken as true stress. h0, a and q_latent are held fixed (see config.STUDY).

    python calibrate.py            # writes results/calibration.json and results/tension_curves.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from cpfe_lattice.config import get_params
from cpfe_lattice import cp_solver

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TARGET_YS, TARGET_UTS = 427.7, 787.4


def random_orientations(n, seed=7):
    """Uniform (random-texture) Bunge Euler angles in degrees: phi1, phi2 ~ U(0, 360), cos(Phi) ~ U(-1, 1)."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0, 360, n), np.degrees(np.arccos(rng.uniform(-1, 1, n))), rng.uniform(0, 360, n)], 1)


def tension(p, eul, bound="taylor", eps_max=0.30, n=300):
    cls = cp_solver.TaylorCP if bound == "taylor" else cp_solver.SachsCP
    m = cls(eul, np.full(len(eul), p["g0"]), p, 0.0)
    de = eps_max / n; dt = de / p["strain_rate"]
    eps, sig = [0.0], [0.0]
    for e in np.linspace(de, eps_max, n):
        _, _, sl, _ = m.step(e, dt)
        eps.append(e); sig.append(float(sl[:, 2, 2].mean()))
    return np.array(eps), np.array(sig)


def ys_uts(eps, sig, E):
    off = sig - E * (eps - 0.002)
    i = np.flatnonzero(off <= 0)
    ys = float(np.interp(0.0, -off[i[0] - 1:i[0] + 1], sig[i[0] - 1:i[0] + 1])) if i.size and i[0] > 0 else float("nan")
    uts = float(np.max(sig * np.exp(-eps)))
    return ys, uts


def main():
    base = get_params()
    eul = random_orientations(400)

    def resid(x):
        p = dict(base, g0=x[0], g_inf=x[1])
        e, s = tension(p, eul, n=120)
        E = s[1] / e[1]
        ys, uts = ys_uts(e, s, E)
        return [(ys - TARGET_YS) / TARGET_YS, (uts - TARGET_UTS) / TARGET_UTS]

    sol = least_squares(resid, x0=[150.0, 400.0], bounds=([50, 150], [400, 2000]), diff_step=1e-3)
    g0, ginf = map(float, sol.x)
    p = dict(base, g0=g0, g_inf=ginf)
    rows, out = [], {"target": {"YS_MPa": TARGET_YS, "UTS_MPa": TARGET_UTS, "source": "Song et al. 2023, Sci. Rep. 13:19235"},
                     "fixed": {k: base[k] for k in ("h0", "a", "q_latent", "m", "gamma0", "C11", "C12", "C44")},
                     "identified": {"g0_MPa": g0, "g_inf_MPa": ginf}}
    for bound in ("taylor", "sachs"):
        e, s = tension(p, eul, bound)
        E = s[1] / e[1]
        ys, uts = ys_uts(e, s, E)
        out[bound] = {"E_GPa": E / 1e3, "YS_MPa": ys, "UTS_MPa": uts}
        rows += [{"bound": bound, "strain": a, "stress_MPa": b} for a, b in zip(e, s)]
    os.makedirs(RES, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "tension_curves.csv"), index=False)
    json.dump(out, open(os.path.join(RES, "calibration.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
