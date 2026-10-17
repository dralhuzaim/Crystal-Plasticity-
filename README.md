# CPFE-Lattice: iso-strain and iso-stress crystal-plasticity bounds on segregation effects in DED IN718

Code and data for the manuscript **"Iso-Strain and Iso-Stress Crystal-Plasticity Bounds on the Effect of Process-Induced
Interdendritic Segregation on Fatigue Crack-Initiation Driving Forces in Pulsed Laser Directed Energy Deposited IN718
Lattices"** (*Journal of Manufacturing and Materials Processing*, submission jmmp-4620806, revision 2).

Every number, table and figure in the manuscript is produced by the scripts in this repository from the simulations
archived in `results/sims/`.

## What the model is

- **RVE (Module A):** periodic anisotropic Voronoi tessellation (216 grains, 20³ material points, columnar aspect
  ratio 3, <001> fibre along the build direction), with a Scheil-type Nb field and segregation indicator χ (Eqs. 6–7).
- **Crystal plasticity (Module B):** 12 {111}<110> systems, power-law flow rule and saturating hardening (Eqs. 1–5).
  The equations are integrated at every material point under two mean-field bounds:
  - `TaylorCP`: iso-strain (Taylor). Every point has the macroscopic strain; the macroscopic stress is uniaxial.
  - `SachsCP`: iso-stress (Sachs). Every point carries the same uniaxial stress; the mean axial strain is prescribed.

  A full-field solution with equilibrium and compatibility lies between these bounds. **This package does not contain
  a full-field solver.** `run_from_fields.py` applies the same post-processing to per-element fields from one (e.g.
  DAMASK).
- **Post-processing (Module C):** Fatemi–Socie FIP on the critical slip plane (Eq. 10); volume-weighted band average
  within grains (Eq. 11); maximum-likelihood Gumbel fit of grain maxima with Anderson–Darling test (Eq. 12); life ratio
  η = 1 − (P_seg/P_0)^(1/b), in which the constant C cancels (Eq. 13); co-location metric Ac and enrichment ratio.
- **Strut transfer:** the hotspot strain amplitude ε_h = Kt·ε_n is imposed on the RVE (Eq. 9). Kt is a parameter
  (1–3), not the result of a unit-cell analysis.

## Parameters and provenance

All parameters are in `cpfe_lattice/config.py` (`STUDY`), with their status in `PROVENANCE`: literature, identified,
assumed or design. g0 and g∞ are identified by `calibrate.py` from the yield (427.7 MPa) and ultimate (787.4 MPa)
strengths of as-fabricated DED IN718 (Song et al., *Sci. Rep.* 13 (2023) 19235). Elastic constants and the rate
sensitivity are from Cruzado et al., *Int. J. Fatigue* (2018); a and q follow Kalidindi et al., *J. Mech. Phys. Solids*
(1992). Assumed values are examined in the sensitivity study.

## Reproduce

```bash
pip install -r requirements.txt
python tests/test_equations.py            # 10 equation-level tests
python calibrate.py                       # parameter identification -> results/calibration.json   (~15 min)
python run_study.py --study all --jobs 2  # 402 simulations -> results/sims/*.npz                  (several hours on 2 cores)
python analyze.py                         # tables, figures, dataset, numbers.json                  (~5 min)
```

`run_study.py` skips simulations already present in `results/sims/`, so the archived dataset reproduces the tables and
figures without re-running. Delete `results/sims/` to recompute from scratch. The computation is deterministic for a
given NumPy/SciPy version.

## Dataset (`results/`)

| Path | Content |
|---|---|
| `sims/<bound>_s<seed>_th<θ>_kt<Kt>_gr<gs/g0>[_tag].npz` | One simulation: sorted grain-maximum band-averaged FIPs, Ac, high-χ fraction, enrichment ratio, χ at the stored-energy site, mean FIP of the last two cycles, last-cycle hysteresis loops |
| `dataset/simulations.csv` | One row per simulation with all metrics and Gumbel parameters |
| `dataset/monte_carlo_samples.csv` | η for every Monte Carlo sample |
| `dataset/tension_curves.csv` | Calibration stress–strain curves |
| `calibration.json` | Identified parameters and fit quality |
| `tables/*.csv` | Tables of the manuscript (see the map below) |
| `figures/*.png` | Figures 1–12 of the manuscript (see the map below) |
| `numbers.json` | Every number quoted in the manuscript text |

Tags: `_meanref` homogeneous reference with the same mean slip resistance as the segregated field, `conv_*` convergence
study, `oat_<parameter>_<factor>` one-at-a-time sensitivity, `mc<NNN>` Monte Carlo sample. gs/g0 = 1.00 is the
segregation-free reference. Note that `conv_n216_g20` has the same settings as the reference configuration and reproduces it.

### Manuscript tables and figures

| Manuscript | File |
|---|---|
| Table 1 | `tables/table4_parameters.csv` |
| Table 2 | simulation matrix (defined in `run_study.py`) |
| Table 3 | `tables/table8_convergence.csv` |
| Table 4 | `tables/table5_orientation.csv` |
| Table 5 | `tables/table5b_heterogeneity.csv` |
| Table 6 | `tables/table6_kt.csv` |
| Table 7 | `tables/table7_contrast.csv` |
| Table 8 | `tables/table3_oat.csv` |
| Table 9 | `tables/table9_monte_carlo.csv` |
| Figure 1 | `figures/fig01_workflow.png` |
| Figure 2 | `figures/fig12_calibration.png` |
| Figure 3 | `figures/fig02_rve_segregation.png` |
| Figure 4 | `figures/fig03_bounds.png` |
| Figure 5 | `figures/fig11_convergence.png` |
| Figure 6 | `figures/fig08_hysteresis.png` |
| Figure 7 | `figures/fig05_eta_orientation.png` |
| Figure 8 | `figures/fig04_crack_sites.png` |
| Figure 9 | `figures/fig10_colocation.png` |
| Figure 10 | `figures/fig06_eta_kt.png` |
| Figure 11 | `figures/fig07_eta_contrast.png` |
| Figure 12 | `figures/fig09_gumbel.png` |

File names are those of the analysis script; the manuscript numbering follows the order of appearance.

## Map from manuscript to code

| Manuscript | Code |
|---|---|
| Eqs. (1)–(5), Appendix A | `cp_solver.TaylorCP`, `cp_solver.SachsCP` |
| Eqs. (6)–(7), Section 2.3 | `rve.generate_voronoi_rve`, `rve.seed_segregation_field` |
| Eq. (8), Section 2.4 | `TaylorCP.step` (lateral strains adjusted so ⟨σ⟩ is uniaxial), `SachsCP.step` (⟨e·ε·e⟩ = ε̄) |
| Eq. (9) | `lattice.hotspot_strain`, `pipeline.simulate` |
| Eqs. (10)–(12) | `fip.fatemi_socie_fip`, `fip.band_average`, `fip.fit_gumbel` |
| Eq. (13), TN | `lattice.life_reduction`, `fip.scatter_band` |
| Eqs. (14)–(15), Ac | `crack.*` |
| Table 1, Figure 2 | `calibrate.py`, `config.PROVENANCE` |
| Tables 2–9, Figures 1 and 3–12 | `run_study.py`, `analyze.py` |

## Scope and limitations

These are mean-field bounds: they bracket the effect of segregation but do not compute the full-field value. Laves
fracture, interface decohesion and loss of ductility are not modelled. The calibration data are for wire-arc DED IN718.
Kt is parametric. No fatigue data exist for pulsed-laser DED IN718 lattices, so the results have not been validated
experimentally.

## History

Version 0.1 of this repository accompanied the original submission. It did not compute the metrics it reported and it
disagreed with the manuscript. It has been replaced in full. The original manuscript figures are kept for transparency
in `manuscript_figures/`.

## Licence

MIT (see `LICENSE`).
