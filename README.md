# When overstay matters: congestion-triggered idle fees and reminder rules at shared residential EV charging stations

Code, input data, and instance-level results for the paper of the same title.

The repository contains

* the offline mixed-integer linear program (Section 4.2), the online simulator of the Markov decision process (Section 4.3), and the online methods B0, B1, RH, and RH-N (Section 5), together with the rule-based variants B0-NR, B0-TOU, B0-TOU+, and B0-Q and the robustness variants of Table B3;
* the instance generator with the Shanghai time-of-use tariff, the Shanghai holiday load shape, PVGIS output, and battery capacities of BEV models common in Shanghai;
* scripts that rerun every experiment, check Propositions 1 and 2 numerically, and draw Figures 3 to 11;
* the result files behind every table and figure.

## Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

The models can be solved with three solvers, selected with `--solver`:

| Solver | Installation | Licence | Note |
|---|---|---|---|
| `highs` (default) | included in SciPy | none | open source, 2 to 5 times slower than COPT |
| `copt` | `pip install coptpy` | free academic licence from Cardinal Operations | used for all results in the paper (COPT 8.0.7) |
| `gurobi` | `pip install gurobipy` | academic licence | |

COPT licence files go to `~/copt/`. Without a licence, COPT runs in a size-limited mode that only solves small test instances.

The offline optimum does not depend on the solver. When the window model of RH has several optimal solutions, different solvers may pick different ones, so single-instance online rewards can differ by several percent between solvers and settings, whereas mean rewards move by at most about 2.5% (Appendix B.3). Use one solver for any comparison.

## Quick start

Run from the repository root.

```bash
# small test instances (seconds)
python -m ev_overstay.run_experiments --preset small --seeds 5 --offline --out test.csv

# one base-case comparison: 30 nights, ratio 0.6, fee notices on
python -m ev_overstay.run_experiments --rules none always cong0 --methods B0 RH --seeds 30 --solver copt --set solver.rh_mip_gap=0.001 --out base.csv

# figures 3-11 from the stored results (writes figures/)
python scripts/make_all_figs.py
# figure 12 (served vehicles against late-night reminders)
python scripts/fig_pareto.py
```

Main options of `run_experiments`: `--ratios` (peak charger ratio 0.4, 0.6, 0.8), `--rules` (`none`, `always`, `congN` for the congestion-triggered rule with threshold θ = N, `wait` for the queue-triggered rule), `--methods`, `--seeds` (number of nights, starting at seed 1), `--offline` (also solve the offline model), `--no-notice` (switch off fee notices), `--phis` (fee rate, CNY/h), `--graces` (grace period, h), `--behaviors` (S1 to S7 of Table 9), `--set demand.early_share=0.3` (unannounced early departures), `--set solver.rh_dep_buffer_h=2` (safety margin of RH), `--ns` (requests per night), `--tariff flat`, and `--set section.parameter=value` to override any parameter in `config.py`. Finished combinations are skipped when a run is restarted.

## Reproducing the paper

`scripts/run_all.sh` contains the command for every result file, grouped by section. All runs use COPT with a 0.1% gap per RH step (`solver.rh_mip_gap=0.001`), a 20 s limit per step, and seeds 1 to 30 (1 to 100 where stated). Each night is one seed, and all methods and rules share the random numbers of a seed.

| Paper | Result files | Script |
|---|---|---|
| Table 10, Figures 4-5 (methods, offline bound) | `res_methods.csv` | `run_all.sh` |
| Section 6.3, 60 s per step | `r12_tl60.csv` | `run_all.sh` |
| Table 11, Figure 6 (fee clock from full charge) | `r8_full_t10.csv`, `r14_full.csv`, `r14_full_rh.csv` | `run_all.sh` |
| Table 12, Figure 7 (thresholds, queue trigger) | `res_theta.csv`, `rev_e13_theta.csv`, `rev_e10_wait.csv`, `rev_e2_base.csv` | `run_all.sh` |
| Section 6.5, Figure 8 (weaker response) | `r10_g1_*.csv`, `r7_g1_*.csv`, `r12_ign*.csv`, summary `r13_boundary.csv` | `run_all.sh` |
| Proposition 2, Appendix A (first-order check) | `r15_prop2_b0.csv` | `check_prop2.py` |
| Section 6.6, Figure 9, Table B4 (rule-based variants) | `r8_b0var.csv`, `r12_tou_plus.csv` | `run_all.sh` |
| Section 6.6, Figure 10 (completion times, queue) | `r10_hist_tou.csv`, `r10_hist_flat.csv`, `r13_qhist.csv` | `run_all.sh` |
| Section 6.6, B0-Q (Proposition 1) | `r15_b0q.csv` | `check_b0q.py` |
| Section 6.6, admission vs charging (RH-A; RH from `res_methods.csv`; HiGHS check) | `r16_rha.csv`, `r16_rha_highs.csv` | `run_all.sh` |
| Section 6.6, RH-A by demand (24, 36, 48 requests) and without a waiting area | `r18_rha_n24.csv`, `r17_rha_n.csv`, `r17_rha_k0.csv`, `r17_b0q_n.csv`, `r17_b0q_k0.csv` | `run_all.sh` |
| Section 6.5, solver check of the RH boundary (HiGHS; COPT with tie-breaking) | `r16_hs_g1_*.csv`, `r16_tb_g1_*.csv` | `run_all.sh` |
| Section 6.7.3, Appendix B.7 residential fee rate and fee cap | `r16_feecap.csv` | `check_feecap.py` |
| Section 6.7.5, unannounced early departures and RH with a safety margin (RH without margin and early departures from `res_n.csv`) | `r19_rb_e*.csv`, `r19_rh_e*.csv`, `r20_rh_b*_e*.csv`, `r21_*.csv` (up to 3 h) | `run_all.sh` |
| Section 6.7.6, owners who move out without a reminder (`behavior.spont_psi_day`, `behavior.spont_psi_night`; reference without such move-outs from `res_n.csv` and `r19_rb_e0.csv`) | `r22_rb_s1.csv`, `r22_rb_s2.csv`, `r22_rh_s1.csv`, `r22_rh_s2.csv` | `run_all.sh` |
| Table B6 (behavior scenarios) | `res_behavior.csv`, `r5_hetero.csv` | `run_all.sh` |
| Section 6.7.2, Figure 11(a, b) (demand) | `res_n.csv`, `r7_n1620.csv` | `run_all.sh` |
| Section 6.7.6, Figure 12 (served vehicles against late-night reminders) | `res_methods.csv`, `r12_tou_plus.csv`, `r15_b0q.csv`, `res_n.csv`, `r19_*_e0*.csv`, `r20_rh_b2_e0*.csv` | `scripts/fig_pareto.py` |
| Section 6.7.3, Figure 11(c, d) (fee rate, grace period) | `res_phi.csv`, `r6_lowphi.csv`, `res_grace.csv`, `rev_e14_gphi.csv`, `r5_grace_g3.csv` | `run_all.sh` |
| Table B7 (fee clock, fee-insensitive owners) | `rev_e2_full.csv`, `rev_e3_ins20.csv`, `rev_e3_ins0_384.csv` | `run_all.sh` |
| Supplement SM1 Objective weights | `rev_e4_a5.csv`, `rev_e4_a20.csv` | `run_all.sh` |
| Supplement SM2 Tariff structure | `rev_e5_flat.csv` | `run_all.sh` |
| Supplement SM3 Arrival times | `rev_e6_late.csv` | `run_all.sh` |
| Appendix B.1 Parameters of RH | `rev_e7_p.csv`, `rev_e8_h*.csv`, `rev_e9_z*.csv` | `run_all.sh` |
| Appendix B.2 Waiting area and maximum wait | `rev_e11_k*.csv`, `rev_e12_w*.csv` | `run_all.sh` |
| Supplement SM4 Rooftop PV | `rev_e15_pv*.csv` | `run_all.sh` |
| Appendix B.3 Solver settings and window model | `rev_e1_tiebreak.csv`, `r6_det_*.csv`, `r6_fcfs.csv`, `r5_tb_gap*.csv`, `r5_rhn_tl60_*.csv` | `run_all.sh` |
| Supplement SM5 Number of instances | `r5_n100_*.csv` | `run_all.sh` |
| Supplement SM6 Consecutive nights | `r5_multinight.csv` | `run_all.sh` |
| Supplement SM7 Plug-in delays | `r7_plug_*.csv` | `run_all.sh` |
| Supplement SM8 Multi-scenario benchmark | `r5_rhms.csv` | `run_all.sh` |
| Supplement SM9 Weakness of B1 | `b1diag.csv` | |
| Supplement SM10 Transformer limit binding all night | `r8_floor.csv` | `run_all.sh` |
| Appendix B.5 Reward weights | `r9_cn*.csv`, `r10_beta*.csv` | `run_all.sh` |
| Supplement SM11 Late-night window | `r10_night*.csv` | `run_all.sh` |
| Supplement SM12 Persistent response to fees | `r11_persist*.csv` | `run_all.sh` |
| Supplement SM13 Fee clock, threshold sweep | `r8_full_t11.csv` | `run_all.sh` |
| Appendix B.6 Proportional weakening | `r12_ign*.csv` | `run_all.sh` |

Statistical comparisons follow Section 6.1: paired differences in percent of the reference mean, 95% confidence intervals clustered by night when the three capacity levels are pooled, Wilcoxon signed-rank tests with Holm adjustment, and equivalence within ±2% by two one-sided tests. `ev_overstay/analyze.py` produces summary tables and tests for a result file.

Each result file has one row per instance, rule, and method. The main columns are `reward` (online reward, Eq. 21), `off_bound` (offline optimum, where solved), `accepted`, `served`, `energy_cost`, `reminders`, `night_reminders`, `avg_overstay_h`, `block_h`, `total_fee`, `avg_fee`, `fee_payers`, and `fee_state_share`.

## Repository layout

```
ev_overstay/            package
  config.py             all parameters (station, demand, behavior, fee rule, weights, solver)
  behavior.py           move-out response model, Eqs. (22)-(24), delay quantiles
  instance.py           instance generator and common random numbers
  lp.py                 solver-independent modelling layer (HiGHS, COPT, Gurobi)
  model.py              offline MILP, objective (1) and constraints (2)-(20)
  offline_copt.py       stand-alone COPT version of the offline model for one instance
  simulator.py          online simulation: event order, reward (21), fees (25), indicators
  policies.py           B0, B1, RH, RH-N and all variants
  run_experiments.py    batch experiments (parallel, resumable)
  multinight.py         consecutive-night simulation (Supplement SM6)
  analyze.py            summary tables and tests
  data/                 battery capacities, PVGIS hourly output for Shanghai
scripts/
  run_all.sh            commands for all result files
  check_prop2.py        first-order check of Proposition 2
  check_b0q.py          reminder rule B0-Q derived from Proposition 1
  check_feecap.py       residential fee rate and fee cap (Appendix B.7)
  make_all_figs.py      Figures 3-11
  fig_pareto.py         Figure 12
  fetch_acn.py          download ACN-Data and compute the overstay statistics of Section 6.2.2
results/                instance-level results behind all tables and figures
```

## Data sources

* **Arrival times.** The arrival-time distribution (mean 18.6 h, standard deviation 1.6 h, truncated to 4 p.m. to 11:30 p.m.) is fitted to ACN-Data sessions starting after 4 p.m. ACN-Data are publicly available from the Adaptive Charging Network (https://ev.caltech.edu/dataset) and are not redistributed here; `scripts/fetch_acn.py` downloads them with a free API token (`export ACN_TOKEN=...`).
* **PV output.** PVGIS hourly output for central Shanghai, 2023 (`ev_overstay/data/pv_shanghai_2023_hourly.csv`).
* **Battery capacities.** 97 configurations of BEV models common in Shanghai, compiled from manufacturers' specifications (`ev_overstay/data/battery_models_bev.csv`).
* **Tariff and base load.** Shanghai residential time-of-use tariff and the intraday shape of the Shanghai holiday load curve, both written in `config.py` and `instance.py`.

Owner-behavior parameters are assumptions, not estimates (Section 6.2.2 and Table 9).

## Licence

MIT, see `LICENSE`.

## Citation

If you use this code, please cite the paper (reference to be added on publication).
