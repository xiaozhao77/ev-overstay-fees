#!/bin/bash
# Reproduces every result file in results/ (run from the repository root).
# The full set takes several days with COPT; each block can be run on its own.
# Finished parameter combinations are skipped when a run is restarted.
#
#   bash scripts/run_all.sh            # everything
#   JOBS=4 bash scripts/run_all.sh     # parallel processes (default 2)
#
# Settings used in the paper: COPT 8.0.7, 0.1% relative gap per rolling-horizon step
# (solver.rh_mip_gap=0.001), 20 s per step, 1,800 s for the offline model.
# Without a COPT licence, use --solver highs (slower; single-instance RH rewards can differ, see Appendix B.7).

JOBS=${JOBS:-2}
SOLVER=${SOLVER:-copt}
G="solver.rh_mip_gap=0.001"
run() { python -m ev_overstay.run_experiments "$@" --solver $SOLVER --jobs $JOBS; }
O=results

# ---------------------------------------------------------------- Section 6.3, Table 9, Figures 4-5
run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 B1 RH RH-N --offline --no-notice --seeds 30 --set $G --out $O/res_methods.csv
run --ratios 0.4 0.6 0.8 --rules always cong0 --methods RH RH-N --no-notice --seeds 30 --set $G solver.rh_time_limit=60 --out $O/r12_tl60.csv

# ---------------------------------------------------------------- Section 6.4, Table 10, Figure 6 (fee clock from full charge)
run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 RH --no-notice --seeds 30 --set $G rule.fee_from_full=1 --out $O/r8_full_t10.csv
run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 B0-NR B0-TOU B0-TOU+ --no-notice --seeds 30 --set $G rule.fee_from_full=1 --out $O/r14_full.csv
run --ratios 0.4 0.6 0.8 --rules always cong0 --methods RH --no-notice --seeds 30 --set $G rule.fee_from_full=1 --out $O/r14_full_rh.csv

# ---------------------------------------------------------------- Section 6.4, Table 11, Figure 7 (thresholds and queue-triggered rule)
run --rules cong0 cong6 --methods B0 RH --seeds 30 --set $G --out $O/res_theta.csv
run --rules cong1 cong2 cong3 cong4 cong6 cong9 --methods B0 RH --seeds 30 --set $G --out $O/rev_e13_theta.csv
run --rules wait --methods B0 RH --seeds 30 --set $G --out $O/rev_e10_wait.csv
run --rules always cong0 --methods B0 RH --seeds 30 --set $G --out $O/rev_e2_base.csv

# ---------------------------------------------------------------- Section 6.5, Figure 8 (weaker response to congestion-triggered fees)
run --rules none always cong0 --methods B0 RH --seeds 100 --set $G behavior.gamma1=0.045 --out $O/r10_g1_0.045.csv
for g in 0.0425 0.04 0.0375 0.035; do
  run --rules cong0 --methods B0 RH --seeds 100 --set $G behavior.gamma1=$g --out $O/r10_g1_$g.csv
done
for g in 0.035 0.025 0.015 0.0; do
  run --rules cong0 --methods B0 RH --seeds 30 --set $G behavior.gamma1=$g --out $O/r7_g1_$g.csv
done
for q in 0.1 0.2 0.3; do
  run --rules cong0 --methods B0 RH --seeds 100 --set $G behavior.insensitive_share=$q --out $O/r12_ign$q.csv
done
# r13_boundary.csv (the points of Figure 8) is the paired summary of the files above.
python scripts/check_prop2.py --seeds 1 101 --out $O/r15_prop2_b0.csv          # Proposition 2, Appendix A

# ---------------------------------------------------------------- Section 6.6, Figures 9-10, Table B4
run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 B0-NR B0-TOU --no-notice --seeds 30 --set $G --out $O/r8_b0var.csv
run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 B0-NR B0-TOU B0-TOU+ --no-notice --seeds 30 --set $G --out $O/r12_tou_plus.csv
run --ratios 0.4 0.6 0.8 --rules cong0 --methods B0 B0-NR B0-TOU B1 RH RH-N --no-notice --seeds 30 --set $G --out $O/r10_hist_tou.csv
run --ratios 0.4 0.6 0.8 --rules cong0 --methods B0 RH RH-N --no-notice --seeds 30 --tariff flat --set $G --out $O/r10_hist_flat.csv
run --ratios 0.4 0.6 0.8 --rules cong0 --methods B0 B1 RH RH-N --no-notice --seeds 30 --set $G --out $O/r13_qhist.csv
python scripts/check_b0q.py --out $O/r15_b0q.csv                                 # B0-Q, Proposition 1

# ---------------------------------------------------------------- Section 6.7, Tables 12-13, Figure 11
run --behaviors S1 S2 S3 S4 S5 S6 S7 --rules none always cong0 --methods B0 RH RH-N --seeds 30 --set $G --out $O/res_behavior.csv
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G behavior.hetero_share=0.5 --out $O/r5_hetero.csv
run --ns 12 24 36 48 --rules none always cong0 --methods B0 B1 RH RH-N --seeds 30 --set $G --out $O/res_n.csv
run --ns 16 20 --rules none always cong0 --methods B0 RH --seeds 30 --set $G --out $O/r7_n1620.csv
run --phis 30 60 120 384 --rules always cong0 --methods B0 RH --seeds 30 --set $G --out $O/res_phi.csv
run --phis 5 10 20 --rules always cong0 --methods B0 RH --seeds 30 --set $G --out $O/r6_lowphi.csv
run --graces 0.1667 0.3333 0.5 --rules always cong0 --methods B0 RH --seeds 30 --set $G --out $O/res_grace.csv
run --rules cong0 --graces 0.25 0.5 0.75 --phis 30 60 120 --methods B0 RH --seeds 30 --set $G --out $O/rev_e14_gphi.csv
run --graces 0.25 0.5 0.75 --phis 60 120 --rules always cong0 --methods B0 RH --seeds 30 --set $G behavior.gamma3=0.5 --out $O/r5_grace_g3.csv
run --rules always cong0 --methods B0 RH --seeds 30 --set $G rule.fee_from_full=1 --out $O/rev_e2_full.csv
run --rules always cong0 --phis 60 384 --methods B0 RH --seeds 30 --set $G behavior.insensitive_share=0.2 --out $O/rev_e3_ins20.csv
run --rules always cong0 --phis 384 --methods B0 RH --seeds 30 --set $G --out $O/rev_e3_ins0_384.csv

# ---------------------------------------------------------------- Appendix B
run --rules cong0 --methods B0 RH --seeds 30 --set $G weights.alpha=5 --out $O/rev_e4_a5.csv                     # B.1
run --rules cong0 --methods B0 RH --seeds 30 --set $G weights.alpha=20 --out $O/rev_e4_a20.csv                   # B.1
run --rules cong0 --methods B0 RH RH-N RH-TB --seeds 30 --tariff flat --set $G --out $O/rev_e5_flat.csv          # B.2
run --rules cong0 --methods B0 RH --seeds 30 --set $G demand.arr_mean=19.6 --out $O/rev_e6_late.csv              # B.3
run --rules cong0 --pquants 0.3 0.7 --methods RH RH-N --seeds 30 --set $G --out $O/rev_e7_p.csv                  # B.4
run --rules cong0 --methods RH --seeds 30 --set $G solver.forecast_hours=1.5 --out $O/rev_e8_h15.csv            # B.4
run --rules cong0 --methods RH --seeds 30 --set $G solver.forecast_hours=4.5 --out $O/rev_e8_h45.csv            # B.4
run --rules cong0 --methods RH --seeds 30 --set $G solver.forecast_weight=0.5 --out $O/rev_e9_z05.csv           # B.4
run --rules cong0 --methods RH --seeds 30 --set $G solver.forecast_weight=0.9 --out $O/rev_e9_z09.csv           # B.4
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G station.K=0 --out $O/rev_e11_k0.csv            # B.5
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G station.K=6 --out $O/rev_e11_k6.csv            # B.5
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G station.W_max_hours=1.5 --out $O/rev_e12_w15.csv   # B.5
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G station.W_max_hours=5 --out $O/rev_e12_w5.csv      # B.5
run --rules cong0 --methods B0 RH --seeds 30 --set $G station.pv_kwp=0 solver.cap_target=1 --out $O/rev_e15_pv0cap.csv              # B.6
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G station.pv_kwp=50 solver.cap_target=1 --out $O/rev_e15_pv50.csv   # B.6
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G station.pv_kwp=100 solver.cap_target=1 --out $O/rev_e15_pv100.csv # B.6
run --rules cong0 --methods RH --seeds 30 --set $G station.pv_kwp=100 solver.cap_target=1 solver.pv_future_zero=0 --out $O/rev_e15_pv100f.csv  # B.6
run --ratios 0.4 0.6 0.8 --rules cong0 --methods RH RH-N RH-TB --no-notice --seeds 30 --set $G --out $O/rev_e1_tiebreak.csv   # B.7
run --ratios 0.4 0.6 0.8 --rules cong0 --methods RH RH-N --no-notice --seeds 30 --set $G --out $O/r6_det_a.csv                # B.7
run --ratios 0.4 0.6 0.8 --rules cong0 --methods RH RH-N --no-notice --seeds 30 --set $G --out $O/r6_det_b.csv                # B.7
run --ratios 0.4 0.6 0.8 --rules cong0 --methods RH RH-N --no-notice --seeds 30 --set $G solver.rh_fcfs=1 --out $O/r6_fcfs.csv   # B.7
run --ratios 0.4 0.6 0.8 --rules cong0 --methods RH RH-N --no-notice --seeds 30 --set solver.tb_lex=0.02 solver.rh_mip_gap=1e-6 --out $O/r5_tb_gap1e-6.csv  # B.7
run --ratios 0.4 0.6 0.8 --rules cong0 --methods RH RH-N --no-notice --seeds 30 --set solver.tb_lex=0.02 solver.rh_mip_gap=1e-7 --out $O/r5_tb_gap1e-7.csv  # B.7
run --ns 36 48 --rules cong0 --methods RH-N --seeds 30 --set $G solver.rh_time_limit=60 --out $O/r5_rhn_tl60_n.csv              # B.7
run --behaviors S2 S3 S5 S7 --rules cong0 --methods RH-N --seeds 30 --set $G solver.rh_time_limit=60 --out $O/r5_rhn_tl60_beh.csv # B.7
run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 B1 RH RH-N --offline --no-notice --seeds 100 --set $G --out $O/r5_n100_methods.csv  # B.8
run --rules none always cong0 wait --methods B0 RH --seeds 100 --set $G --out $O/r5_n100_base.csv                               # B.8
run --rules cong0 --methods RH-N --seeds 100 --set $G --out $O/r5_n100_base_rhn.csv                                             # B.8
python -m ev_overstay.multinight --households 96 --prob 0.25 --nights 14 --methods B0 RH RH-F --seeds 10 --prio 5 --set $G --solver $SOLVER --out $O/r5_multinight.csv  # B.10
run --ratios 0.4 0.6 0.8 --rules cong0 --methods B0 B1 RH RH-N --no-notice --seeds 30 --set $G behavior.plugin_psi_day=0.95 behavior.plugin_psi_night=0.5 --out $O/r7_plug_mod.csv   # B.11
run --ratios 0.4 0.6 0.8 --rules cong0 --methods B0 B1 RH RH-N --no-notice --seeds 30 --set $G behavior.plugin_psi_day=0.65 behavior.plugin_psi_night=0.2 --out $O/r7_plug_slow.csv  # B.11
run --rules none always cong0 --methods B0 RH --seeds 30 --set $G behavior.plugin_psi_day=0.95 behavior.plugin_psi_night=0.5 --out $O/r7_plug_rules.csv                            # B.11
run --ratios 0.4 0.6 0.8 --rules cong0 --methods RH-MS --no-notice --seeds 30 --set $G --out $O/r5_rhms.csv                     # B.12
# b1diag.csv (B.13) comes from a diagnostic run of B1 and B1+ with the rule-based simulator.
run --ratios 0.6 0.8 --rules none always cong0 --methods B0 B0-NR B1 RH RH-N --offline --no-notice --seeds 30 --set $G station.base_night_floor=1 --out $O/r8_floor.csv   # B.14
for C in 0 2.5 10; do
  run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 B0-NR RH RH-N --no-notice --seeds 30 --set $G weights.cN_night=$C --out $O/r9_cn$C.csv   # B.15
done
for B in 50 200; do
  run --ratios 0.4 0.6 0.8 --rules none always cong0 --methods B0 B0-NR RH --no-notice --seeds 30 --set $G weights.beta=$B --out $O/r10_beta$B.csv           # B.15
done
for NS in 23 0; do
  run --ratios 0.4 0.6 0.8 --rules none cong0 --methods B0 B0-NR RH RH-N --no-notice --seeds 30 --set $G behavior.night_start=$NS --out $O/r10_night$NS.csv # B.16
done
for P in 0.25 0.5 1.0; do
  run --ratios 0.4 0.6 0.8 --rules none always cong0 wait --methods B0 RH --no-notice --seeds 30 --set $G behavior.persist=$P --out $O/r11_persist$P.csv   # B.17
done
run --rules cong0 cong1 cong2 cong3 cong4 cong6 cong9 always wait --methods B0 RH --seeds 30 --set $G rule.fee_from_full=1 --out $O/r8_full_t11.csv  # B.18
