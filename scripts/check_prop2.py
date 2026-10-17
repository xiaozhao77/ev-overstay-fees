"""
First-order check of Proposition 2 (Appendix A).

For each instance, the congestion-triggered rule (theta = 0) is simulated with unchanged response
probabilities, and every owner who moves within the grace period at full occupancy is then forced
to stay, one at a time, to obtain the reward loss v_k. The first-order prediction of the gap at a
weaker fee sensitivity gamma_1 is compared with the simulated gap.

Usage (from the repository root):
    python scripts/check_prop2.py --seeds 1 101 --out results/r15_prop2_b0.csv
Rule-based policies run in seconds; B0 with 100 instances takes about a minute.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse, copy, csv, math
import numpy as np
from ev_overstay import run_experiments as RE
from ev_overstay.simulator import Simulator
from ev_overstay.policies import make_policy
from ev_overstay.instance import generate
import ev_overstay.behavior as bh

ap = argparse.ArgumentParser()
ap.add_argument('--method', default='B0')
ap.add_argument('--seeds', type=int, nargs=2, default=[1, 101])
ap.add_argument('--weak', type=float, nargs='+', default=[0.0375, 0.035])
ap.add_argument('--out', default='results/r15_prop2_b0.csv')
a = ap.parse_args()

def scen(rule, g1=0.045):
    sc = copy.deepcopy(RE.preset('paper'))
    sc.station.cap_ratio_peak = 0.6; sc.rule.rule = rule; sc.rule.theta = 0
    sc.solver.fee_onset_notice = True; sc.behavior.gamma1 = g1
    return sc

def run(sc, seed, force=None, log=None):
    inst = generate(sc, seed); orig = inst.Gam
    def G(j, t):
        if force is not None and j == force:
            return bh.INF
        g = orig(j, t)
        if log is not None and g < bh.INF:
            log[j] = t
        return g
    inst.Gam = G
    sim = Simulator(inst, make_policy(a.method)); sim.run()
    return sim.reward()

def p(g1, night):
    b = scen('congestion').behavior
    return 1 / (1 + math.exp(-(b.gamma0 + g1 * 60 - b.gamma2 * night)))

rows, RA, RH, RC, PRED = [], [], [], {g: [] for g in a.weak}, {g: [] for g in a.weak}
for seed in range(*a.seeds):
    ra = run(scen('always'), seed)
    log = {}; sc = scen('congestion'); rh = run(sc, seed, log=log)
    ev = [(j, t, int(bh.is_night(sc.station.clock(t), sc.behavior)), rh - run(sc, seed, force=j)) for j, t in log.items()]
    RA.append(ra); RH.append(rh)
    for g in a.weak:
        RC[g].append(run(scen('congestion', g), seed))
        PRED[g].append((rh - ra) - sum((1 - p(g, n) / p(0.045, n)) * v for _, _, n, v in ev))
    for j, t, n, v in ev or [(None, None, None, None)]:
        rows.append([seed, ra, rh] + [RC[g][-1] for g in a.weak] + [j, t, n, v])
RA, RH = np.array(RA), np.array(RH); base = RA.mean()
nev = sum(1 for r in rows if r[-1] is not None)
print(f'g0 = {100 * (RH - RA).mean() / base:.2f}% ; owners moving within the grace period at full occupancy per night = {nev / len(RA):.2f}')
for g in a.weak:
    sim_, pred_ = np.array(RC[g]) - RA, np.array(PRED[g])
    print(f'gamma1 = {g}: simulated gap {100 * sim_.mean() / base:.2f}% , first-order prediction {100 * pred_.mean() / base:.2f}%')
with open(a.out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['seed', 'reward_always', 'reward_congestion'] + [f'reward_congestion_g1_{g}' for g in a.weak] + ['vehicle', 'event_period', 'late_night', 'reward_loss'])
    w.writerows(rows)
