"""
Reminder rule derived from Proposition 1 (B0-Q, Section 6.6): B0 that reminds late at night only
while a vehicle waits. Compared with B0 and B0-NR under the reminder clock and the fee clock
starting at full charge (90 instances, fee notices off).

Usage (from the repository root):
    python scripts/check_b0q.py --out results/r15_b0q.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse, copy
import pandas as pd
from ev_overstay import run_experiments as RE
from ev_overstay.simulator import Simulator
from ev_overstay.policies import make_policy
from ev_overstay.instance import generate

ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/r15_b0q.csv'); a = ap.parse_args()
rows = []
for clock in ['rem', 'full']:
    for rule in ['none', 'always', 'congestion']:
        for ratio in [0.4, 0.6, 0.8]:
            for seed in range(1, 31):
                for meth in ['B0', 'B0-NR', 'B0-Q']:
                    sc = copy.deepcopy(RE.preset('paper'))
                    sc.station.cap_ratio_peak = ratio; sc.rule.rule = rule; sc.rule.theta = 0
                    sc.solver.fee_onset_notice = False; sc.rule.fee_from_full = (clock == 'full')
                    sim = Simulator(generate(sc, seed), make_policy(meth)); sim.run(); m = sim.metrics()
                    rows.append(dict(clock=clock, notice=False, rule=rule, ratio=ratio, seed=seed, method=meth,
                                     reward=sim.reward(), served=m.get('served'), night_rem=m.get('night_reminders'),
                                     energy=m.get('energy_cost'), fee=m.get('avg_fee'), nn=sim.n_notice_night))
d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
print(d.groupby(['clock', 'rule', 'method'])[['reward', 'served', 'night_rem', 'energy', 'fee']].mean().round(2))
