import sys, copy, pandas as pd
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from ev_overstay import run_experiments as RE
from ev_overstay.simulator import Simulator
from ev_overstay.policies import make_policy
from ev_overstay.instance import generate
rows = []
for cap in [0, 60, 30]:
    for phi in [60, 30]:
        for rule in ['always', 'congestion']:
            for ratio in [0.4, 0.6, 0.8]:
                for seed in range(1, 31):
                    for meth in ['B0', 'B0-NR', 'B0-Q']:
                        sc = copy.deepcopy(RE.preset('paper')); sc.station.cap_ratio_peak = ratio
                        sc.rule.rule = rule; sc.rule.theta = 0; sc.rule.phi = phi; sc.rule.fee_cap = cap
                        sc.rule.fee_from_full = True; sc.solver.fee_onset_notice = False
                        sim = Simulator(generate(sc, seed), make_policy(meth)); sim.run(); m = sim.metrics()
                        fees = [x.fee for x in sim.v if x.fee > 0]
                        rows.append(dict(cap=cap, phi=phi, rule=rule, ratio=ratio, seed=seed, method=meth, reward=sim.reward(),
                                         served=m['served'], avg_fee=m['avg_fee'], payers=len(fees), per_payer=(sum(fees) / len(fees) if fees else 0)))
d = pd.DataFrame(rows); d.to_csv('results/r16_feecap.csv', index=False)
g = d.groupby(['cap', 'phi', 'method', 'rule'])[['reward', 'served', 'avg_fee', 'payers']].mean().round(2)
print(g.to_string())
