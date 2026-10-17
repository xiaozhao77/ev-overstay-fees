"""
Consecutive-night simulation (Appendix B.10: fairness over several nights).

The set of households is fixed. Each night, households rejected (or abandoned in the queue) the night before apply again, and the others apply with probability prob.
A household that applies again starts with an SOC 0.1 lower than before (not below the lower bound). All methods face the same households and random numbers,
but the set of rejected households differs by method. RH-F raises the admission reward of a household rejected on k consecutive nights to alpha(1 + prio*k).

Example:
  python -m ev_overstay.multinight --households 96 --prob 0.25 --nights 14 --methods B0 RH RH-F \
         --seeds 10 --solver copt --jobs 4 --set solver.rh_mip_gap=0.001 --out mn_results.csv
"""
import argparse
import copy
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .config import Scenario
from .instance import generate_from, _battery_table, _trunc_normal, _hour_to_t
from .simulator import Simulator
from .policies import make_policy
from .run_experiments import parse_rule

EXTRA = {}


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def run_one(task):
    (seed, method, H, prob, nights, ratio, rule_s, prio, solver, threads, extra) = task
    sc = Scenario()
    for k_, v_ in extra.items():
        o, a = k_.split('.')
        setattr(getattr(sc, o), a, v_)
    sc.solver.solver = solver
    if threads:
        sc.solver.threads = threads
    sc.station.cap_ratio_peak = ratio
    sc.rule.rule, sc.rule.theta = parse_rule(rule_s)
    st, dc = sc.station, sc.demand
    sc.demand.n = int(round(H * prob))            # expected requests per night used for forecasting (excluding repeat applications)
    rng0 = np.random.default_rng([seed, 1])
    bat = _battery_table(dc)
    batt = []
    for h in range(H):
        if bat is not None:
            models, groups = bat
            g = groups[models[int(rng0.integers(len(models)))]]
            batt.append(float(g[int(rng0.integers(len(g)))]))
        else:
            batt.append(float(rng0.choice(dc.batteries)))
    pending = {}           # household -> (consecutive rejected nights, last initial SOC)
    rej_count = np.zeros(H, int)
    req_count = np.zeros(H, int)
    max_consec = np.zeros(H, int)
    rows = []
    t0 = time.time()
    for night in range(nights):
        raw = []
        for h in range(H):
            r_ = np.random.default_rng([seed, night, h, 2])     # fixed random numbers per household and night, shared by all methods
            u_req, a, dpt, s_new = r_.random(), r_.normal(dc.arr_mean, dc.arr_sd), r_.normal(dc.dep_mean, dc.dep_sd), \
                _trunc_normal(r_, dc.soc0_mean, dc.soc0_sd, dc.soc0_low, dc.soc0_high)
            if h in pending:
                k, s_prev = pending[h]
                s0 = max(dc.soc0_low, s_prev - 0.1)
            elif u_req < prob:
                k, s0 = 0, s_new
            else:
                continue
            a = float(np.clip(a, 16.0, 23.5)); dpt = float(np.clip(dpt, 6.0, 10.0))
            r = max(0, min(st.T - 2, _hour_to_t(a, st)))
            d = max(r + 1, min(st.T, _hour_to_t(dpt, st, next_day=dpt < st.start_hour)))
            raw.append((r, d, batt[h], s0, h, prio * k))
        inst = generate_from(sc, seed * 1000 + night, raw)
        pol = make_policy(method)
        sim = Simulator(inst, pol)
        met = sim.run() if raw else {}
        new_pending = {}
        rows.append(dict(seed=seed, method=method, night=night, n_req=len(raw), **{k_: met.get(k_) for k_ in ('reward', 'accepted', 'served', 'abandoned', 'avg_fee', 'fee_payers')}))
        # results per vehicle
        for x in (sim.v if raw else []):
            h = x.hh
            req_count[h] += 1
            if x.outcome in ('rejected', 'abandoned'):
                rej_count[h] += 1
                k = (pending.get(h, (0, 0))[0]) + 1
                new_pending[h] = (k, x.soc0)
                max_consec[h] = max(max_consec[h], k)
        pending = new_pending
    asked = req_count > 0
    summ = dict(seed=seed, method=method, night=-1, n_req=int(req_count.sum()),
                reward=float(np.mean([r['reward'] for r in rows if r['reward'] is not None])),
                reject_share=float(rej_count.sum() / max(1, req_count.sum())),
                hh_rejected_any=float((rej_count[asked] > 0).mean()),
                hh_consec2=float((max_consec[asked] >= 2).mean()),
                hh_consec3=float((max_consec[asked] >= 3).mean()),
                max_consec=int(max_consec.max()),
                rej_p90=float(np.percentile(rej_count[asked], 90)),
                rej_max=int(rej_count.max()),
                rej_gini=gini(rej_count[asked]),
                wall_s=time.time() - t0)
    return rows + [summ]


def _task(arg):
    return run_one(arg)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Consecutive-night simulation')
    ap.add_argument('--households', type=int, default=96)
    ap.add_argument('--prob', type=float, default=0.25)
    ap.add_argument('--nights', type=int, default=14)
    ap.add_argument('--methods', nargs='+', default=['B0', 'RH', 'RH-F'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--ratio', type=float, default=0.6)
    ap.add_argument('--rule', default='cong0')
    ap.add_argument('--prio', type=float, default=1.0, help='RH-F: admission reward rises by prio*alpha for each consecutive rejected night')
    ap.add_argument('--solver', default='highs', choices=['highs', 'copt', 'gurobi'])
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--out', default='mn_results.csv')
    a = ap.parse_args(argv)
    for kv in a.set:
        k_, v_ = kv.split('=')
        EXTRA[k_] = float(v_)
    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r['night'] == '-1':
                    done.add((r['seed'], r['method']))
    threads = 1 if a.jobs > 1 else 0
    tasks = [(s, m, a.households, a.prob, a.nights, a.ratio, a.rule, a.prio, a.solver, threads, dict(EXTRA))
             for s in range(1, a.seeds + 1) for m in a.methods if (str(s), m) not in done]
    print(f'{len(tasks)} tasks ({len(done)} finished tasks skipped)', flush=True)
    cols = ['seed', 'method', 'night', 'n_req', 'reward', 'accepted', 'served', 'abandoned', 'avg_fee', 'fee_payers',
            'reject_share', 'hh_rejected_any', 'hh_consec2', 'hh_consec3', 'max_consec', 'rej_p90', 'rej_max', 'rej_gini', 'wall_s']
    if not os.path.exists(a.out):
        with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(cols)

    def write(rows):
        with open(a.out, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            for r in rows:
                w.writerow(r)
    t_start = time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(_task, t) for t in tasks]
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    write(fu.result())
                except Exception:
                    import traceback
                    print('[error] a task failed:\n' + traceback.format_exc(), flush=True)
                print(f'[{i}/{len(tasks)}] elapsed {time.time() - t_start:.0f}s', flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            write(_task(t))
            print(f'[{i}/{len(tasks)}] elapsed {time.time() - t_start:.0f}s', flush=True)
    print('Results written to', a.out)


if __name__ == '__main__':
    main()
