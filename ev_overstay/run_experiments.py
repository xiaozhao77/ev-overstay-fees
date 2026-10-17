"""
Batch experiments (Section 6 of the paper; scripts/run_all.sh lists the commands for all result files).

Examples:
  # small test run (also works with a size-limited solver licence)
  python -m ev_overstay.run_experiments --preset small --seeds 5 --offline --out results_small.csv

  # method comparison (fee notices off so that the offline solution is a strict upper bound)
  python -m ev_overstay.run_experiments --ratios 0.4 0.6 0.8 --rules cong0 --methods B0 B1 RH RH-N \
         --seeds 30 --offline --no-notice --jobs 8 --out res_methods.csv

  # rule comparison
  python -m ev_overstay.run_experiments --ratios 0.4 0.6 0.8 --rules none always cong0 cong5 \
         --methods RH B0 --seeds 30 --jobs 8 --out res_rules.csv

  # sensitivity analysis: fee rate, grace period, delay quantile
  python -m ev_overstay.run_experiments --rules cong0 --phis 30 60 90 120 --methods RH --seeds 30 --out res_phi.csv
  python -m ev_overstay.run_experiments --rules cong0 --graces 0.17 0.33 0.5 --methods RH --seeds 30 --out res_grace.csv
  python -m ev_overstay.run_experiments --rules cong0 --pquants 0.3 0.5 0.7 0.9 --methods RH --seeds 30 --out res_p.csv

Rules: none | always | congN (congestion-triggered, threshold theta = N idle chargers) | wait (queue-triggered)
Rows are appended to the CSV file; rerunning the same command skips finished combinations.
"""
import argparse
import copy
import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

if __package__ in (None, ''):          # when the file is run directly (python run_experiments.py)
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ev_overstay.config import Scenario
    from ev_overstay.instance import generate
    from ev_overstay.simulator import Simulator
    from ev_overstay.policies import make_policy
    from ev_overstay.model import solve_offline
else:
    from .config import Scenario
    from .instance import generate
    from .simulator import Simulator
    from .policies import make_policy
    from .model import solve_offline

EXTRA = {}
KEY = ['behavior', 'n', 'ratio', 'rule', 'theta', 'phi', 'grace_h', 'quantile_p', 'notice', 'seed', 'method']


def preset(name) -> Scenario:
    sc = Scenario()
    if name == 'small':        # toy size for a restricted licence (<= 2000 variables), only for checking the code
        sc.station.tau = 0.5
        sc.station.m = 3
        sc.station.K = 2
        sc.station.base_peak_kw = 20
        sc.station.cap_ratio_peak = 0.8
        sc.demand.n = 5
        sc.demand.soc0_low = 0.4
        sc.rule.grace_hours = 0.5
    return sc


def parse_rule(s):
    if s in ('none', 'always'):
        return s, 0
    if s == 'wait':
        return 'waiting', 0
    if s.startswith('cong'):
        return 'congestion', int(s[4:] or 0)
    raise ValueError(s)


def run_task(task):
    """One task = one instance x one set of rule parameters; returns one row per method."""
    (base, n, ratio, rule_s, phi, grace, pq, notice, seed, methods, offline, threads, solver, beh) = task
    sc = copy.deepcopy(base)
    for k_, v_ in EXTRA.items():
        obj_, attr_ = k_.split('.')
        setattr(getattr(sc, obj_), attr_, v_)
    if beh not in ('base',):
        from ev_overstay.config import BEHAVIOR_SCENARIOS
        for k_, v_ in BEHAVIOR_SCENARIOS[beh].items():
            setattr(sc.behavior, k_, v_)
    sc.solver.solver = solver
    sc.demand.n = n
    sc.station.cap_ratio_peak = ratio
    rule, theta = parse_rule(rule_s)
    sc.rule.rule, sc.rule.theta = rule, theta
    if phi is not None:
        sc.rule.phi = phi
    if grace is not None:
        sc.rule.grace_hours = grace
    if pq is not None:
        sc.solver.quantile_p = pq
    sc.solver.fee_onset_notice = notice
    if threads:
        sc.solver.threads = threads
    inst = generate(sc, seed)
    common = dict(behavior=beh, n=n, ratio=ratio, rule=rule, theta=theta, phi=sc.rule.phi, grace_h=round(sc.rule.grace_hours, 4),
                  quantile_p=sc.solver.quantile_p, notice=int(notice), seed=seed,
                  m=sc.station.m, tau=sc.station.tau, solver=solver)
    rows = []
    if offline:
        t0 = time.time()
        try:
            res = solve_offline(inst)
        except Exception:
            import traceback
            err = traceback.format_exc()
            print(f'[error] offline model, instance seed={seed}, ratio={ratio}, rule={rule_s}\n{err}', flush=True)
            res = dict(status='error', error=err.strip().splitlines()[-1])
        rows.append(dict(common, method='OFF', reward=res.get('obj', float('nan')),
                         off_bound=res.get('bound', float('nan')), off_gap=res.get('gap', float('nan')),
                         accepted=res.get('accepted', float('nan')), served=res.get('satisfied', float('nan')),
                         service_rate=res.get('satisfied', float('nan')) / max(1, n),
                         accept_rate=res.get('accepted', float('nan')) / max(1, n),
                         wall_s=time.time() - t0))
    for mname in methods:
        t0 = time.time()
        try:
            pol = make_policy(mname)
            met = Simulator(inst, pol).run()
            met['rh_fail'] = getattr(pol, 'n_fail', 0)
            rows.append(dict(common, method=mname, wall_s=time.time() - t0, **met))
        except Exception:
            import traceback
            err = traceback.format_exc()
            print(f'[error] method {mname}, instance seed={seed}, ratio={ratio}, rule={rule_s}\n{err}', flush=True)
            rows.append(dict(common, method=mname, wall_s=time.time() - t0, error=err.strip().splitlines()[-1]))
    return rows


def _run_with_extra(arg):
    extra, t = arg
    EXTRA.clear(); EXTRA.update(extra)
    return run_task(t)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Idle-fee experiments at a shared residential charging station')
    ap.add_argument('--preset', default='paper', choices=['paper', 'small'])
    ap.add_argument('--ns', type=int, nargs='+', default=None, help='requests per night')
    ap.add_argument('--ratios', type=float, nargs='+', default=None, help='share of chargers that can operate at the evening peak')
    ap.add_argument('--rules', nargs='+', default=['cong0'])
    ap.add_argument('--phis', type=float, nargs='+', default=[None], help='idle fee rate, CNY/h')
    ap.add_argument('--graces', type=float, nargs='+', default=[None], help='grace period, h')
    ap.add_argument('--pquants', type=float, nargs='+', default=[None], help='delay quantile p')
    ap.add_argument('--behaviors', nargs='+', default=['base'],
                    help='move-out behavior scenarios S1-S7 (S2 is the base case, same as base)')
    ap.add_argument('--methods', nargs='+', default=['B0', 'B1', 'RH', 'RH-N'])
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--seed0', type=int, default=1)
    ap.add_argument('--offline', action='store_true', help='also solve the offline model')
    ap.add_argument('--no-notice', action='store_true', help='switch off fee notices in the online simulation')
    ap.add_argument('--solver', default='highs', choices=['highs', 'copt', 'gurobi'],
                    help='MILP solver: highs (open source, default) / copt / gurobi')
    ap.add_argument('--jobs', type=int, default=1, help='parallel processes')
    ap.add_argument('--out', default='results.csv')
    ap.add_argument('--set', nargs='*', default=[], help='override parameters, e.g. rule.fee_from_full=1 behavior.insensitive_share=0.2')
    ap.add_argument('--tariff', default=None, choices=[None, 'flat'], help='flat = flat tariff all day')
    ap.add_argument('--methods-extra', nargs='*', default=[])
    a = ap.parse_args(argv)
    for kv in a.set:
        k_, v_ = kv.split('=')
        EXTRA[k_] = (v_ in ('1', 'True', 'true')) if k_.endswith(('fee_from_full',)) else float(v_)
    if a.tariff == 'flat':
        EXTRA['station.tou'] = [(0, 24, 0.617)]

    base = preset(a.preset)
    ns = a.ns or [base.demand.n]
    ratios = a.ratios or [base.station.cap_ratio_peak]
    notice = not a.no_notice
    threads = 1 if a.jobs > 1 else 0

    done = set()
    if os.path.exists(a.out):
        with open(a.out, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                done.add((r.get('behavior', 'base'), r['n'], r['ratio'], r['rule'] + r['theta'], r['phi'], r['grace_h'],
                          r['quantile_p'], r['seed']))

    tasks = []
    for beh, n, ratio, rule_s, phi, grace, pq, s in itertools.product(
            a.behaviors, ns, ratios, a.rules, a.phis, a.graces, a.pquants, range(a.seed0, a.seed0 + a.seeds)):
        sc_tmp = copy.deepcopy(base)
        rule, theta = parse_rule(rule_s)
        key = (beh, str(n), str(ratio), rule + str(theta),
               str(phi if phi is not None else sc_tmp.rule.phi),
               str(round(grace if grace is not None else sc_tmp.rule.grace_hours, 4)),
               str(pq if pq is not None else sc_tmp.solver.quantile_p), str(s))
        if key in done:
            continue
        tasks.append((base, n, ratio, rule_s, phi, grace, pq, notice, s, a.methods, a.offline, threads, a.solver, beh))
    print(f'{len(tasks)} tasks ({len(done)} finished tasks skipped)')

    header = None
    if os.path.exists(a.out):
        with open(a.out, newline='', encoding='utf-8-sig') as f:
            header = next(csv.reader(f), None)

    def write(rows):
        nonlocal header
        if header is None:
            cols = []
            for r in rows:
                for k in r:
                    if k not in cols:
                        cols.append(k)
            # OFF rows and method rows have different fields; collect all possible fields first
            extra = ['off_bound', 'off_gap', 'rh_fail', 'error']
            header = cols + [c for c in extra if c not in cols]
            with open(a.out, 'w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(header)
        with open(a.out, 'a', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            for r in rows:
                w.writerow(r)

    t_start = time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(_run_with_extra, (dict(EXTRA), t)) for t in tasks]
            for i, fu in enumerate(as_completed(futs), 1):
                try:
                    write(fu.result())
                except Exception:
                    import traceback
                    print('[error] a task failed and was skipped:\n' + traceback.format_exc(), flush=True)
                print(f'[{i}/{len(tasks)}] elapsed {time.time() - t_start:.0f}s')
    else:
        for i, t in enumerate(tasks, 1):
            write(run_task(t))
            print(f'[{i}/{len(tasks)}] elapsed {time.time() - t_start:.0f}s')
    print('Results written to', a.out)


if __name__ == '__main__':
    main()
