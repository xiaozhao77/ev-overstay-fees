"""
Native COPT implementation of the deterministic offline model (Section 4.2 of the paper).

Unlike model.py, which builds the model through lp.py so that the solver can be switched for batch experiments,
this file writes the objective (1) and constraints (2)-(18) one by one in coptpy syntax with variable names that follow the paper,
so that the model can be checked against the paper and one instance can be solved with the schedule of every vehicle printed.

Run from the repository root:
  python -m ev_overstay.offline_copt                 # default parameters (full size, requires a COPT academic licence)
  python -m ev_overstay.offline_copt --small         # toy size
  python3 -m ev_overstay.offline_copt --rule none --seed 3
"""
import argparse
try:
    import coptpy as cp
    from coptpy import COPT
except ModuleNotFoundError:
    raise SystemExit('coptpy is not installed in this Python environment. Run, with the same Python:\n'
                     '    python3 -m pip install coptpy\n'
                     'In VS Code, check the selected interpreter first and run the command above in its terminal.')

if __package__ in (None, ''):          # when the file is run directly (python offline_copt.py)
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ev_overstay.config import Scenario
    from ev_overstay.instance import generate
else:
    from .config import Scenario
    from .instance import generate


def build_and_solve(inst, time_limit=1800, gap=1e-3, verbose=True):
    sc = inst.sc
    st, wc, rc = sc.station, sc.weights, sc.rule
    T = range(st.T)                                   # periods
    k = st.eta * st.tau * st.w                        # energy charged in one period, kWh
    J = [q for q in inst.requests if q.D > 1e-9]      # requests
    P = {q.j: list(range(q.r, q.d)) for q in J}       # periods in which vehicle j is present
    Rl = {q.j: list(range(q.r + 1, q.d + 1)) for q in J}   # possible release periods
    cap = {t: max(0.0, inst.PT + inst.PV[t] - inst.B[t]) for t in T}   # power available for charging
    L0 = {(q.j, t): inst.L0(q.j, t) for q in J for t in P[q.j]}       # release period after a reminder in the non-fee state
    LF = {(q.j, t): inst.LF(q.j, t) for q in J for t in P[q.j]}       # release period after a reminder in the fee state
    Wmax = st.W_max                                   # maximum waiting periods

    env = cp.Envr()
    m = env.createModel('offline')
    m.setParam(COPT.Param.TimeLimit, time_limit)
    m.setParam(COPT.Param.RelGap, gap)
    m.setParam(COPT.Param.Logging, 1 if verbose else 0)

    jt = [(q.j, t) for q in J for t in P[q.j]]
    jr = [(q.j, t) for q in J for t in Rl[q.j]]
    B = COPT.BINARY

    # ---------------- decision variables ----------------
    a = m.addVars([q.j for q in J], vtype=B, nameprefix='a')       # admission
    f = m.addVars([q.j for q in J], vtype=B, nameprefix='f')       # target reached
    s = m.addVars(jt, vtype=B, nameprefix='s')                    # plug-in period
    pi = m.addVars(jr, vtype=B, nameprefix='pi')                  # release period
    u = m.addVars(jt, vtype=B, nameprefix='u')                    # occupancy
    x = m.addVars(jt, vtype=B, nameprefix='x')                    # charging
    h = m.addVars(jt, vtype=B, nameprefix='h')                    # target flag
    v = m.addVars(jt, vtype=B, nameprefix='v')                    # reminder
    e = m.addVars(list(T), lb=0.0, nameprefix='e')                # grid purchase, kWh
    if rc.rule == 'congestion':
        kap = m.addVars(list(T), vtype=B, nameprefix='kappa')     # fee state
        xi = m.addVars(jt, vtype=B, nameprefix='xi')              # ξ = v·κ
    else:
        kap = {t: (1 if rc.rule == 'always' else 0) for t in T}
        xi = v if rc.rule == 'always' else {key: 0 for key in jt}

    # waiting q_jt is an expression: accepted but not yet plugged in
    def S_cum(j, t):
        return cp.quicksum(s[j, tt] for tt in P[j] if tt <= t)

    def Pi_cum(j, t):
        return cp.quicksum(pi[j, tt] for tt in Rl[j] if tt <= t)

    q_ = {(j, t): a[j] - S_cum(j, t) for (j, t) in jt}

    for q in J:
        j = q.j
        # (2) an accepted vehicle plugs in once and is released once
        m.addConstr(cp.quicksum(s[j, t] for t in P[j]) == a[j], name=f'c2s_{j}')
        m.addConstr(cp.quicksum(pi[j, t] for t in Rl[j]) == a[j], name=f'c2p_{j}')
        for t in P[j]:
            # (3) occupancy = plugged in - released
            m.addConstr(u[j, t] == S_cum(j, t) - Pi_cum(j, t), name=f'c3_{j}_{t}')
            # (7) only plugged-in vehicles can be charged
            m.addConstr(x[j, t] <= u[j, t], name=f'c7_{j}_{t}')
        # (6) maximum waiting time per vehicle
        m.addConstr(cp.quicksum(q_[j, t] for t in P[j]) <= Wmax, name=f'c6w_{j}')
        Ej = k * cp.quicksum(x[j, t] for t in P[j])
        # (8) state of charge at most the battery capacity
        m.addConstr(q.soc0 + Ej * (1.0 / q.b) <= 1.0, name=f'c8_{j}')
        # (9) minimum charged energy
        m.addConstr(Ej >= wc.mu * q.D * a[j], name=f'c9_{j}')
        # (10) target flag
        m.addConstr(q.socd * f[j] <= q.soc0 + Ej * (1.0 / q.b), name=f'c10_{j}')
        m.addConstr(f[j] <= a[j], name=f'c10a_{j}')
        for t in P[j]:
            # (11) h_jt = 1 only if D_j has been charged before t
            m.addConstr(q.D * h[j, t] <= k * cp.quicksum(x[j, tt] for tt in P[j] if tt < t), name=f'c11_{j}_{t}')
            # (12) only plugged-in vehicles that reached their targets can be reminded
            m.addConstr(v[j, t] <= h[j, t], name=f'c12h_{j}_{t}')
            m.addConstr(v[j, t] <= u[j, t], name=f'c12u_{j}_{t}')
            # (14) linearization of xi = v * kappa
            if rc.rule == 'congestion':
                m.addConstr(xi[j, t] <= v[j, t])
                m.addConstr(xi[j, t] <= kap[t])
                m.addConstr(xi[j, t] >= v[j, t] + kap[t] - 1)
        # (12) at most one reminder per vehicle
        m.addConstr(cp.quicksum(v[j, t] for t in P[j]) <= a[j], name=f'c12n_{j}')
        # (13) release period: d_j without a reminder, otherwise L0 or LF depending on the fee state
        m.addConstr(cp.quicksum(t * pi[j, t] for t in Rl[j]) ==
                    q.d * (a[j] - cp.quicksum(v[j, t] for t in P[j]))
                    + cp.quicksum(L0[j, t] * v[j, t] - (L0[j, t] - LF[j, t]) * xi[j, t] for t in P[j]),
                    name=f'c13_{j}')

    Mbig = st.m + 1
    for t in T:
        here = [q.j for q in J if t in P[q.j]]
        U_t = cp.quicksum(u[j, t] for j in here)
        X_t = cp.quicksum(x[j, t] for j in here)
        m.addConstr(U_t <= st.m, name=f'c4_{t}')                                       # (4) number of chargers
        m.addConstr(cp.quicksum(q_[j, t] for j in here) <= st.K, name=f'c6k_{t}')     # (6) waiting spaces
        if rc.rule == 'congestion':                                                   # (15)(16) fee state
            m.addConstr(st.m - U_t - rc.theta <= Mbig * (1 - kap[t]), name=f'c15_{t}')
            m.addConstr(st.m - U_t - rc.theta >= 1 - Mbig * kap[t], name=f'c16_{t}')
        m.addConstr(st.w * X_t <= cap[t], name=f'c17_{t}')                            # (17) transformer capacity
        m.addConstr(e[t] >= st.tau * (st.w * X_t - inst.PV[t]), name=f'c18_{t}')      # (18) grid purchase

    # ---------------- objective (1) ----------------
    obj = (wc.alpha * cp.quicksum(a[q.j] for q in J)
           + wc.beta * cp.quicksum(f[q.j] for q in J)
           - wc.omega * cp.quicksum(inst.pr[t] * e[t] for t in T)
           - wc.lam * cp.quicksum(q_[j, t] for (j, t) in jt)
           - cp.quicksum(inst.cN[t] * v[j, t] for (j, t) in jt))
    m.setObjective(obj, COPT.MAXIMIZE)
    m.solve()

    res = dict(status=m.status, n_vars=m.getAttr(COPT.Attr.Cols), n_cons=m.getAttr(COPT.Attr.Rows))
    if not m.hasmipsol:
        return res, None
    res.update(obj=m.objval, bound=m.bestbnd)
    val = lambda z: z if isinstance(z, (int, float)) else z.x
    sched = []
    for q in J:
        j = q.j
        if val(a[j]) < 0.5:
            sched.append(dict(j=j, r=q.r, d=q.d, accepted=0))
            continue
        plug = next(t for t in P[j] if s[j, t].x > 0.5)
        rel = next(t for t in Rl[j] if pi[j, t].x > 0.5)
        ch = [t for t in P[j] if x[j, t].x > 0.5]
        rem = [t for t in P[j] if v[j, t].x > 0.5]
        sched.append(dict(j=j, r=q.r, d=q.d, accepted=1, plug=plug, release=rel,
                          charge_periods=ch, remind=rem[0] if rem else None,
                          kappa_at_remind=(val(kap[rem[0]]) if rem else None),
                          full=int(val(f[j]) > 0.5)))
    return res, sched


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--small', action='store_true', help='toy size for checking')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--rule', default='congestion', choices=['none', 'always', 'congestion'])
    ap.add_argument('--theta', type=int, default=0)
    ap.add_argument('--time', type=float, default=1800)
    a = ap.parse_args(argv)
    sc = Scenario()
    if a.small:
        from ev_overstay.run_experiments import preset
        sc = preset('small')
    sc.rule.rule, sc.rule.theta = a.rule, a.theta
    inst = generate(sc, a.seed)
    res, sched = build_and_solve(inst, time_limit=a.time)
    print('\nResult:', res)
    if sched:
        clk = lambda t: (lambda h: f'{int(h):02d}:{int(round((h % 1) * 60)):02d}')(sc.station.clock(t))
        print(f"{'veh':>4} {'arrive':>6} {'depart':>6} {'acc':>4} {'plug':>6} {'release':>6} {'remind':>6} {'fee':>4} {'tgt':>4} charging periods")
        for r in sched:
            if not r['accepted']:
                print(f"{r['j']:>4} {clk(r['r']):>6} {clk(r['d']):>6} {'no':>4}")
                continue
            rem = clk(r['remind']) if r['remind'] is not None else '-'
            kp = str(int(round(r['kappa_at_remind']))) if r['remind'] is not None else '-'
            print(f"{r['j']:>4} {clk(r['r']):>6} {clk(r['d']):>6} {'yes':>4} {clk(r['plug']):>6} "
                  f"{clk(r['release']):>6} {rem:>6} {kp:>4} {r['full']:>4} {len(r['charge_periods'])}")


if __name__ == '__main__':
    main()
