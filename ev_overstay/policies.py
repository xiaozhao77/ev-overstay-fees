"""
Online methods (Section 5 of the paper): B0, B1, RH, RH-N, and the variants of Table B3.

Interface:
  admit(sim, t, arrivals) -> set(j)          step 3: admission decisions for new requests in this period
  operate(sim, t) -> (charge_list, remind_list)   steps 4 and 6: charging and reminder decisions
The charging list is ordered by priority and truncated by the simulator under the transformer limit.
"""
import time
import copy
import numpy as np

from . import behavior as bh
from .model import ChargingMILP, VehWin, WindowData


# ---------------- shared functions ----------------
def est_release(sim, x, t, project=False):
    """Estimated release period R-hat: min(d, n + delay quantile) for reminded vehicles;
    d for unreminded vehicles (forecast with remind-when-full if project=True); overdue estimates move to t+1."""
    sc = sim.sc
    if x.reminded:
        dq = bh.quantile_delay(x.n_rem, x.d, x.kappa_rem, sc.solver.quantile_p,
                               sc.station, sc.behavior, sc.rule)
        R = min(x.d, x.n_rem + dq)
    elif project:
        # remind-when-full forecast: reminder after the remaining charging periods, plus the delay quantile (fee state at the reminder estimated by the current state)
        need = int(np.ceil(max(0.0, x.D - x.E) / sim.k - 1e-9))
        n_hat = min(x.d - 1, max(t, (x.t_full if x.t_full is not None else t + need)))
        dq = bh.quantile_delay(n_hat, x.d, sim.kappa_now(), sc.solver.quantile_p,
                               sc.station, sc.behavior, sc.rule)
        R = min(x.d, n_hat + dq)
    else:
        R = x.d
    return max(R, t + 1)


def remind_when_full(sim, t):
    return [j for j in sim.plugged if not sim.v[j].reminded
            and sim.v[j].t_full is not None and sim.v[j].t_full <= t]


class Policy:
    name = 'base'

    def admit(self, sim, t, arrivals):
        raise NotImplementedError

    def operate(self, sim, t):
        raise NotImplementedError


# ---------------- B0: first-come, first-served ----------------
class B0(Policy):
    name = 'B0'

    def admit(self, sim, t, arrivals):
        return {x.j for x in arrivals}          # rejected automatically by the simulator when the waiting area is full

    def operate(self, sim, t):
        cand = [j for j in sim.plugged if not sim.v[j].full]
        cand.sort(key=lambda j: (sim.v[j].d, j))  # earliest declared departure first
        return cand, remind_when_full(sim, t)


# ---------------- B1: feasibility check ----------------
class B1(Policy):
    name = 'B1'

    def admit(self, sim, t, arrivals):
        st = sim.sc.station
        free = sim.free()
        qlen = len(sim.queue)
        Rhat = sorted(est_release(sim, sim.v[j], t, sim.sc.solver.b1_project) for j in sim.plugged)
        out = set()
        for x in arrivals:
            if free > 0:
                s_hat = t
            elif qlen < st.K and qlen < len(Rhat):
                s_hat = Rhat[qlen]
            else:
                continue
            if sim.k * (x.d - s_hat) >= x.D - 1e-6:       # condition (27)
                out.add(x.j)
                if free > 0:
                    free -= 1
                    Rhat.append(x.d)
                    Rhat.sort()
                else:
                    qlen += 1
        return out

    def operate(self, sim, t):
        cand = [j for j in sim.plugged if not sim.v[j].full]

        def urgency(j):
            x = sim.v[j]
            need = np.ceil((x.D - x.E) / sim.k - 1e-9)
            return need / max(1, x.d - t)
        cand.sort(key=lambda j: (-urgency(j), sim.v[j].d, j))
        return cand, remind_when_full(sim, t)


# ---------------- expected scenario of future requests ----------------
_FC_CACHE = {}


def forecast_table(sc):
    """Estimate from historical samples the expected number of requests per arrival period and the mean departure period and required energy
    of vehicles arriving in that period. Online methods use only this distribution, not the actual requests of the night."""
    from .instance import sample_requests
    st, dc = sc.station, sc.demand
    key = (st.start_hour, st.horizon_hours, st.tau, dc.n, dc.arr_mean, dc.arr_sd, dc.dep_mean,
           dc.dep_sd, tuple(dc.batteries), dc.battery_csv, dc.soc0_mean, dc.soc0_sd, dc.soc0_low, dc.soc0_high, dc.socd, dc.csv)
    if key in _FC_CACHE:
        return _FC_CACHE[key]
    sc2 = copy.deepcopy(sc)
    sc2.demand.n = sc.solver.forecast_pool
    reqs = sample_requests(sc2, np.random.default_rng(20240601))
    T = st.T
    cnt = np.zeros(T)
    dsum = np.zeros(T)
    Dsum = np.zeros(T)
    bsum = np.zeros(T)
    for q in reqs:
        cnt[q.r] += 1
        dsum[q.r] += q.d
        Dsum[q.r] += q.D
        bsum[q.r] += q.b
    lam = cnt / len(reqs) * dc.n                 # expected arrivals per period
    with np.errstate(invalid='ignore', divide='ignore'):
        dbar = np.where(cnt > 0, dsum / np.maximum(cnt, 1), T)
        Dbar = np.where(cnt > 0, Dsum / np.maximum(cnt, 1), 0)
        bbar = np.where(cnt > 0, bsum / np.maximum(cnt, 1), np.mean(dc.batteries))
    _FC_CACHE[key] = (lam, dbar, Dbar, bbar)
    return _FC_CACHE[key]


def pseudo_requests(sc, t):
    """Accumulate expected arrivals after t and round them into representative future requests (r, d, b, D)."""
    lam, dbar, Dbar, bbar = forecast_table(sc)
    out, acc, emitted = [], 0.0, 0
    r_max = min(sc.station.T - 1, t + 1 + int(round(sc.solver.forecast_hours / sc.station.tau)))
    for r in range(t + 1, r_max):
        acc += lam[r]
        while acc - emitted >= 0.5:
            emitted += 1
            d = int(max(r + 1, min(sc.station.T, round(dbar[r]))))
            out.append((r, d, float(bbar[r]), float(Dbar[r])))
    return out


# ---------------- RH / RH-N: rolling horizon ----------------
class RH(Policy):
    name = 'RH'
    force_remind = False

    fair = 0.0                     # household priority (used by RH-F only)

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.fallback = B1()
        self._mdl = None
        self._t = None
        self.n_fail = 0

    def _window(self, sim, t, arrivals):
        sc, inst = sim.sc, sim.inst
        st, bc, rc = sc.station, sc.behavior, sc.rule
        p = sc.solver.quantile_p

        buf = int(round(getattr(sc.solver, 'rh_dep_buffer_h', 0.0) / st.tau))

        def dplan(x):   # planning departure: declared departure minus the safety margin, at least one period ahead
            return x.d if buf <= 0 else max(min(x.d, t + 1), x.d - buf)

        def lmaps(x, dd=None):
            dd = dplan(x) if dd is None else dd
            L0, LF = {}, {}
            for tt in range(max(t, getattr(x, 'start', t)), dd):
                L0[tt] = min(dd, tt + bh.quantile_delay(tt, dd, 0, p, st, bc, rc))
                LF[tt] = min(dd, tt + bh.quantile_delay(tt, dd, 1, p, st, bc, rc))
            return L0, LF

        vehs = []
        for j in sim.plugged:
            x = sim.v[j]
            vw = VehWin(j, 'plugged', t, dplan(x), x.b, x.soc, x.socd, x.D, E_done=x.E,
                        reminded=x.reminded)
            if x.reminded:
                vw.R_fixed = est_release(sim, x, t)
            else:
                vw.L0, vw.LF = lmaps(x)
            vehs.append(vw)
        for j in sim.queue:
            x = sim.v[j]
            vw = VehWin(j, 'queued', t, dplan(x), x.b, x.soc, x.socd, x.D,
                        wait_left=max(0, st.W_max - x.q_periods))
            vw.L0, vw.LF = lmaps(x)
            vehs.append(vw)
        for x in arrivals:
            vw = VehWin(x.j, 'new', t, dplan(x), x.b, x.soc, x.socd, x.D, wait_left=st.W_max,
                        prio=getattr(x, 'prio', 0.0) * getattr(self, 'fair', 1.0))
            vw.L0, vw.LF = lmaps(x)
            vehs.append(vw)
        if not vehs:
            return None
        self._lmaps = lmaps
        if sc.solver.rh_forecast and not getattr(self, 'multi_scenario', False):
            for i, (r, d, b, D) in enumerate(pseudo_requests(sc, t)):
                if D <= 1e-9:
                    continue
                d2 = d if buf <= 0 else max(r + 1, d - buf)
                vw = VehWin(-1 - i, 'pseudo', r, d2, b, sc.demand.socd - D / b, sc.demand.socd, D,
                            wait_left=st.W_max, weight=sc.solver.forecast_weight)
                vw.L0, vw.LF = lmaps(vw, d2)
                vehs.append(vw)
        t_end = max(v.d for v in vehs)
        cap, PV = {}, {}
        for tt in range(t, t_end):
            pv = inst.PV[tt] if (tt == t or not sc.solver.pv_future_zero) else 0.0
            PV[tt] = pv
            cap[tt] = max(0.0, inst.PT + pv - inst.B[tt])
        return WindowData(t, t_end, vehs, cap, PV,
                          {tt: inst.pr[tt] for tt in range(t, t_end)},
                          {tt: inst.cN[tt] for tt in range(t, t_end)},
                          free_now=sim.free())

    def _solve(self, sim, t, arrivals):
        self._t, self._mdl = t, None
        wd = self._window(sim, t, arrivals)
        if wd is None:
            return
        t0 = time.time()
        try:
            old_eps = sim.sc.solver.tie_eps
            sim.sc.solver.tie_eps = getattr(self, 'tie', old_eps)
            mdl = ChargingMILP(sim.sc, wd, force_remind=self.force_remind,
                               time_limit=sim.sc.solver.rh_time_limit, verbose=self.verbose,
                               mip_gap=sim.sc.solver.rh_mip_gap)
            ok = mdl.solve()
            sim.sc.solver.tie_eps = old_eps
        except Exception as ex:          # e.g. the size limit of a restricted licence
            if self.n_fail == 0:
                print(f'[{self.name}] t={t} solve failed, falling back to the B1 rule: {ex}')
            ok = False
        sim.solve_times.append(time.time() - t0)
        if ok:
            self._mdl = mdl
        else:
            self.n_fail += 1

    def admit(self, sim, t, arrivals):
        self._solve(sim, t, arrivals)
        if self._mdl is None:
            return self.fallback.admit(sim, t, arrivals)
        a = self._mdl.a
        return {x.j for x in arrivals if x.j in a and self._mdl.val(a[x.j]) > 0.5}

    def operate(self, sim, t):
        if self._t != t:                 # no new request in this period: solve here
            self._solve(sim, t, [])
        mdl = self._mdl
        if mdl is None:
            if not sim.plugged:
                return [], []
            return self.fallback.operate(sim, t)
        charge, remind = [], []
        for j in sim.plugged:
            xv = mdl.x.get(j, {}).get(t)
            if xv is not None and mdl.val(xv) > 0.5:
                charge.append(j)
            vv = mdl.v.get(j, {}).get(t)
            if vv is not None and mdl.val(vv) > 0.5:
                remind.append(j)
        # cases not covered by the model (rare): fill remaining power by B1 urgency
        rest, _ = self.fallback.operate(sim, t)
        charge += [j for j in rest if j not in charge and j not in mdl.x]
        return charge, remind


class RHN(RH):
    name = 'RH-N'
    force_remind = True


class RHTB(RH):
    """Tie-breaking variant: when the window model is indifferent about a zero-cost reminder, remind at once (earlier release)."""
    name = 'RH-TB'

    def operate(self, sim, t):
        charge, remind = RH.operate(self, sim, t)
        if sim.inst.cN[t] <= 0:
            extra = [j for j in remind_when_full(sim, t) if j not in remind]
            remind = list(remind) + extra
        return charge, remind


class B1S(B1):
    """Feasibility check with a slack factor; sigma > 1 is more optimistic"""
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = f'B1s{sigma}'

    def admit(self, sim, t, arrivals):
        k0 = sim.k
        sim.k = k0 * self.sigma
        try:
            return B1.admit(self, sim, t, arrivals)
        finally:
            sim.k = k0


# ---------------- multi-scenario rolling horizon (RH-MS, Appendix B.12) ----------------
_POOL = {}


def _pool_by_arrival(sc):
    """Group historical samples by arrival period for sampling future-request scenarios."""
    from .instance import sample_requests
    st, dc = sc.station, sc.demand
    key = (st.tau, st.horizon_hours, dc.n, dc.arr_mean, dc.arr_sd, dc.dep_mean, dc.dep_sd, dc.battery_csv, dc.soc0_mean, dc.soc0_sd)
    if key not in _POOL:
        sc2 = copy.deepcopy(sc)
        sc2.demand.n = sc.solver.forecast_pool
        reqs = sample_requests(sc2, np.random.default_rng(20240601))
        by_r = {}
        for q in reqs:
            by_r.setdefault(q.r, []).append((q.d, q.b, q.D))
        _POOL[key] = by_r
    return _POOL[key]


class RHMS(RH):
    """Multi-scenario rolling horizon: each step samples S future-request scenarios, copies the window model for each,
    requires the current decisions (admission, plug-in, charging, reminders) to agree across scenarios, and maximizes the average (sample average approximation)."""
    name = 'RH-MS'
    multi_scenario = True

    def _solve(self, sim, t, arrivals):
        from . import lp
        from .model import ChargingMILP
        self._t, self._mdl = t, None
        wd0 = self._window(sim, t, arrivals)
        if wd0 is None:
            return
        sc, st = sim.sc, sim.sc.station
        S = max(1, int(sc.solver.rh_scen))
        lam, _, _, _ = forecast_table(sc)
        pool = _pool_by_arrival(sc)
        r_max = min(st.T - 1, t + 1 + int(round(sc.solver.forecast_hours / st.tau)))
        rng = np.random.default_rng([int(sim.inst.seed), int(t), 7919])
        M = lp.Model('charging-ms')
        M.time_limit = sc.solver.rh_time_limit
        M.mip_gap = sc.solver.rh_mip_gap
        M.threads = sc.solver.threads
        t0 = time.time()
        mdls = []
        try:
            for s in range(S):
                vehs = list(wd0.vehicles)
                k = 0
                for r in range(t + 1, r_max):
                    cands = pool.get(r, [])
                    for _ in range(int(rng.poisson(lam[r])) if cands else 0):
                        d, b, D = cands[int(rng.integers(len(cands)))]
                        if D <= 1e-9:
                            continue
                        k += 1
                        vw = VehWin(-1000 * (s + 1) - k, 'pseudo', r, d, b, sc.demand.socd - D / b, sc.demand.socd, D,
                                    wait_left=st.W_max, weight=1.0)
                        vw.L0, vw.LF = self._lmaps(vw)
                        vehs.append(vw)
                t_end = max(v.d for v in vehs)
                cap = {tt: wd0.cap.get(tt, max(0.0, sim.inst.PT - sim.inst.B[tt])) for tt in range(t, t_end)}
                PV = {tt: wd0.PV.get(tt, 0.0) for tt in range(t, t_end)}
                wd = WindowData(t, t_end, vehs, cap, PV, {tt: sim.inst.pr[tt] for tt in range(t, t_end)},
                                {tt: sim.inst.cN[tt] for tt in range(t, t_end)}, free_now=wd0.free_now)
                mdls.append(ChargingMILP(sc, wd, force_remind=self.force_remind, m_ext=M, tag=f's{s}_', set_obj=False))
            base = mdls[0]
            for md in mdls[1:]:
                for j, a in base.a.items():
                    if j >= 0 and not isinstance(a, int) and j in md.a:
                        M.addConstr(md.a[j] == a)
                for dname in ('x', 'v', 's'):
                    for j, dd in getattr(base, dname).items():
                        if j < 0 or t not in dd or j not in getattr(md, dname) or t not in getattr(md, dname)[j]:
                            continue
                        M.addConstr(getattr(md, dname)[j][t] == dd[t])
            obj = lp.Expr()
            for md in mdls:
                obj += md.obj * (1.0 / S)
            M.setObjective(obj, maximize=True)
            ok = M.optimize(sc.solver.solver)
        except Exception as ex:
            if self.n_fail == 0:
                print(f'[{self.name}] t={t} solve failed, falling back to the B1 rule: {ex}')
            ok = False
        sim.solve_times.append(time.time() - t0)
        if ok:
            self._mdl = mdls[0]
        else:
            self.n_fail += 1


class RHF(RH):
    """Multi-night fairness (Appendix B.10): households rejected on earlier nights receive a higher admission reward."""
    name = 'RH-F'
    fair = 1.0



# ---------------- tariff-aware rule-based baseline B0-TOU ----------------
class B0T(B0):
    """First-come, first-served admission; charging is deferred to off-peak hours while no vehicle waits (unless the remaining off-peak time is too short)
    and starts at once while vehicles wait; earliest departure first; remind at full charge, but late-night reminders are postponed to 7 a.m."""
    name = 'B0-TOU'

    def operate(self, sim, t):
        inst = sim.inst
        pmin = min(inst.pr)
        cand = []
        for j in sim.plugged:
            x = sim.v[j]
            if x.full:
                continue
            need = int(np.ceil(max(0.0, x.D - x.E) / sim.k - 1e-9))
            off_left = sum(1 for tt in range(t, x.d) if inst.pr[tt] <= pmin + 1e-9)
            if inst.pr[t] <= pmin + 1e-9 or off_left < need or sim.queue:   # no deferral while vehicles wait, so that chargers are not held idle
                cand.append(j)
        cand.sort(key=lambda j: (sim.v[j].d, j))
        st = sim.sc.station
        if bh.is_night(st.clock(t), sim.sc.behavior):
            rem = []
        else:
            rem = remind_when_full(sim, t)
        return cand, rem


class B0N(B0):
    """B0 charging (charge on arrival, earliest departure first), with late-night reminders postponed to 7 a.m."""
    name = 'B0-NR'

    def operate(self, sim, t):
        cand, rem = B0.operate(self, sim, t)
        if bh.is_night(sim.sc.station.clock(t), sim.sc.behavior):
            rem = []
        return cand, rem


class B0TP(B0):
    """Stronger tariff-aware baseline: first-come, first-served admission; charging is deferred to off-peak hours while no vehicle waits and a charger is idle
    (unless the remaining off-peak time is too short), otherwise earliest departure first at once; remind at full charge in the day period, late at night only while a vehicle waits, otherwise at 7 a.m."""
    name = 'B0-TOU+'

    def operate(self, sim, t):
        inst = sim.inst
        pmin = min(inst.pr)
        busy = bool(sim.queue) or sim.free() <= 0
        cand = []
        for j in sim.plugged:
            x = sim.v[j]
            if x.full:
                continue
            need = int(np.ceil(max(0.0, x.D - x.E) / sim.k - 1e-9))
            off_left = sum(1 for tt in range(t, x.d) if inst.pr[tt] <= pmin + 1e-9)
            if busy or inst.pr[t] <= pmin + 1e-9 or off_left < need:
                cand.append(j)
        cand.sort(key=lambda j: (sim.v[j].d, j))
        st = sim.sc.station
        if bh.is_night(st.clock(t), sim.sc.behavior) and not sim.queue:
            rem = []
        else:
            rem = remind_when_full(sim, t)
        return cand, rem


class B0Q(B0):
    """Reminder rule derived from Proposition 1(i): B0 charging, late-night reminders only while a vehicle waits, otherwise at 7 a.m."""
    name = 'B0-Q'

    def operate(self, sim, t):
        cand, rem = B0.operate(self, sim, t)
        if bh.is_night(sim.sc.station.clock(t), sim.sc.behavior) and not sim.queue:
            rem = []
        return cand, rem


class RHA(RH):
    """RH with first-come, first-served admission: every request that fits is accepted, while charging and reminders are optimized (Section 6.6)."""
    name = 'RH-A'

    def _solve(self, sim, t, arrivals):
        old = getattr(sim.sc.solver, 'rh_accept_bonus', 0.0)
        sim.sc.solver.rh_accept_bonus = 1000.0
        try:
            super()._solve(sim, t, arrivals)
        finally:
            sim.sc.solver.rh_accept_bonus = old

    def admit(self, sim, t, arrivals):
        self._solve(sim, t, arrivals)
        return {x.j for x in arrivals}          # rejected by the simulator only when the waiting area is full


POLICIES = {'B0': B0, 'B0-TOU': B0T, 'B0-NR': B0N, 'B0-TOU+': B0TP, 'B0-Q': B0Q, 'RH-A': RHA, 'B1': B1, 'RH': RH, 'RH-N': RHN, 'RH-TB': RHTB, 'RH-MS': RHMS, 'RH-F': RHF}


def make_policy(name, **kw):
    if name.startswith('B1s'):
        return B1S(float(name[3:]))
    return POLICIES[name](**kw) if name.startswith('RH') else POLICIES[name]()
