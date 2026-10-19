"""
Mixed-integer linear program (Section 4 of the paper; constraint numbers follow the paper).

The same builder is used for
  * the offline model: the window starts at t0 = 0, all vehicles are new requests, and realized move-out delays are used;
  * the rolling-horizon window model: the window starts at the current period, implemented decisions are fixed, and delays are estimated by quantiles.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from . import lp
from .lp import quicksum

from .config import Scenario


@dataclass
class VehWin:
    j: int
    status: str               # 'new' | 'queued' | 'plugged' | 'pseudo' (forecast future request)
    start: int                # earliest period in the window at which the vehicle can occupy a charger
    d: int                    # declared departure period
    b: float
    soc_now: float            # state of charge at the start of the window
    socd: float
    D: float                  # total required energy, kWh
    E_done: float = 0.0       # energy already charged, kWh
    reminded: bool = False
    R_fixed: Optional[int] = None    # (estimated) release period of a reminded vehicle
    wait_left: int = 10 ** 6         # remaining waiting periods
    L0: Dict[int, int] = field(default_factory=dict)   # release period after a reminder at t in the non-fee state
    LF: Dict[int, int] = field(default_factory=dict)   # release period after a reminder at t in the fee state
    weight: float = 1.0       # reward weight: < 1 for forecast requests, reflecting their uncertainty
    prio: float = 0.0         # multi-night fairness: admission reward multiplier, alpha_j = alpha(1 + prio)


@dataclass
class WindowData:
    t0: int
    t_end: int                # end of the window (exclusive)
    vehicles: List[VehWin]
    cap: Dict[int, float]     # power available for charging in each period = max(0, PT + PV_t - B_t)
    PV: Dict[int, float]
    pr: Dict[int, float]
    cN: Dict[int, float]
    free_now: Optional[int] = None   # online window: idle chargers in the current period after plug-in (newly accepted vehicles plug in if a charger is idle)


class ChargingMILP:
    def __init__(self, sc: Scenario, wd: WindowData, force_remind: bool = False,
                 time_limit: float = 60.0, verbose: bool = False, mip_gap: Optional[float] = None,
                 m_ext=None, tag: str = '', set_obj: bool = True):
        self.sc, self.wd = sc, wd
        st, wc, rc = sc.station, sc.weights, sc.rule
        if m_ext is None:   # multi-scenario model: scenarios share one solver model (m_ext), variable names prefixed by tag
            m_ = lp.Model('charging')
            m_.verbose = verbose
            m_.time_limit = time_limit
            m_.mip_gap = sc.solver.mip_gap if mip_gap is None else mip_gap
            m_.threads = sc.solver.threads
        else:
            m_ = m_ext
        self.m = m_
        _add = m_.addVar
        def _addVar(vtype='C', lb=0.0, ub=None, name=''):
            return _add(vtype=vtype, lb=lb, ub=ub, name=tag + name)
        m_ = _Proxy(m_, _addVar)
        k = st.eta * st.tau * st.w        # energy charged in one period, kWh
        periods = range(wd.t0, wd.t_end)

        self.a, self.z, self.s, self.pi, self.u, self.x = {}, {}, {}, {}, {}, {}
        self.aeff = {}
        self.h, self.v, self.xi, self.f, self.q = {}, {}, {}, {}, {}
        new_now = []      # (a_j, s_j,t0): new requests in the current period
        obj = lp.Expr()
        U_t = {t: lp.Expr() for t in periods}   # Σ_j u_jt
        X_t = {t: lp.Expr() for t in periods}   # Σ_j x_jt
        Q_t = {t: lp.Expr() for t in periods}   # Σ_j q_jt
        new_rank = {vh.j: i for i, vh in enumerate(sorted([v for v in wd.vehicles if v.status == 'new'], key=lambda v: v.j))}
        T_w = max(1, wd.t_end - wd.t0)

        # fee state kappa_t
        kap = {}
        for t in periods:
            if rc.rule == 'none':
                kap[t] = 0
            elif rc.rule == 'always':
                kap[t] = 1
            else:
                kap[t] = m_.addVar(vtype='B', name=f'kappa[{t}]')
        self.kappa = kap

        for vh in wd.vehicles:
            j = vh.j
            P = [t for t in range(vh.start, vh.d) if wd.t0 <= t < wd.t_end]
            if not P:
                continue
            Rel = list(range(P[0] + 1, vh.d + 1))
            # ---- admission a_j and slack z_j for failing to plug in ----
            if vh.status in ('new', 'pseudo'):
                a = m_.addVar(vtype='B', name=f'a[{j}]')
                obj += vh.weight * wc.alpha * (1.0 + vh.prio) * a
                if vh.status == 'new' and getattr(sc.solver, 'rh_accept_bonus', 0.0) > 0:   # RH-A: accept every request that fits (not counted in the reward)
                    obj += sc.solver.rh_accept_bonus * a
                if sc.solver.tb_lex > 0 and vh.status == 'new':   # deterministic tie-breaking: prefer earlier requests among equal optima
                    obj += sc.solver.tb_lex * (1.0 - 0.5 * new_rank.get(j, 0) / max(1, len(new_rank))) * a
                a_eff = a
            elif vh.status == 'queued':
                a = 1
                z = m_.addVar(vtype='B', name=f'z[{j}]')
                self.z[j] = z
                obj += -1000.0 * z
                a_eff = 1 - z
            else:
                a = 1
                a_eff = 1
            self.a[j] = a
            self.aeff[j] = a_eff
            # ---- plug-in s and release pi ----
            s = {}
            if vh.status in ('new', 'queued', 'pseudo'):
                for t in P:
                    s[t] = m_.addVar(vtype='B', name=f's[{j},{t}]')
                m_.addConstr(quicksum(s.values()) == a_eff)                         # (2)
            pi = {t: m_.addVar(vtype='B', name=f'pi[{j},{t}]') for t in Rel}
            m_.addConstr(quicksum(pi.values()) == a_eff)                             # (2)
            self.s[j], self.pi[j] = s, pi
            if vh.status == 'new' and wd.t0 in s:
                new_now.append((a, s[wd.t0]))
            # ---- occupancy u, waiting q, charging x ----
            u, x, q = {}, {}, {}
            cum_s, cum_pi = lp.Expr(), lp.Expr()
            for t in P:
                if t in s:
                    cum_s += s[t]
                if t in pi:
                    cum_pi += pi[t]
                u[t] = m_.addVar(vtype='B', name=f'u[{j},{t}]')
                occupied = (cum_s if vh.status != 'plugged' else 1)
                m_.addConstr(u[t] == occupied - cum_pi)                                  # (3)
                x[t] = m_.addVar(vtype='B', name=f'x[{j},{t}]')
                m_.addConstr(x[t] <= u[t])                                               # (7)
                if sc.solver.tb_lex > 0 and vh.status != 'pseudo':   # deterministic tie-breaking: prefer earlier charging among equal optima
                    obj += -sc.solver.tb_lex * (t - wd.t0) / T_w * x[t]
                U_t[t] += u[t]
                X_t[t] += x[t]
                if vh.status != 'plugged':
                    q[t] = a_eff - cum_s                                                 # (5)
                    Q_t[t] += q[t]
                    obj += -wc.lam * q[t]
            if q:
                m_.addConstr(quicksum(q.values()) <= vh.wait_left)                   # (6)
            self.u[j], self.x[j], self.q[j] = u, x, q
            # ---- state of charge ----
            Ew = k * quicksum(x.values())
            m_.addConstr(vh.soc_now + Ew / vh.b <= 1.0)                                  # (8)
            if sc.solver.cap_target:   # stop charging at the target (PV scenario)
                m_.addConstr(vh.E_done + Ew <= vh.D + k - 1e-6)
            if vh.status in ('new', 'pseudo'):
                m_.addConstr(vh.E_done + Ew >= wc.mu * vh.D * a_eff)                     # (9)
            else:
                sl = m_.addVar(lb=0.0, name=f'slackE[{j}]')
                m_.addConstr(vh.E_done + Ew + sl >= wc.mu * vh.D * a_eff)                # (9), softened
                obj += -100.0 * sl
            f = m_.addVar(vtype='B', name=f'f[{j}]')
            m_.addConstr(vh.socd * f <= vh.soc_now + Ew / vh.b)                          # (10)
            m_.addConstr(f <= a_eff)
            obj += vh.weight * wc.beta * f
            self.f[j] = f
            # ---- target flag h and reminder v ----
            if vh.reminded:
                rf = min(max(vh.R_fixed, P[0] + 1), vh.d)
                for t in Rel:
                    pi[t].ub = pi[t].lb = 1.0 if t == rf else 0.0
                continue
            h, v, xi = {}, {}, {}
            cum_x_before = lp.Expr()
            cum_v_before = lp.Expr()
            for t in P:
                h[t] = m_.addVar(vtype='B', name=f'h[{j},{t}]')
                m_.addConstr(vh.D * h[t] <= vh.E_done + k * cum_x_before)               # (11)
                v[t] = m_.addVar(vtype='B', name=f'v[{j},{t}]')
                m_.addConstr(v[t] <= h[t])                                               # (12)
                m_.addConstr(v[t] <= u[t])                                               # (12)
                obj += -wd.cN[t] * v[t]
                if force_remind:   # RH-N: flag at full charge and remind once flagged (Eqs. 28-29)
                    m_.addConstr(vh.E_done + k * cum_x_before - vh.D + 1e-4 <= (vh.b + 1) * h[t])
                    m_.addConstr(v[t] >= h[t] - cum_v_before)
                if rc.rule in ('congestion', 'waiting'):
                    xi[t] = m_.addVar(vtype='B', name=f'xi[{j},{t}]')
                    m_.addConstr(xi[t] <= v[t])                                          # (14)
                    m_.addConstr(xi[t] <= kap[t])
                    m_.addConstr(xi[t] >= v[t] + kap[t] - 1)
                elif rc.rule == 'always':
                    xi[t] = v[t]
                cum_x_before += x[t]
                cum_v_before += v[t]
            m_.addConstr(quicksum(v.values()) <= a_eff)                              # (12)
            # release period (13)
            rhs = vh.d * (a_eff - quicksum(v.values()))
            for t in P:
                L0, LF = vh.L0[t], vh.LF[t]
                rhs += L0 * v[t]
                if t in xi:
                    rhs += -(L0 - LF) * xi[t]
            m_.addConstr(quicksum(t * pi[t] for t in Rel) == rhs)                   # (13)
            if sc.solver.tie_eps > 0 and vh.status != 'pseudo':   # tie-breaking term for earlier release
                obj += -sc.solver.tie_eps * quicksum(t * pi[t] for t in Rel)
            self.h[j], self.v[j], self.xi[j] = h, v, xi

        # ---- online window: waiting and newly arrived vehicles plug in first-come, first-served (if rh_fcfs > 0) ----
        if wd.free_now is not None and getattr(sc.solver, 'rh_fcfs', 0) > 0:
            order = {'queued': 0, 'new': 1, 'pseudo': 2}
            wl = [vh for vh in wd.vehicles if vh.status in order and self.s.get(vh.j)]
            wl.sort(key=lambda v: (v.start, order[v.status], v.j))
            for ii in range(len(wl)):
                for kk in range(ii + 1, len(wl)):
                    vi, vk = wl[ii], wl[kk]
                    si, sk = self.s[vi.j], self.s[vk.j]
                    ai = self.aeff[vi.j]
                    for t in sorted(set(si) | set(sk)):
                        lhs = quicksum(si[tt] for tt in si if tt <= t)
                        rhs = quicksum(sk[tt] for tt in sk if tt <= t)
                        m_.addConstr(lhs >= rhs - 1 + ai)

        # ---- online window: accepted new requests plug in if a charger is idle (as in the simulation) ----
        if wd.free_now is not None and new_now:
            y = m_.addVar(vtype='B', name='full_now')
            for a, s0 in new_now:
                m_.addConstr(s0 >= a - y)
            m_.addConstr(quicksum(s0 for _, s0 in new_now) >= min(wd.free_now, len(new_now)) * y)

        # ---- station-level constraints ----
        Mbig = st.m + 1
        self.e = {}
        for t in periods:
            m_.addConstr(U_t[t] <= st.m)                                                 # (4)
            m_.addConstr(Q_t[t] <= st.K)                                                 # (6)
            if rc.rule == 'congestion':
                m_.addConstr(st.m - U_t[t] - rc.theta <= Mbig * (1 - kap[t]))           # (15)
                m_.addConstr(st.m - U_t[t] - rc.theta >= 1 - Mbig * kap[t])             # (16)
            if rc.rule == 'waiting':                                                    # (20) queue-triggered rule: fees only while a vehicle waits
                m_.addConstr(kap[t] <= Q_t[t])
                m_.addConstr(st.K * kap[t] >= Q_t[t])
            m_.addConstr(st.w * X_t[t] <= wd.cap[t])                                     # (17)
            e = m_.addVar(lb=0.0, name=f'e[{t}]')
            m_.addConstr(e >= st.tau * (st.w * X_t[t] - wd.PV[t]))                       # (18)
            obj += -wc.omega * wd.pr[t] * e
            self.e[t] = e
        self.obj = obj
        if set_obj:
            m_.setObjective(obj, maximize=True)

    def solve(self):
        return self.m.optimize(self.sc.solver.solver)

    def val(self, var):
        return self.m.value(var)


class _Proxy:
    """Wrap addVar with a name prefix and forward all other methods to the real model."""
    def __init__(self, m, addVar):
        self._m, self.addVar = m, addVar

    def __getattr__(self, k):
        return getattr(self._m, k)


# ------------------------------------------------------------------
def offline_window(inst) -> WindowData:
    sc = inst.sc
    vehs = []
    for q in inst.requests:
        if q.D <= 1e-9:
            continue
        vehs.append(VehWin(j=q.j, status='new', start=q.r, d=q.d, b=q.b, soc_now=q.soc0,
                           socd=q.socd, D=q.D, wait_left=sc.station.W_max,
                           L0={t: inst.L0(q.j, t) for t in range(q.r, q.d)},
                           LF={t: inst.LF(q.j, t) for t in range(q.r, q.d)}))
    T = inst.T
    cap = {t: max(0.0, inst.PT + inst.PV[t] - inst.B[t]) for t in range(T)}
    return WindowData(0, T, vehs, cap, {t: inst.PV[t] for t in range(T)},
                      {t: inst.pr[t] for t in range(T)}, {t: inst.cN[t] for t in range(T)})


def solve_offline(inst, verbose=False):
    """Solve the offline model and return dict(obj, bound, gap, status, runtime)."""
    wd = offline_window(inst)
    mdl = ChargingMILP(inst.sc, wd, time_limit=inst.sc.solver.offline_time_limit, verbose=verbose)
    mdl.solve()
    g = mdl.m
    res = dict(status=g.status, runtime=g.runtime, n_vars=g.NumVars, n_cons=len(g.rows))
    if g.has_solution:
        res.update(obj=g.obj_val, bound=g.obj_bound, gap=g.gap)
        res['accepted'] = sum(round(mdl.val(mdl.a[j])) for j in mdl.a)
        res['satisfied'] = sum(round(mdl.val(mdl.f[j])) for j in mdl.f)
    return res
