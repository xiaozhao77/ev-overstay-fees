"""
Online simulation environment (Section 4.3 of the paper).

Order of events in each period:
  1 release: vehicles at their declared departure and reminded vehicles whose move-out delay has elapsed leave;
  2 plug-in: waiting vehicles plug into idle chargers first-come, first-served; vehicles that have waited W_max periods abandon;
  3 admission: new requests arrive in order and the policy accepts or rejects them; accepted vehicles plug in if a charger is idle, otherwise wait, and are rejected if the waiting area is full;
    (the fee state kappa_t follows from the occupied chargers in this period: fees apply when idle chargers <= theta, as in constraints (15)(16))
  4 charging: the policy selects vehicles to charge and the simulator enforces the transformer limit;
  5 update: vehicles that reach their targets can be reminded from the next period on (as t' < t in constraint (11));
  6 reminders: the policy selects fully charged vehicles to remind, and move-out delays come from the common random numbers;
  7 fees: when the station enters the fee state, vehicles reminded in the non-fee state that are still present receive a fee notice; idle fees accrue in fee periods.

The cumulative reward matches the offline objective (1):
  alpha*accepted + beta*released at target - omega*sum pr_t e_t - lambda*sum |Q_t| - sum c^N_t * reminders
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from . import behavior as bh

EPS = 1e-9


@dataclass
class VState:
    j: int
    r: int
    d: int
    b: float
    soc0: float
    socd: float
    D: float
    status: str = 'future'          # future | queued | plugged | gone | rejected
    accepted: bool = False
    admit_t: Optional[int] = None
    plug_t: Optional[int] = None
    E: float = 0.0                  # energy charged, kWh
    t_full: Optional[int] = None    # fully charged (can be reminded) from this period on
    reminded: bool = False
    n_rem: Optional[int] = None     # reminder period
    kappa_rem: int = 0              # fee state at the reminder
    notified: bool = False          # whether a fee notice was received
    R: Optional[int] = None         # actual release period (internal to the simulation, not visible to policies)
    fee_start: Optional[int] = None
    fee: float = 0.0
    q_periods: int = 0              # periods spent waiting
    left_t: Optional[int] = None
    outcome: str = ''               # served | unmet | abandoned | rejected
    prio: float = 0.0               # multi-night fairness: admission priority bonus
    hh: int = -1                    # multi-night simulation: household index

    @property
    def soc(self):
        return self.soc0 + self.E / self.b

    @property
    def full(self):
        return self.E >= self.D - 1e-6


class Simulator:
    def __init__(self, inst, policy, fee_onset_notice: Optional[bool] = None):
        self.inst = inst
        self.sc = inst.sc
        self.policy = policy
        self.notice = self.sc.solver.fee_onset_notice if fee_onset_notice is None else fee_onset_notice
        st = self.sc.station
        self.k = st.eta * st.tau * st.w
        self.G = self.sc.rule.G(st.tau)
        self.v: List[VState] = [VState(q.j, q.r, q.d, q.b, q.soc0, q.socd, q.D,
                                       prio=getattr(q, 'prio', 0.0), hh=getattr(q, 'hh', -1)) for q in inst.requests]
        self.queue: List[int] = []          # waiting order (first-come, first-served)
        self.reserved = {}                  # waiting vehicles assigned a charger whose owner has not yet plugged in: j -> plug-in period (Appendix B.11)
        self.plugged: List[int] = []
        self.kappa_hist: List[int] = []
        self.t = 0
        # reward components
        self.n_accept = 0
        self.n_full_release = 0
        self.energy_cost = 0.0
        self.grid_kwh = 0.0
        self.qh = np.zeros(18)    # hourly waiting vehicle-hours (Figure 10b)
        self.bh = np.zeros(18)    # hourly blocking vehicle-hours
        self.offpeak_kwh = 0.0   # grid purchase in off-peak periods
        self.queue_periods = 0
        self.remind_cost = 0.0
        self.n_remind = 0
        self.n_remind_night = 0
        self.n_notice = 0
        self.n_notice_night = 0
        self.charge_periods = 0
        self.occupy_periods = 0
        self.block_periods = 0      # vehicle-periods in which fully charged vehicles occupy chargers while vehicles wait
        self.solve_times: List[float] = []

    # ------------------------------------------------------------------
    def free(self) -> int:
        return self.sc.station.m - len(self.plugged) - len(self.reserved)

    def plugin_delay(self, j, t) -> int:
        """Delay (periods, possibly 0) until the owner of a waiting vehicle assigned a charger arrives to plug in. plugin_psi_* are hourly arrival probabilities; 0 means immediate plug-in (Appendix B.11)."""
        bc = self.sc.behavior
        pd_, pn_ = getattr(bc, 'plugin_psi_day', 0.0), getattr(bc, 'plugin_psi_night', 0.0)
        if pd_ <= 0 and pn_ <= 0:
            return 0
        st = self.sc.station
        psi = pn_ if bh.is_night(st.clock(t), bc) else pd_
        q = 1.0 - (1.0 - min(psi, 0.999999)) ** st.tau
        u = np.random.default_rng([int(self.inst.seed), int(j), int(t), 17]).random()
        return int(np.floor(np.log(max(u, 1e-12)) / np.log(1.0 - q))) if q < 1 else 0

    def cap(self, t) -> float:
        i = self.inst
        return max(0.0, i.PT + i.PV[t] - i.B[t])

    def kappa_now(self) -> int:
        rc = self.sc.rule
        if rc.rule == 'none':
            return 0
        if rc.rule == 'always':
            return 1
        if rc.rule == 'waiting':      # queue-triggered rule (20): fees while vehicles wait
            return 1 if self.queue else 0
        return 1 if self.free() <= rc.theta else 0

    def _release(self, x: VState, t):
        self.plugged.remove(x.j)
        x.status, x.left_t = 'gone', t
        if x.full:
            self.n_full_release += 1
            x.outcome = 'served'
        else:
            x.outcome = 'unmet'

    def _plug(self, x: VState, t):
        x.status, x.plug_t = 'plugged', t
        self.plugged.append(x.j)

    # ------------------------------------------------------------------
    def step(self, t):
        st, rc, wc = self.sc.station, self.sc.rule, self.sc.weights
        inst = self.inst
        self.t = t
        # 1 release
        for j in list(self.plugged):
            x = self.v[j]
            R = x.R if x.R is not None else x.d
            if t >= min(R, x.d):
                self._release(x, t)
        # 2 plug-in and abandonment (with plug-in delays, an assigned charger is held until the owner arrives)
        for j in list(self.reserved):
            if t >= self.reserved[j]:
                del self.reserved[j]
                self.queue.remove(j)
                self._plug(self.v[j], t)
        while self.free() > 0:
            nxt = [j for j in self.queue if j not in self.reserved]
            if not nxt:
                break
            j = nxt[0]
            dly = self.plugin_delay(j, t)
            if dly <= 0:
                self.queue.remove(j)
                self._plug(self.v[j], t)
            else:
                self.reserved[j] = t + dly
        for j in list(self.queue):
            if j in self.reserved:
                if t >= self.v[j].d:          # declared departure reached before the owner arrived
                    del self.reserved[j]
                    self.queue.remove(j)
                    x = self.v[j]
                    x.status, x.left_t, x.outcome = 'gone', t, 'abandoned'
                continue
            x = self.v[j]
            if x.q_periods >= st.W_max or t >= x.d:
                self.queue.remove(j)
                x.status, x.left_t, x.outcome = 'gone', t, 'abandoned'
        # 3 admission
        arrivals = [x for x in self.v if x.r == t and x.D > EPS]
        for x in self.v:
            if x.r == t and x.D <= EPS:
                x.status, x.outcome = 'rejected', 'no_demand'
        if arrivals:
            adm = self.policy.admit(self, t, arrivals)
            for x in arrivals:
                ok = x.j in adm
                if ok and self.free() > 0:
                    self._plug(x, t)
                elif ok and len(self.queue) < st.K:
                    x.status = 'queued'
                    self.queue.append(x.j)
                else:
                    ok = False
                if ok:
                    x.accepted, x.admit_t = True, t
                    self.n_accept += 1
                else:
                    x.status, x.outcome = 'rejected', 'rejected'
        kap = self.kappa_now()
        self.kappa_hist.append(kap)
        # 4 charging
        charge, remind = self.policy.operate(self, t)
        nmax = int(np.floor(self.cap(t) / st.w + 1e-9))
        sel = []
        for j in charge:
            x = self.v[j]
            if x.status != 'plugged' or j in sel:
                continue
            if x.soc + self.k / x.b > 1.0 + 1e-6:
                continue
            if self.sc.solver.cap_target and x.full:   # stop charging at the target
                continue
            if len(sel) >= nmax:
                break
            sel.append(j)
        P = st.w * len(sel)
        e = st.tau * max(0.0, P - inst.PV[t])
        self.grid_kwh += e
        if inst.pr[t] <= float(np.min(inst.pr)) + 1e-9:
            self.offpeak_kwh += e
        self.energy_cost += inst.pr[t] * e
        self.charge_periods += len(sel)
        self.occupy_periods += len(self.plugged)
        self.queue_periods += len(self.queue)
        for j in self.queue:
            self.v[j].q_periods += 1
        if self.queue:
            nfull = sum(1 for j in self.plugged if self.v[j].t_full is not None and self.v[j].t_full <= t)
            self.block_periods += min(len(self.queue), nfull)
            _hb = int((st.clock(t) - 16) % 24)
            if _hb < 18:
                self.qh[_hb] += len(self.queue) * st.tau
                self.bh[_hb] += min(len(self.queue), nfull) * st.tau
        # 5 update state of charge
        for j in sel:
            x = self.v[j]
            x.E += self.k
            if x.t_full is None and x.full:
                x.t_full = t + 1
        # 6 reminders
        night = bh.is_night(st.clock(t), self.sc.behavior)
        for j in remind:
            x = self.v[j]
            if x.status != 'plugged' or x.reminded or x.t_full is None or x.t_full > t:
                continue
            x.reminded, x.n_rem, x.kappa_rem = True, t, kap
            x.R = inst.LF(j, t) if kap else inst.L0(j, t)
            x.fee_start = t + self.G
            self.n_remind += 1
            self.n_remind_night += int(night)
            self.remind_cost += inst.cN[t]
        # 7 fee notices and idle fees
        prev = self.kappa_hist[-2] if len(self.kappa_hist) >= 2 else kap
        if self.notice and kap == 1 and prev == 0:
            for j in self.plugged:
                x = self.v[j]
                if x.reminded and x.kappa_rem == 0 and not x.notified and x.n_rem < t:
                    x.notified = True
                    self.n_notice += 1
                    self.n_notice_night += int(bh.is_night(st.clock(t), self.sc.behavior))
                    g = inst.Gam(j, t)
                    if g < bh.INF:
                        x.R = min(x.R, t + g)
                    x.fee_start = max(x.fee_start, t + self.G)
        if getattr(rc, 'fee_from_full', False) and kap and rc.rule != 'none':
            # variant: grace period from full charge; unreminded fully charged vehicles receive a fee notice when fees first apply
            for j in self.plugged:
                x = self.v[j]
                if x.t_full is None or x.t_full > t:
                    continue
                if not x.reminded and not x.notified and t >= x.t_full + self.G:
                    x.notified = True
                    self.n_notice += 1
                    self.n_notice_night += int(bh.is_night(st.clock(t), self.sc.behavior))
                    g = inst.Gam(j, t)
                    if g < bh.INF:
                        x.R = min(x.R if x.R is not None else x.d, t + g)
                if t >= x.t_full + self.G and (x.R is None or t < x.R):
                    x.fee += rc.phi * st.tau
        elif kap and rc.rule != 'none':
            for j in self.plugged:
                x = self.v[j]
                if x.reminded and t >= x.fee_start and (x.R is None or t < x.R):
                    x.fee += rc.phi * st.tau
                    ps = getattr(self.sc.behavior, 'persist', 0.0)
                    if ps > 0:   # persistent response (Appendix B.17): after the grace period, move out with probability persist x p per hour
                        ph = min(1.0, ps * bh.p_comply(t, st, self.sc.behavior, rc))
                        pt = 1.0 - (1.0 - ph) ** st.tau
                        if np.random.default_rng([int(self.inst.seed), int(j), int(t), 29]).random() < pt:
                            x.R = min(x.R if x.R is not None else x.d, t + 1)

    def run(self):
        T = self.sc.station.T
        for t in range(T):
            self.step(t)
        for j in list(self.plugged):
            self._release(self.v[j], T)
        for j in list(self.queue):
            x = self.v[j]
            x.status, x.left_t, x.outcome = 'gone', T, 'abandoned'
        self.queue.clear()
        self.reserved.clear()
        return self.metrics()

    # ------------------------------------------------------------------
    def reward(self):
        wc = self.sc.weights
        return (wc.alpha * self.n_accept + wc.beta * self.n_full_release
                - wc.omega * self.energy_cost - wc.lam * self.queue_periods - self.remind_cost)

    def metrics(self):
        st = self.sc.station
        tau = st.tau
        vs = [x for x in self.v if x.D > EPS]
        n = len(vs)
        acc = [x for x in vs if x.accepted]
        full = [x for x in acc if x.t_full is not None and x.left_t is not None]
        over = [(x.left_t - x.t_full) * tau for x in full]
        qt = [x.q_periods * tau for x in acc]
        fees = [x.fee for x in acc]
        mu = self.sc.weights.mu
        return dict(
            reward=self.reward(),
            n=n,
            accepted=len(acc),
            accept_rate=len(acc) / n if n else 0,
            served=self.n_full_release,
            service_rate=self.n_full_release / n if n else 0,
            served_of_accepted=self.n_full_release / len(acc) if acc else 0,
            abandoned=sum(x.outcome == 'abandoned' for x in vs),
            min_energy_violations=sum(1 for x in acc if x.E < mu * x.D - 1e-6),
            avg_queue_h=float(np.mean(qt)) if qt else 0.0,
            avg_overstay_h=float(np.mean(over)) if over else 0.0,
            block_h=self.block_periods * tau,
            reminders=self.n_remind,
            night_reminders=self.n_remind_night,
            notices=self.n_notice,
            night_notices=self.n_notice_night,
            remind_cost=self.remind_cost,
            total_fee=float(np.sum(fees)) if fees else 0.0,
            avg_fee=float(np.mean(fees)) if fees else 0.0,
            fee_p95=float(np.percentile(fees, 95)) if fees else 0.0,
            max_fee=float(np.max(fees)) if fees else 0.0,
            fee_payers=sum(1 for f in fees if f > 0),
            energy_cost=self.energy_cost,
            grid_kwh=self.grid_kwh,
            delivered_kwh=self.charge_periods * self.k,
            utilization=self.charge_periods / self.occupy_periods if self.occupy_periods else 0.0,
            fee_state_share=float(np.mean(self.kappa_hist)) if self.kappa_hist else 0.0,
            # completion times (hourly counts from 4 p.m. to 10 a.m.) and off-peak share of grid purchases
            queue_hist=';'.join(f'{v:.3f}' for v in self.qh),
            block_hist=';'.join(f'{v:.3f}' for v in self.bh),
            full_hist=';'.join(str(c) for c in np.bincount(
                [int(((st.clock(x.t_full) - 16) % 24)) for x in acc if x.t_full is not None and x.t_full < st.T], minlength=18)[:18]),
            full_night=sum(1 for x in acc if x.t_full is not None and x.t_full < st.T and (st.clock(x.t_full) >= 22 or st.clock(x.t_full) < 7)),
            offpeak_share=self.offpeak_kwh / self.grid_kwh if self.grid_kwh > 0 else 0.0,
            avg_solve_s=float(np.mean(self.solve_times)) if self.solve_times else 0.0,
            max_solve_s=float(np.max(self.solve_times)) if self.solve_times else 0.0,
        )
