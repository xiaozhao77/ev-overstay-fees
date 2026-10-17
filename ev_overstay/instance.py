"""
Instance generation (Section 6.2 of the paper).

An Instance represents one night: all charging requests, the base load, PV output and prices of each period,
and the realized move-out delay of every vehicle in every possible reminder period (shared by the offline model and the online simulation).
"""
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd

from .config import Scenario
from . import behavior as bh


@dataclass
class Request:
    j: int
    r: int            # arrival period
    d: int            # declared departure period (the vehicle leaves at the start of d and is present in periods r..d-1)
    b: float          # battery capacity, kWh
    soc0: float
    socd: float

    @property
    def D(self) -> float:   # required energy, kWh
        return max(0.0, (self.socd - self.soc0) * self.b)


@dataclass
class Instance:
    sc: Scenario
    seed: int
    requests: List[Request]
    B: np.ndarray        # base load, kW
    PT: float            # transformer capacity, kW
    PV: np.ndarray       # PV output, kW
    pr: np.ndarray       # price, CNY/kWh
    cN: np.ndarray       # reminder disturbance cost
    U: Dict[int, np.ndarray]         # U[j]: uniform random numbers of shape (d-r, 4), common random numbers
    V: Dict[int, float] = None       # owner-type random number (fee-insensitive or not)
    V2: Dict[int, float] = None      # owner-type random number (heterogeneous owners: weak or strong response)

    def bc_of(self, j):
        """Behavior parameters followed by owner j: S1 or S3 by type if hetero_share > 0, otherwise the scenario parameters."""
        bc = self.sc.behavior
        sh = getattr(bc, 'hetero_share', 0.0)
        if sh <= 0 or self.V2 is None:
            return bc
        import copy as _c
        from .config import BEHAVIOR_SCENARIOS
        key = ('S1' if self.V2[j] < sh else 'S3', id(bc))
        cache = self.__dict__.setdefault('_bc_cache', {})
        if key not in cache:
            b2 = _c.copy(bc)
            for k_, v_ in BEHAVIOR_SCENARIOS[key[0]].items():
                setattr(b2, k_, v_)
            b2.hetero_share = 0.0
            cache[key] = b2
        return cache[key]

    @property
    def T(self):
        return self.sc.station.T

    # realized delays use the current scenario (behavior and rule parameters); all rules share the random numbers of a seed
    def D0(self, j, t) -> int:
        q = self.requests[j]
        u = self.U[j][t - q.r]
        _, rho, psi = bh.params_at(t, self.sc.station, self.bc_of(j))
        if u[0] < rho:
            return max(1, q.d - t)
        return 1 + int(np.floor(np.log(1 - u[1]) / np.log(1 - psi)))

    def Gam(self, j, t) -> int:
        q = self.requests[j]
        u = self.U[j][t - q.r]
        G = self.sc.rule.G(self.sc.station.tau)
        bc = self.bc_of(j)
        ish = getattr(bc, 'insensitive_share', 0.0)
        if ish > 0 and self.V is not None:
            if self.V[j] < ish:
                return bh.INF          # fee-insensitive owner: does not move within the grace period even under fees
            pc = bh.p_comply(t, self.sc.station, bc, self.sc.rule) / (1 - ish)
        else:
            pc = bh.p_comply(t, self.sc.station, bc, self.sc.rule)
        if u[2] < pc:
            return 1 + int(u[3] * G)
        return bh.INF

    def L0(self, j, t):
        return min(self.requests[j].d, t + self.D0(j, t))

    def LF(self, j, t):
        return min(self.requests[j].d, t + min(self.D0(j, t), self.Gam(j, t)))


# ---------------- external profiles ----------------
# base-load shape (relative to the peak) by hour, read from the typical holiday load curve of the Shanghai grid (provincial
# typical load curves published by the NDRC in 2020); the holiday curve is dominated by residential load and peaks around 19:00
_SH_HOLIDAY_MW = [11.8, 11.5, 11.1, 10.8, 10.8, 11.1, 11.6, 12.0, 12.4, 13.3, 14.2, 14.6,
                  14.5, 14.5, 14.4, 14.1, 14.1, 14.6, 15.0, 15.2, 14.9, 14.6, 14.0, 13.0]
_BASE_SHAPE = {h: v / max(_SH_HOLIDAY_MW) for h, v in enumerate(_SH_HOLIDAY_MW)}


def base_load(sc: Scenario) -> (np.ndarray, float):
    st = sc.station
    fl = getattr(st, 'base_night_floor', 0.0)   # floor of the night base load (share of the peak), for the all-night transformer limit scenario
    B = np.array([st.base_peak_kw * max(fl, _BASE_SHAPE[int(st.clock(t)) % 24]) for t in range(st.T)])
    # transformer capacity: cap_ratio_peak x m chargers can operate at the evening peak
    PT = st.base_peak_kw + st.cap_ratio_peak * st.m * st.w
    return B, PT


def pv_profile(sc: Scenario, rng) -> np.ndarray:
    st = sc.station
    if st.pv_kwp <= 0:
        return np.zeros(st.T)
    if st.pv_csv:   # PVGIS export with columns 'hour' and 'P_kw_per_kwp'
        df = pd.read_csv(st.pv_csv)
        prof = dict(zip(df['hour'].astype(int), df['P_kw_per_kwp']))
        return np.array([st.pv_kwp * prof.get(int(st.clock(t)) % 24, 0.0) for t in range(st.T)])
    # fallback: bell-shaped curve from 6:00 to 19:00 with a weather factor
    weather = rng.uniform(0.4, 1.0)
    out = []
    for t in range(st.T):
        h = st.clock(t)
        v = max(0.0, np.sin(np.pi * (h - 6) / 13)) if 6 <= h <= 19 else 0.0
        out.append(st.pv_kwp * 0.8 * weather * v)
    return np.array(out)


def price_profile(sc: Scenario) -> np.ndarray:
    st = sc.station
    out = []
    for t in range(st.T):
        h = st.clock(t)
        p = None
        for a, b, v in st.tou:
            if a <= h < b:
                p = v
        out.append(p if p is not None else st.tou[-1][2])
    return np.array(out)


def notify_cost(sc: Scenario) -> np.ndarray:
    st, bc, wc = sc.station, sc.behavior, sc.weights
    return np.array([wc.cN_night if bh.is_night(st.clock(t), bc) else wc.cN_evening for t in range(st.T)])


# ---------------- requests ----------------
def _hour_to_t(h, st, next_day=False):
    hh = h + (24 if next_day else 0)
    return int(round((hh - st.start_hour) / st.tau))


_BAT_CACHE = {}


def _battery_table(dc):
    """Read the BEV model table and return (models, {model: [capacities]}); None if not given."""
    if not dc.battery_csv:
        return None
    if dc.battery_csv not in _BAT_CACHE:
        df = pd.read_csv(dc.battery_csv, encoding='utf-8-sig')
        groups = {}
        for r in df.itertuples(index=False):
            groups.setdefault(f'{r[0]} {r[1]}', []).append(float(r[3]))
        _BAT_CACHE[dc.battery_csv] = (list(groups), groups)   # file order (the file is sorted so that sampling matches the original runs)
    return _BAT_CACHE[dc.battery_csv]


def _trunc_normal(rng, mu, sd, lo, hi):
    for _ in range(1000):
        x = rng.normal(mu, sd)
        if lo <= x <= hi:
            return float(x)
    return float(np.clip(mu, lo, hi))


def sample_requests(sc: Scenario, rng) -> List[Request]:
    st, dc = sc.station, sc.demand
    reqs = []
    if dc.csv:
        df = pd.read_csv(dc.csv)  # columns: arr_hour, dep_hour, battery, soc0
        rows = df.sample(n=dc.n, replace=True, random_state=int(rng.integers(1 << 31)))
        raw = [(r.arr_hour, r.dep_hour, r.battery, r.soc0) for r in rows.itertuples()]
    else:
        bat = _battery_table(dc)
        raw = []
        for _ in range(dc.n):
            a = float(np.clip(rng.normal(dc.arr_mean, dc.arr_sd), 16.0, 23.5))
            dpt = float(np.clip(rng.normal(dc.dep_mean, dc.dep_sd), 6.0, 10.0))
            if bat is not None:
                models, groups = bat
                g = groups[models[int(rng.integers(len(models)))]]
                b = float(g[int(rng.integers(len(g)))])
            else:
                b = float(rng.choice(dc.batteries))
            s0 = _trunc_normal(rng, dc.soc0_mean, dc.soc0_sd, dc.soc0_low, dc.soc0_high)
            raw.append((a, dpt, b, s0))
    for a, dpt, b, s0 in raw:
        r = max(0, min(st.T - 2, _hour_to_t(a, st)))
        d = _hour_to_t(dpt, st, next_day=dpt < st.start_hour)
        d = max(r + 1, min(st.T, d))
        reqs.append((r, d, b, s0))
    reqs.sort(key=lambda x: (x[0], x[1]))
    return [Request(j, r, d, b, s0, dc.socd) for j, (r, d, b, s0) in enumerate(reqs)]


def generate(sc: Scenario, seed: int) -> Instance:
    rng = np.random.default_rng(seed)
    reqs = sample_requests(sc, rng)
    B, PT = base_load(sc)
    PV = pv_profile(sc, rng)
    pr = price_profile(sc)
    cN = notify_cost(sc)
    # move-out random numbers: 4 uniforms per vehicle and period present, shared by offline and online models and by all rules
    U = {q.j: rng.random((q.d - q.r, 4)) for q in reqs}
    V = {q.j: float(rng.random()) for q in reqs}     # drawn after U, so the earlier common random numbers are unchanged
    V2 = {q.j: float(rng.random()) for q in reqs}    # drawn after V
    return Instance(sc, seed, reqs, B, PT, PV, pr, cN, U, V, V2)


def generate_from(sc: Scenario, seed: int, raw) -> Instance:
    """Consecutive-night simulation: build one night from given requests (r, d, b, soc0, hh, prio)."""
    rng = np.random.default_rng(seed)
    raw = sorted(raw, key=lambda x: (x[0], x[1]))
    reqs = []
    for j, (r, d, b, s0, hh, prio) in enumerate(raw):
        q = Request(j, r, d, b, s0, sc.demand.socd)
        q.hh, q.prio = hh, prio
        reqs.append(q)
    B, PT = base_load(sc)
    PV = pv_profile(sc, rng)
    pr = price_profile(sc)
    cN = notify_cost(sc)
    U = {q.j: rng.random((q.d - q.r, 4)) for q in reqs}
    V = {q.j: float(rng.random()) for q in reqs}
    V2 = {q.j: float(rng.random()) for q in reqs}
    return Instance(sc, seed, reqs, B, PT, PV, pr, cN, U, V, V2)
