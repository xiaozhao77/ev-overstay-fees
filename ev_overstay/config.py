"""
Experiment parameters.

Data sources of the Shanghai instances are listed in README.md (Data sources); parameters marked [assumption] are assumptions explained in the paper.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


@dataclass
class StationConfig:
    # ---- time ----
    start_hour: float = 16.0          # start of the horizon, 4 p.m.
    horizon_hours: float = 18.0       # until 10 a.m. the next day
    tau: float = 0.25                 # period length (h), 15 minutes

    # ---- station ----
    m: int = 12                       # number of chargers
    w: float = 7.0                    # charger power, kW
    eta: float = 0.9                  # charging efficiency
    K: int = 3                        # waiting spaces
    W_max_hours: float = 3.0          # maximum waiting time (h)

    # ---- transformer and base load ----
    # share of chargers that can operate simultaneously at the evening peak (experimental factor: 0.4 / 0.6 / 0.8)
    cap_ratio_peak: float = 0.6
    base_night_floor: float = 0.0     # floor of the night base load (share of the peak, 0 = original load shape); 1 keeps only the evening-peak residual capacity all night (Appendix B.14)
    base_peak_kw: float = 400.0       # [assumption] peak base load of the community, kW; the shape follows the Shanghai holiday load curve

    # ---- PV ----
    pv_kwp: float = 0.0               # rooftop PV capacity, kWp; none in the base case (Appendix B.6 uses 50 and 100)
    pv_csv: Optional[str] = os.path.join(DATA_DIR, 'pv_shanghai_2023_hourly.csv')  # PVGIS hourly output for Shanghai, 2023 (Beijing time)

    # ---- time-of-use tariff (CNY/kWh): Shanghai residential tariff, peak 6:00-22:00, off-peak 22:00-6:00 ----
    # source: Shanghai residential price guide (2026); shared-meter users pay 0.641 / 0.331
    tou: List[tuple] = field(default_factory=lambda: [
        (0, 6, 0.307),    # off-peak
        (6, 22, 0.617),   # peak
        (22, 24, 0.307),  # off-peak
    ])

    @property
    def T(self) -> int:
        return int(round(self.horizon_hours / self.tau))

    @property
    def W_max(self) -> int:
        return int(round(self.W_max_hours / self.tau))

    def clock(self, t: int) -> float:
        """Clock time (0-24) at the start of period t"""
        return (self.start_hour + t * self.tau) % 24


@dataclass
class DemandConfig:
    n: int = 24                       # charging requests per night (base case: 2 per charger; 12/24/36/48 in Section 6.7)
    # arrival time: normal (clock time), truncated to [16, 23.5]; mean and standard deviation of ACN-Data sessions starting after 4 p.m.
    arr_mean: float = 18.6
    arr_sd: float = 1.6
    # departure time next morning: normal (clock time), truncated to [6, 10] [assumption]
    dep_mean: float = 7.5
    dep_sd: float = 1.0
    # battery capacity: sampled from BEV models common in Shanghai (model first, then configuration, each with equal probability)
    battery_csv: Optional[str] = os.path.join(DATA_DIR, 'battery_models_bev.csv')
    batteries: List[float] = field(default_factory=lambda: [50, 60, 70, 80])  # used when battery_csv is not given
    # initial SOC: normal N(0.35, 0.15^2) truncated to [soc0_low, soc0_high],
    # so that about 37% of vehicles start charging below 30% SOC, as in a national survey of charging behavior (2023)
    soc0_mean: float = 0.35
    soc0_sd: float = 0.15
    soc0_low: float = 0.10
    soc0_high: float = 0.75
    socd: float = 0.80
    csv: Optional[str] = None         # optional CSV of real requests (arr_hour, dep_hour, battery, soc0) to sample from
    # unannounced early departures (robustness, Section 6.7): a share of owners leaves earlier than declared, by a uniform
    # 0.5 to 2 h; the operator plans with the declared departure and learns of the early departure only when the vehicle leaves
    early_share: float = 0.0
    early_lo_h: float = 0.5
    early_hi_h: float = 2.0


@dataclass
class BehaviorConfig:
    """Parameters of the move-out response model (Eqs. 22-24). Values are assumptions (Section 6.2.2, Table 8)."""
    night_start: float = 22.0         # start of the late night
    night_end: float = 7.0            # end of the late night
    # base scenario S2 (values and ranges in Section 6.2.2 and Table 8)
    rho_evening: float = 0.15         # day period: probability of staying until departure
    rho_night: float = 0.75           # late night
    # psi is an hourly move-out probability, converted to 1-(1-psi)^tau per period
    psi_evening: float = 0.65         # owners who do not wait until departure move after about 1.1 h on average
    psi_night: float = 0.20           # about 4.6 h on average
    gamma0: float = -2.20
    gamma1: float = 0.045             # fee sensitivity (phi in CNY/h)
    gamma2: float = 1.40              # late-night effect
    insensitive_share: float = 0.0    # share of owners who ignore fees (p is capped at 1 minus this share)
    gamma3: float = 0.0               # effect of the grace period on compliance (logit rises by gamma3 per additional period)
    plugin_psi_day: float = 0.0       # hourly probability that the owner of a waiting vehicle arrives to plug in once a charger is assigned (day period); 0 = immediate (Appendix B.11)
    plugin_psi_night: float = 0.0     # same, late at night
    spont_psi_day: float = 0.0        # hourly probability that a fully charged, unreminded owner moves out without a reminder (day period); 0 = assumption A3
    spont_psi_night: float = 0.0      # the same late at night
    persist: float = 0.0              # persistent response after the grace period (hourly move-out probability = persist x p); 0 = off (Appendix B.17)
    hetero_share: float = 0.0         # heterogeneous owners: this share behaves as S1 (weak), the rest as S3 (strong); policies still plan with this configuration (S2)


# move-out behavior scenarios for the sensitivity analysis (Table 8); values are assumptions
BEHAVIOR_SCENARIOS = {
    #        rho_day rho_night psi_day psi_night gamma0 gamma1 gamma2
    'S1': dict(rho_evening=0.25, rho_night=0.90, psi_evening=0.40, psi_night=0.10,
               gamma0=-2.80, gamma1=0.030, gamma2=1.80),   # weak response
    'S2': dict(rho_evening=0.15, rho_night=0.75, psi_evening=0.65, psi_night=0.20,
               gamma0=-2.20, gamma1=0.045, gamma2=1.40),   # base
    'S3': dict(rho_evening=0.08, rho_night=0.55, psi_evening=0.80, psi_night=0.30,
               gamma0=-1.80, gamma1=0.060, gamma2=1.00),   # strong response
    'S4': dict(rho_evening=0.15, rho_night=0.75, psi_evening=0.65, psi_night=0.20,
               gamma0=-2.20, gamma1=0.025, gamma2=1.40),   # low fee sensitivity
    'S5': dict(rho_evening=0.15, rho_night=0.75, psi_evening=0.65, psi_night=0.20,
               gamma0=-2.20, gamma1=0.070, gamma2=1.40),   # high fee sensitivity
    'S6': dict(rho_evening=0.15, rho_night=0.90, psi_evening=0.65, psi_night=0.10,
               gamma0=-2.20, gamma1=0.045, gamma2=2.20),   # strong late-night inertia
    'S7': dict(rho_evening=0.15, rho_night=0.60, psi_evening=0.65, psi_night=0.30,
               gamma0=-2.20, gamma1=0.045, gamma2=0.60),   # responsive late at night
}
BEHAVIOR_SCENARIOS['base'] = BEHAVIOR_SCENARIOS['S2']


@dataclass
class RuleConfig:
    rule: str = "congestion"          # "none" | "always" | "congestion"
    theta: int = 0                    # congestion threshold (fees apply when idle chargers <= theta)
    grace_hours: float = 20 / 60      # grace period (h)
    phi: float = 60.0                 # idle fee rate (CNY/h), charged phi*tau per period; 60 CNY/h = 1 CNY/min
    fee_from_full: bool = False       # fee clock from full charge (grace period starts at full charge; unreminded vehicles also pay and receive a fee notice)
    fee_cap: float = 0.0               # cap on the idle fee per vehicle and night (CNY); 0 = no cap

    def G(self, tau: float) -> int:
        return max(1, int(round(self.grace_hours / tau)))


@dataclass
class WeightConfig:
    alpha: float = 10.0               # admission reward
    beta: float = 100.0               # target reward
    omega: float = 1.0                # electricity cost weight
    lam: float = 0.5                  # waiting cost per vehicle per period
    cN_evening: float = 0.0           # reminder disturbance cost, day period
    cN_night: float = 5.0             # reminder disturbance cost, late night
    mu: float = 0.5                   # minimum charged share of the demand


@dataclass
class SolverConfig:
    solver: str = "highs"             # "highs" (open source, default) | "copt" | "gurobi"
    offline_time_limit: float = 1800.0
    rh_time_limit: float = 20.0
    mip_gap: float = 1e-3
    threads: int = 0                  # 0 = chosen by the solver
    quantile_p: float = 0.5           # quantile of the move-out delay used for estimates
    pv_future_zero: bool = True       # future PV set to zero in the rolling horizon (conservative)
    rh_forecast: bool = True          # add expected-scenario future requests to the rolling-horizon window
    forecast_pool: int = 5000         # sample size for estimating the arrival distribution
    forecast_weight: float = 0.7      # reward discount of forecast requests (avoids rejecting real requests for hypothetical ones)
    forecast_hours: float = 3.0       # forecast arrivals only for the next hours, to bound the window size
    rh_mip_gap: float = 0.01          # relative gap of each rolling-horizon solve (the paper uses 0.001)
    b1_project: bool = True           # B1 estimates the release of unreminded vehicles as expected full charge plus the delay quantile; False uses d
    fee_onset_notice: bool = True     # send fee notices in the online simulation
    cap_target: bool = False          # stop charging at the target (used in the PV scenario)
    tie_eps: float = 0.0              # tie-breaking coefficient for earlier release in the RH window model (per period)
    tb_lex: float = 0.0               # deterministic tie-breaking (>0 enables): prefer earlier requests and earlier charging among equal optima; use with a very small rh_mip_gap
    rh_fcfs: float = 0.0              # >0: the RH window model imposes first-come, first-served plug-in of waiting vehicles (Appendix B.7)
    rh_dep_buffer_h: float = 0.0      # >0: RH plans as if every vehicle left this many hours before its declared departure (safety margin against early departures, Section 6.7.5)
    rh_scen: int = 5                  # number of sampled future-request scenarios per step in RH-MS (Appendix B.12)
    rh_accept_bonus: float = 0.0      # set internally by RH-A (accept every request that fits)


@dataclass
class Scenario:
    station: StationConfig = field(default_factory=StationConfig)
    demand: DemandConfig = field(default_factory=DemandConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    weights: WeightConfig = field(default_factory=WeightConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
