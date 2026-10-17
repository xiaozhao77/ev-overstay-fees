"""
Move-out response model (Section 4.3.2 of the paper).

Non-fee state: Δ0 = d_j - t        with probability ρ_h
           Δ0 = 1 + Y, Y~Geom(ψ_h)  with probability 1-ρ_h          (Eq. 22)
Fee state:     ΔF = min(Δ0, Γ),  Γ ~ U{1..G} with probability p, otherwise +∞   (Eq. 23)
           p = 1 / (1 + exp(-(γ0 + γ1 φ - γ2·1[late night])))       (Eq. 24, φ in CNY/h)
"""
import math
import numpy as np
from .config import BehaviorConfig, StationConfig, RuleConfig

INF = 10 ** 6


def is_night(clock: float, bc: BehaviorConfig) -> bool:
    if bc.night_start > bc.night_end:      # window crosses midnight
        return clock >= bc.night_start or clock < bc.night_end
    return bc.night_start <= clock < bc.night_end


def params_at(t: int, sc: StationConfig, bc: BehaviorConfig):
    night = is_night(sc.clock(t), bc)
    rho = bc.rho_night if night else bc.rho_evening
    psi_h = bc.psi_night if night else bc.psi_evening
    psi = 1.0 - (1.0 - psi_h) ** sc.tau          # hourly probability -> per-period probability
    return night, rho, psi


def p_comply(t: int, sc: StationConfig, bc: BehaviorConfig, rc: RuleConfig) -> float:
    night = is_night(sc.clock(t), bc)
    z = bc.gamma0 + bc.gamma1 * rc.phi - bc.gamma2 * (1.0 if night else 0.0) \
        + getattr(bc, 'gamma3', 0.0) * (rc.G(sc.tau) - 1)          # variant: a longer grace period raises the grace-period move-out probability
    return (1.0 - getattr(bc, 'insensitive_share', 0.0)) / (1.0 + math.exp(-z))


def sample_delay0(t, d, sc, bc, rng: np.random.Generator) -> int:
    _, rho, psi = params_at(t, sc, bc)
    if rng.random() < rho:
        return max(1, d - t)
    # numpy geometric counts from 1, so 1+Y (Y counted from 0) equals numpy geometric
    return int(rng.geometric(psi))


def sample_gamma(t, sc, bc, rc, rng: np.random.Generator) -> int:
    G = rc.G(sc.tau)
    if rng.random() < p_comply(t, sc, bc, rc):
        return int(rng.integers(1, G + 1))
    return INF


# ---------- quantiles (delay estimates used by B1 and the rolling-horizon policy) ----------
def quantile_delay(t, d, kappa: int, p: float, sc, bc, rc) -> int:
    """Return the p quantile of the move-out delay (in periods), capped at d - t."""
    horizon = max(1, d - t)
    _, rho, psi = params_at(t, sc, bc)
    G = rc.G(sc.tau)
    pc = p_comply(t, sc, bc, rc) if kappa else 0.0
    for k in range(1, horizon):
        cdf0 = (1 - rho) * (1 - (1 - psi) ** k)            # P(Δ0 ≤ k), k < d-t
        if kappa:
            surv_gamma = 1 - pc * min(k, G) / G             # P(Γ > k)
            cdf = 1 - (1 - cdf0) * surv_gamma
        else:
            cdf = cdf0
        if cdf >= p:
            return k
    return horizon
