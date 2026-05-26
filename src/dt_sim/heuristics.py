"""Chain-length estimator and decision-rule scoring.

The anchored chain estimator maps a detector support interval inside two
schedule anchors to E[K].  The left anchor is placed at zero by translation;
the detector support can start after the left anchor and end before the right
anchor, so boundary clearances are applied only when the support approaches an
anchor closer than the required spacing g.

`score_lookahead` returns the per-pitch decision score: visible count for
"greedy", finite-window advantage for "two_anchor", and break-even spacing
margin for "break_even".
"""
import numpy as np
import scipy.stats as st

from .constants import Constants


def subtending_ground_angle(off_nadir_rad: float, altitude_km: float) -> float:
    """Ground central angle subtended by an off-nadir viewing ray."""
    if altitude_km <= 0:
        return 0.0
    radius = Constants.R_E
    semimajor = radius + altitude_km
    max_angle = np.arcsin(radius / semimajor)
    sign = np.sign(off_nadir_rad)
    angle = min(abs(float(off_nadir_rad)), float(max_angle) - 1e-12)
    if angle <= 0:
        return 0.0
    ratio = np.clip(semimajor / radius * np.sin(angle), -1.0, 1.0)
    beta = np.pi - np.arcsin(ratio)
    return float(sign * (np.pi - (angle + beta)))


def access_time_from_off_nadir(off_nadir_rad: float, altitude_km: float) -> float:
    """Approximate along-track access-time offset for an off-nadir ray."""
    semimajor = Constants.R_E + altitude_km
    mean_motion = np.sqrt(Constants.mu / semimajor ** 3)
    return subtending_ground_angle(off_nadir_rad, altitude_km) / mean_motion


def geometric_lookahead_window(
    orbit, pitch_deg: float, fov_deg: float, *,
    earliest_offset_s: float = 0.0,
) -> tuple[float, float, float]:
    """Return the usable geometric access-time window for a pitched camera.

    The window is set by the near and far vertical detector edges along the
    access-time direction. The far edge is clipped at the limb; the near edge is
    clipped by ``earliest_offset_s`` for right-anchor timing.
    """
    altitude_km = float(orbit.a - Constants.R_E)
    horizon_deg = float(np.rad2deg(np.arcsin(Constants.R_E / orbit.a)))
    near_angle = max(float(pitch_deg) - float(fov_deg) / 2.0, 0.0)
    far_angle = min(float(pitch_deg) + float(fov_deg) / 2.0,
                    horizon_deg - 1e-9)
    if far_angle <= near_angle:
        return 0.0, 0.0, 0.0
    near_s = access_time_from_off_nadir(np.deg2rad(near_angle), altitude_km)
    far_s = access_time_from_off_nadir(np.deg2rad(far_angle), altitude_km)
    start_s = max(float(earliest_offset_s), near_s)
    return max(far_s - start_s, 0.0), start_s, far_s


def chain_two_boundary(lam: float, L: float, g: float) -> float:
    """Expected chain length when the detector support spans both anchors."""
    return chain_anchor_support(
        lam, support_window=L, g=g, left_anchor_gap=0.0, right_anchor_gap=0.0)


def _usable_support_span(
    support_window: float,
    g: float,
    left_anchor_gap: float,
    right_anchor_gap: float,
) -> float:
    left_clearance = max(0.0, float(g) - max(float(left_anchor_gap), 0.0))
    right_clearance = max(0.0, float(g) - max(float(right_anchor_gap), 0.0))
    return max(0.0, float(support_window) - left_clearance - right_clearance)


def _max_positive_support_spacing(
    support_window: float,
    left_anchor_gap: float,
    right_anchor_gap: float,
) -> float:
    """Largest spacing with nonzero usable support."""
    support = max(float(support_window), 0.0)
    left_gap = max(float(left_anchor_gap), 0.0)
    right_gap = max(float(right_anchor_gap), 0.0)
    small_gap = min(left_gap, right_gap)
    large_gap = max(left_gap, right_gap)
    if support + small_gap <= large_gap:
        return float(support + small_gap)
    return float(0.5 * (support + left_gap + right_gap))


def chain_anchor_support(
    lam: float,
    support_window: float,
    g: float,
    left_anchor_gap: float,
    right_anchor_gap: float,
) -> float:
    """Expected chain length for detector support inside two schedule anchors.

    ``left_anchor_gap`` is the time from the left anchor to the detector-support
    start. ``right_anchor_gap`` is the time from the detector-support end to the
    right anchor. Both are zero when the detector support coincides with the
    anchors, recovering the standard two-boundary form.
    """
    if lam <= 0 or support_window <= 0 or g <= 0:
        return 0.0
    usable = _usable_support_span(
        support_window, g, left_anchor_gap, right_anchor_gap)
    if usable <= 0:
        return 0.0
    total = 0.0
    max_k = int(usable / g) + 3
    for k in range(1, max_k + 1):
        remaining = usable - (k - 1) * g
        if remaining <= 0:
            break
        p = float(st.gamma.cdf(remaining, a=k, scale=1.0 / lam))
        if p < 1e-15:
            break
        total += p
    return total


def break_even_spacing(lam: float, L: float, baseline_b: float) -> float:
    """Full-support spacing threshold from the support-aware rate approximation."""
    expected_clear = lam * L
    if L <= 0 or expected_clear <= baseline_b:
        return -np.inf
    return float(L * (expected_clear - baseline_b)
                 / (expected_clear * (2.0 + baseline_b)))


def renewal_rate_anchor_support(
    lam: float,
    support_window: float,
    g: float,
    left_anchor_gap: float,
    right_anchor_gap: float,
) -> float:
    """Renewal-rate approximation for detector support inside fixed anchors."""
    if lam <= 0 or support_window <= 0 or g < 0:
        return 0.0
    usable = _usable_support_span(
        support_window, g, left_anchor_gap, right_anchor_gap)
    if usable <= 0:
        return 0.0
    return float(lam * usable / (1.0 + lam * g))


def break_even_spacing_anchor_support(
    lam: float,
    support_window: float,
    baseline_b: float,
    left_anchor_gap: float,
    right_anchor_gap: float,
) -> float:
    """Numerical spacing threshold for the support-aware rate approximation."""
    if lam <= 0 or support_window <= 0:
        return -np.inf
    if renewal_rate_anchor_support(
        lam, support_window, 0.0, left_anchor_gap, right_anchor_gap
    ) <= baseline_b:
        return -np.inf
    if baseline_b <= 0:
        return _max_positive_support_spacing(
            support_window, left_anchor_gap, right_anchor_gap)

    low = 0.0
    high = max(
        1.0,
        _max_positive_support_spacing(
            support_window, left_anchor_gap, right_anchor_gap),
    )
    for _ in range(64):
        if renewal_rate_anchor_support(
            lam, support_window, high, left_anchor_gap, right_anchor_gap
        ) <= baseline_b:
            break
        high *= 2.0
    else:
        return high

    for _ in range(64):
        mid = 0.5 * (low + high)
        if renewal_rate_anchor_support(
            lam, support_window, mid, left_anchor_gap, right_anchor_gap
        ) > baseline_b:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def score_lookahead(variant, *, n_visible, t_window, p_clear, n_sched, avg_u,
                    missed_utility, g_eff, support_window=None,
                    left_anchor_gap=0.0, right_anchor_gap=0.0):
    if variant == "greedy":
        return float(n_visible)
    if variant not in ("two_anchor", "break_even"):
        raise ValueError(f"score_lookahead: variant {variant!r} not scoring-capable")
    support = float(t_window if support_window is None else support_window)
    if n_visible <= 0 or t_window <= 0 or support <= 0 or p_clear <= 0:
        return -np.inf
    lam = p_clear * n_visible / support
    baseline_b = p_clear * n_sched + missed_utility / avg_u
    if variant == "break_even":
        return (
            break_even_spacing_anchor_support(
                lam, support, baseline_b, left_anchor_gap, right_anchor_gap)
            - g_eff
        )
    chain = chain_anchor_support(
        lam, support, g_eff, left_anchor_gap, right_anchor_gap)
    return (chain - baseline_b) * avg_u
