# ruff: noqa
# type: ignore

"""Circadian biomass estimation for tropical tuna under drifting FADs.

Two interchangeable circadian modulations are provided:

* :func:`plateau_biomass` -- smooth day/night plateau with sigmoidal
  transitions, calibrated against Forget et al. (2015) (see the function
  docstring for the full rationale and references).
* :func:`cosine_biomass` -- symmetric cosine modulation.

Use :func:`circadian_biomass` to dispatch between them by name (e.g. from a
``circadian_model`` key in ``config.toml``). All functions take an
``hour_local`` (local/solar clock time) and a reference daytime biomass
``b_ref`` and return the modulated biomass in the same units as ``b_ref``.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Plateau-model defaults (see the docstring of plateau_biomass for the full
# rationale). All values from Forget et al. 2015
# ---------------------------------------------------------------------------
T_SUNRISE = 6.5  # h, local clock time; species-averaged dawn (Forget 2015)
T_SUNSET = 16.5  # h, local clock time; species-averaged dusk (Forget 2015)
K_SUNRISE = 1.1  # 1/h; corresponds to a 10-90% transition of ~4 h
K_SUNSET = 1.1  # 1/h; corresponds to a 10-90% transition of ~4 h
P_BASELINE = 0.369  # night/day biomass ratio, species-averaged (Forget 2015)

# ---------------------------------------------------------------------------
# Cosine-model defaults (symmetric day/night, zero floor at midnight)
# ---------------------------------------------------------------------------
HOUR_PERIODICITY = 24.0
COSINE_CENTER = 0.5
COSINE_AMPLITUDE = 0.5


def plateau_biomass(
    hour_local,
    b_ref,
    t_sunrise=T_SUNRISE,
    t_sunset=T_SUNSET,
    k_sunrise=K_SUNRISE,
    k_sunset=K_SUNSET,
    p_baseline=P_BASELINE,
):
    """Estimate tuna biomass under a drifting FAD from local hour and a
    daytime reference biomass.

    The circadian modulation is a smooth day/night plateau with sigmoidal
    transitions centred on species-averaged sunrise (t_r) and sunset (t_s)
    times. The night biomass is a fixed fraction P of the day biomass.

    Model
    -----
    biomass(t) = b_ref * h(t)
    h(t)       = P + (1 - P) * sigma(k_r * (t - t_r)) * sigma(-k_s * (t - t_s))
    sigma(x)   = 1 / (1 + exp(-x))

    By construction, h(t) is bounded in [P, 1] and monotonically increases
    from P at night to 1 at midday. The min(1, .) clip is retained for
    numerical robustness under alternative parameter choices.

    Calibration (all values from Forget et al. 2015)
    ---------------------------------------------------------------------
    Transition centres t_r and t_s are the arithmetic mean of the reported
    peak/decline times of the three tropical tuna species (skipjack,
    yellowfin, bigeye) in the Mozambique Channel and the Seychelles:

        t_r = mean(6.0, 6.5, 7.0) = 6.5   (skipjack 06:00, yellowfin 06:30,
                                            bigeye 07:00)
        t_s = mean(15.0, 17.0, 17.5) = 16.5 (skipjack 15:00, yellowfin
                                              17:00, bigeye 17:30)

    Transition sharpness k_r and k_s are derived from visually estimated
    10%-90% transition widths in Fig. 2 of Forget et al. (2015), averaged
    across species and regions to obtain approximately 4 h for both
    sunrise and sunset. Using the identity

        Delta_t_{10-90} = 2 * ln(9) / k  ~=  4.394 / k

    this yields k_r = k_s = 4.394 / 4 ~= 1.1.

    The night/day biomass ratio P is the arithmetic mean of the ratios
    read from Fig. 2 for each species and region:

        P = mean(0.0225/0.055, 0.02/0.06, 0.02/0.055) = 0.369

    Assumptions and limitations
    ---------------------------
    - Local clock time convention. In the study region, sunrise and
      sunset varied by less than 1 h year-round, so local time
      approximates solar time (Forget et al. 2015).
    - Species-averaged: does not distinguish skipjack, yellowfin, or
      bigeye individually.
    - No dawn peak term is included. Forget et al. (2015) describe the
      presence rate as alternating between a high (daytime) and a low
      (nighttime) plateau -- the behaviour this model reproduces. Their
      aggregated curves reach the daily maximum at dawn (skipjack and
      yellowfin at 06:00, bigeye at 07:00 in the Mozambique Channel), at
      the night->day transition, but this dawn maximum sits only
      marginally above the daytime plateau that follows it.

    Parameters
    ----------
    hour_local : float or array_like
        Hour of the day in local clock time, in [0, 24). Can be a scalar,
        a list, or a NumPy array of any shape.
    b_ref : float or array_like
        Reference daytime biomass, in whatever unit is appropriate for
        the application (kg, tonnes, echo-sounder intensity, etc.).
    t_sunrise, t_sunset : float, optional
        Sigmoid transition centres for dawn/dusk, in local clock hours.
    k_sunrise, k_sunset : float, optional
        Sigmoid sharpness for dawn/dusk, in 1/h.
    p_baseline : float, optional
        Night/day biomass ratio, in [0, 1].

    Returns
    -------
    biomass : float or ndarray
        Estimated biomass at the given hour(s), same broadcast shape as
        the inputs. Same units as ``b_ref``.

    References
    ----------
    .. [1] Forget, F. G., Capello, M., Filmalter, J. D., Govinden, R., Soria,
    M., Cowley, P. D., & Dagorn, L. (2015). Behaviour and vulnerability
    of target and non-target species at drifting fish aggregating
    devices (FADs) in the tropical tuna purse seine fishery determined
    by acoustic telemetry. Canadian Journal of Fisheries and Aquatic
    Sciences, 72(9), 1398-1405.
    """
    t = np.asarray(hour_local, dtype=float)
    b = np.asarray(b_ref, dtype=float)

    def sigma(x):
        return 1.0 / (1.0 + np.exp(-x))

    plateau_shape = sigma(k_sunrise * (t - t_sunrise)) * sigma(
        -k_sunset * (t - t_sunset)
    )
    h = p_baseline + (1.0 - p_baseline) * plateau_shape
    f = np.minimum(1.0, h)

    return b * f


def cosine_biomass(
    hour_local,
    b_ref,
    periodicity=HOUR_PERIODICITY,
    center=COSINE_CENTER,
    amplitude=COSINE_AMPLITUDE,
):
    """Symmetric cosine circadian modulation.

    biomass(t) = b_ref * (center + amplitude * (-cos(2*pi*t / periodicity)))

    With the defaults (center = amplitude = 0.5), the modulation factor is
    bounded in [0, 1], reaching 0 at midnight and 1 at noon.

    Parameters
    ----------
    hour_local : float or array_like
        Hour of the day in local clock time, in [0, 24).
    b_ref : float or array_like
        Reference daytime biomass.
    periodicity : float, optional
        Cycle length in hours (24 by default).
    center, amplitude : float, optional
        Vertical shift and half-range of the cosine, both 0.5 by default so
        the factor spans [0, 1].

    Returns
    -------
    biomass : float or ndarray
        Estimated biomass at the given hour(s), same units as ``b_ref``.
    """
    t = np.asarray(hour_local, dtype=float)
    b = np.asarray(b_ref, dtype=float)

    # - cos so the minimum (zero) is located at 00:00
    time_in_cycle = -np.cos(2 * np.pi * t / periodicity)

    return b * (amplitude * time_in_cycle + center)


_MODELS = {"plateau": plateau_biomass, "cosine": cosine_biomass}


def circadian_biomass(model, hour_local, b_ref, **params):
    """Dispatch to the requested circadian model by name.

    Parameters
    ----------
    model : str
        Model to use, one of ``"plateau"`` or ``"cosine"``.
    hour_local : float or array_like
        Hour of the day in local clock time, in [0, 24).
    b_ref : float or array_like
        Reference daytime biomass, broadcast-compatible with ``hour_local``.
    **params
        Model-specific keyword parameters forwarded to the chosen model
        (e.g. ``p_baseline`` for ``"plateau"``, ``periodicity`` for
        ``"cosine"``). Typically unpacked from ``config.toml``.

    Returns
    -------
    biomass : float or ndarray
        Estimated biomass at the given hour(s), same units as ``b_ref``.

    Raises
    ------
    ValueError
        If ``model`` is not a recognised model name.
    """
    try:
        model_fn = _MODELS[model]
    except KeyError:
        raise ValueError(
            f"Unknown circadian model {model!r}; choose from {sorted(_MODELS)}."
        )
    return model_fn(hour_local, b_ref, **params)


if __name__ == "__main__":
    # Quick sanity check comparing both models over a day
    hours = np.arange(0, 24, 0.5)

    print(f"{'hour':>6} {'plateau':>9} {'cosine':>9}")
    for h_val in hours:
        p = plateau_biomass(h_val, 1.0)
        c = cosine_biomass(h_val, 1.0)
        print(f"{h_val:>6.1f} {p:>9.4f} {c:>9.4f}")
