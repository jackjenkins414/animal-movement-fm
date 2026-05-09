"""
Synthetic movement data generators.

We initially use synthetic data for 3 key reasons: 
1. Ground truth is known, so we can compute the Bayes-optimal 
   predictor and verify our models do behave as theory predicts. 
2. Bugs that hide inside of real data ('is the transformer better 
   than the LSTM by 0.2km?' is tough to answer) become obvious 
   when a model fails to match a theoretical optimum. 
3. Dataset of escalating difficulty tests different model capabilities
   isolation: BM tests the eval harness, CRW tests autoregression, 
   switching CRW tests memory. 

All generators return a pandas Df with columns: 
    individual_id, timestamp, lon, lat
This serves as a subset of the canonical Movebank schema, so models 
trained here plug into our real data unchanged. 
"""

from __future__ import annotations

import numpy as np 
import pandas as pd

# ------------------------------
# 1. BM (random walk)
# ------------------------------

def generate_brownian_motion(
        n_individuals: int = 10, 
        n_steps: int = 1000,
        step_std: float = 0.01,
        start_lon: float = 0.0,
        start_lat: float = 0.0,
        timestep_seconds: int = 3600,
        seed: int | None = None, 
) -> pd.DataFrame:
    """
    Generate independent 2D BM trajectories. 

    Each step is x_{t+1} = x+t + e_t where e_t ~ N(0, step_std^2*I).
    There is no autocorrelation; persistence (next = currnet) is the 
    Bayes-optimal predictor. Any model that beats persistence here has a bug. 

    Units: degrees of lon/lat (so step_std=0.01 ~ 1.1 km at the equator). This 
    is deliberate choice so the synthetic data is in the same units as the real 
    Movebank data and the haversine metric is meaningful. 

    Parameters
    -----
    n_individuals: number of independent trajectories. 
    n_steps: observations per trajectory (so total_rows = n_individuals * n_steps). 
    step_std: standard deviation of each per-step displacement (degrees). 
    start_lon, start_lat: starting position for all individuals. 
    timestep_seconds: seconds between consecutive observatoons. Constant cadence here, 
    but of course real data is irregular, so future generators should sample stochastically. 
    seed: RNG seed for reproducibility. 

    Returns
    -----
    DataFrame with columns [individual_id, timestamp, lon, lat], 
    sorted by (individual_id, timestamp)
    """

    rng = np.random.default_rng(seed)

    # Generate all displacement increments at once; faster than py loop 
    # Shape: (n_individuals, n_steps, 2); 2 is dlon, dlat
    increments = rng.normal(loc=0.0, scale=step_std, size=(n_individuals, n_steps, 2))

    # A cumulative sum along the time axis gives the trajectory
    # Note; we zero out the first increment per individual before cumsum-ing bc first obs is the starting point 
    increments[:, 0, :] = 0.0 
    positions = np.cumsum(increments, axis=1)
    positions[..., 0] += start_lon
    positions[..., 1] += start_lat

    # Build long-format Df; one row per (individual, step)
    individual_ids = np.repeat(np.arange(n_individuals), n_steps)
    step_indices = np.tile(np.arange(n_steps), n_individuals)

    # Use a fixed reference start time so timestap is reproducible 
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = start_time + pd.to_timedelta(step_indices * timestep_seconds, unit="s")

    df = pd.DataFrame(
        {
            "individual_id": individual_ids,
            "timestamp": timestamps,
            "lon": positions.reshape(-1, 2)[:, 0],
            "lat": positions.reshape(-1, 2)[:, 1],
        }
    )


    return df.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)

# ------------------------------
# 2. Correlated-random walk (CRW)
# ------------------------------

def generate_crw(
        n_individuals: int = 100, 
        n_steps: int = 100, 
        step_shape: float = 2.0, 
        step_scale: float = 0.005, 
        turning_kappa: float = 4.0, 
        start_lon: float = 0.0, 
        start_lat: float = 0.0, 
        timestep_seconds: int = 3600, 
        seed: int | None = None, 
) -> pd.DataFrame: 
    """
    Generate CRW trajectories.

    At each step, draw a step length from Gamma (shape, scale), draw a turning angle
    (angular deviation from previous heading) from von Mises(0, kappa), then displace. 
    High kappa => strongly persistent direction; kappa=0 uniform turning angle => 
    essenntially BM. 

    This is standard CRW formulation within the literature (ADD MY PAPER REF LATER). 
    Persistence is no longer Bayes-optimal; an autoregressive model that picks up the
    heading should beat it. XGBoost with lag features and LSTM/Transformer should (in 
    theory) all be able to learn this. 

    Parameters
    -----
    step_shape, step_scale: Gamma distribution parameters for step length (degrees). The 
        mean step length is shape * scale. 
    turning_kappa: concentration of the von Mises turning angle distribution. 
        kappa=0 is uniform (no directional persistence); kappa ~10+ is persistent. 
    """
    rng = np.random.default_rng(seed)
    # Allocate output array Easier to fil row-by-row here bc heading at step t=1 depends 
    # on the heading at step t, so we can't fully vectorise 
    positions = np.zeros((n_individuals, n_steps, 2))
    positions[:, 0, 0] = start_lon
    positions[:, 0, 1] = start_lat

    # Random initial heading per individual, uniform on [0, 2pi]
    headings = rng.uniform(0.0, 2.0 * np.pi, size=n_individuals)

    # Pre draw all step lengths and turning angles, this still vectorises the random sampling 
    # even though the position update is sequential 
    step_lengths = rng.gamma(shape=step_shape, scale=step_scale, size=(n_individuals, n_steps))
    turning_angles = rng.vonmises(mu=0.0, kappa=turning_kappa, size=(n_individuals, n_steps))

    for t in range (1, n_steps): 
        headings = headings + turning_angles[:, t]

        # Wrap headings to [-pi, pi] to avoid drift in the float representation 
        headings = np.arctan2(np.sin(headings), np.cos(headings))

        dx = step_lengths[:, t] * np.cos(headings)
        dy = step_lengths[:, t] * np.sin(headings)
        positions[:, t, 0] = positions[:, t - 1, 0] + dx 
        positions[:, t, 1] = positions[:, t - 1, 1] + dy

    individual_ids = np.repeat(np.arange(n_individuals), n_steps)
    step_indices = np.tile(np.arange(n_steps), n_individuals)
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = start_time + pd.to_timedelta(step_indices * timestep_seconds, unit="s")

    df = pd.DataFrame(
        {
            "individual_id": individual_ids, 
            "timestamp": timestamps, 
            "lon": positions.reshape(-1, 2)[:, 0],
            "lat": positions.reshape(-1, 2)[:, 1], 
        }
    )

    return df.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)


# ------------------------------
# 3. State-switching CRW 
# ------------------------------
def generate_switching_crw(
    n_individuals: int = 10,
    n_steps: int = 1000,
    foraging_step_scale: float = 0.001,
    foraging_kappa: float = 0.5,
    transit_step_scale: float = 0.01,
    transit_kappa: float = 8.0,
    p_stay: float = 0.95,
    step_shape: float = 2.0,
    start_lon: float = 0.0,
    start_lat: float = 0.0,
    timestep_seconds: int = 3600,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate CRW trajectories with two latent behavioural states.

    State 0 ('foraging'): short steps, weak directional persistence.
    State 1 ('transit'): long steps, strong directional persistence.

    State transitions follow a 2 state hidden Markov chain. p_stay is the probability 
    of remaining in the current state at each step (symmetric for both states). 
    Mean run length in each state is 1 / (1 - p_stay) steps.

    Note, the latent state is included in the output as ground truth for downstream
    diagnostics (such as 'does the LSTM hidden state cluster by behavioural state?'), 
    but is NOT input to any model. Real animal movement data does not have labelled 
    states, so models must infer them implicitly.

    This is the dataset closest to real animal movement, and (hopefully) the one where
    the deep learning models should clearly differentiate from XGBoost (even with
    fixed-lag-features). Detecting the regime requires a state representation that 
    XGBoost cannot maintain.
    """
    rng = np.random.default_rng(seed)

    positions = np.zeros((n_individuals, n_steps, 2))
    positions[:, 0, 0] = start_lon
    positions[:, 0, 1] = start_lat
    states = np.zeros((n_individuals, n_steps), dtype=np.int8)
    headings = rng.uniform(0.0, 2.0 * np.pi, size=n_individuals)

    # Initial states: 50/50 split
    states[:, 0] = rng.integers(0, 2, size=n_individuals)

    # Pre-draw uniform random numbers for state transitions
    transition_draws = rng.uniform(0.0, 1.0, size=(n_individuals, n_steps))

    for t in range(1, n_steps):
        # Markov transition: with prob p_stay, keep current state; else flip
        flip = transition_draws[:, t] > p_stay
        states[:, t] = np.where(flip, 1 - states[:, t - 1], states[:, t - 1])

        # Per-individual step length and kappa depend on current state
        scale_t = np.where(
            states[:, t] == 0, foraging_step_scale, transit_step_scale
        )
        kappa_t = np.where(states[:, t] == 0, foraging_kappa, transit_kappa)

        step_lengths = rng.gamma(shape=step_shape, scale=scale_t)

        # vonmises in numpy doesn't broadcast kappa, so we sample per individual
        # For modest n_individuals this is fast enough
        turning_angles = np.array(
            [rng.vonmises(0.0, k) for k in kappa_t]
        )

        headings = headings + turning_angles
        headings = np.arctan2(np.sin(headings), np.cos(headings))

        dx = step_lengths * np.cos(headings)
        dy = step_lengths * np.sin(headings)
        positions[:, t, 0] = positions[:, t - 1, 0] + dx
        positions[:, t, 1] = positions[:, t - 1, 1] + dy

    individual_ids = np.repeat(np.arange(n_individuals), n_steps)
    step_indices = np.tile(np.arange(n_steps), n_individuals)
    start_time = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = start_time + pd.to_timedelta(step_indices * timestep_seconds, unit="s")

    df = pd.DataFrame(
        {
            "individual_id": individual_ids,
            "timestamp": timestamps,
            "lon": positions.reshape(-1, 2)[:, 0],
            "lat": positions.reshape(-1, 2)[:, 1],
            "state": states.reshape(-1),  # ground truth label, NOT a model input
        }
    )

    return df.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)




