"""Track level randomness descriptors.

Quantify how directed vs diffusive a trajectory is, so we can order tracks from most to least random 
and use this to help understand/quantify model error.

Note, we will explore to modify this in future when graduating from synthetic to real animal movement 
data since that is in (lat, lon) space and turning angles are slighlty distorted at high latitude because
a degree of longitude shrinks. Our synthetic tracks are planar, so FOR NOW this is fine...

"""
import numpy as np
import pandas as pd


def directional_persistence(positions: np.ndarray, *, eps: float = 1e-12) -> float:
    """Mean cosine of turning angles along a track.

    This is the directional persistence parameter rho of a correlated random walk; the expected cosine of 
    the angle between consecutive step vectors. Near 0 means diffusive/random movement (turning angles ~uniform); 
    near 1 means straight, directed movement.

    Parameters
    ----------
    positions : np.ndarray
        Track coordinates, shape (T, 2), in any consistent 2D space.
    eps : float
        Steps shorter than this are dropped; their heading is undefined.

    Returns
    -------
    float
        Mean cosine of turning angles in [-1, 1], or NaN if fewer than two
        valid steps.
    """
    # (T-1, 2) step vectors
    steps = np.diff(positions, axis=0) 

    # (T-1,) step lengths
    lengths = np.hypot(steps[:, 0], steps[:, 1]) 

    # drop stationary steps
    valid = lengths > eps              
    steps, lengths = steps[valid], lengths[valid]
    if len(steps) < 2:
        return np.nan
    
    # unit step vectors
    units = steps / lengths[:, None]

    # cos(turning angle)            
    cos_turns = np.sum(units[:-1] * units[1:], axis=1)  
    return float(np.mean(cos_turns))


# GenAI Reference: I asked GenAI to give me a short helper to apply the previous function I made to each individual; 
# and this version below is what I got. Aside from adding the docstring, I did not modify it because there was 
# not need to do so, BUT I did verify the correctness. I used synthetic trajectories with analyitically know outcomes
# (straight movement, a 90 degree turn, and complete reversal), and confirmed that it returned the expected 
# persistence values of 1, 0, and -1 respectively. 
def track_randomness(
    df: pd.DataFrame,
    *,
    individual_col: str = "individual_id",
) -> pd.DataFrame:
    """Directional persistence per individual track.

    Applies `directional_persistence` to every individual in a long-format trajectory frame: one row per track. The ``persistence`` column is the
    distribution to plot; it is also the per-track value to join against model error later.

    Parameters
    ----------
    df : pd.DataFrame
        Long format with columns [individual_col, "timestamp", "lon", "lat"].
    individual_col : str
        Column identifying each track.

    Returns
    -------
    pd.DataFrame
        One row per individual: [individual_col, "n_steps", "persistence",
        "randomness"], where randomness = 1 - persistence (sort descending for
        most-random first).
    """
    df = df.sort_values([individual_col, "timestamp"])
    rows = []
    for ind, g in df.groupby(individual_col, sort=False):
        positions = g[["lon", "lat"]].to_numpy()
        rho = directional_persistence(positions)
        rows.append(
            {
                individual_col: ind,
                "n_steps": len(positions) - 1,
                "persistence": rho,
                "randomness": 1.0 - rho,
            }
        )
    return pd.DataFrame(rows)

def mean_resultant_length(turn_angles: np.ndarray) -> float:
    """
    Mean resultant length Rbar of turning angles, |mean exp(i theta)|.

    Measures how concentrated turning angles are about any single direction, so a track that keeps reversing scores 1 just like a straight 
    one; rho (the mean cosine) carries the sign. Rbar >= |rho| always.

    """
    turns = np.asarray(turn_angles, dtype=float)
    turns = turns[np.isfinite(turns)]
    if turns.size == 0:
        return np.nan
    return float(np.hypot(np.cos(turns).mean(), np.sin(turns).mean()))


def mean_resultant_length_corrected(turn_angles: np.ndarray) -> float:
    """
    Small sample bias- orrected Rbar, sqrt(max(0, (n Rbar^2 - 1) / (n - 1))).

    Under uniformly random turns the raw Rbar is about sqrt(pi / 4n), not 0, so individuals with few turns look more directed than they are. 
    The correction removes that floor. Returns NaN with fewer than two finite angles.
    """
    turns = np.asarray(turn_angles, dtype=float)
    turns = turns[np.isfinite(turns)]
    n = turns.size
    if n < 2:
        return np.nan
    rbar = mean_resultant_length(turns)
    return float(np.sqrt(max(0.0, (n * rbar**2 - 1.0) / (n - 1.0))))
