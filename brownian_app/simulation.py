"""
simulation.py
-------------
Pure-numpy Brownian motion simulation.
Generates N independent 3D random walkers using the correct isotropic
distribution (cos(phi) ~ Uniform, not phi ~ Uniform — avoids pole-oversampling).
"""

import numpy as np


def generate_walkers(
    n_walkers: int = 5,
    n_steps: int = 500,
    step_size: float = 1.0,
    seed: int | None = None,
) -> dict:
    """
    Simulate N independent 3D random walkers.

    Returns a dict with:
        positions    : (n_walkers, n_steps+1, 3)  — full trajectories incl. origin
        displacement : (n_walkers, n_steps+1)     — |r(t)| from origin at each step
        msd          : (n_steps+1,)               — ensemble mean squared displacement
        t            : (n_steps+1,)               — step indices (time axis)
    """
    rng = np.random.default_rng(seed)

    # Correct isotropic sampling on the unit sphere:
    # sampling phi ~ Uniform(0, π) oversamples the poles because surface area
    # element is sin(phi) dφ — not uniform. Sampling cos(phi) ~ Uniform(-1, 1)
    # is the area-preserving fix.
    theta   = rng.uniform(0, 2 * np.pi, (n_walkers, n_steps))
    cos_phi = rng.uniform(-1, 1,        (n_walkers, n_steps))
    sin_phi = np.sqrt(1 - cos_phi**2)

    dx = step_size * sin_phi * np.cos(theta)
    dy = step_size * sin_phi * np.sin(theta)
    dz = step_size * cos_phi

    # Prepend origin and cumsum to build full trajectories
    steps     = np.stack([dx, dy, dz], axis=-1)        # (n_walkers, n_steps, 3)
    origin    = np.zeros((n_walkers, 1, 3))
    positions = np.concatenate([origin, np.cumsum(steps, axis=1)], axis=1)

    displacement = np.linalg.norm(positions, axis=-1)  # (n_walkers, n_steps+1)
    msd          = np.mean(displacement**2, axis=0)    # ensemble average
    t            = np.arange(n_steps + 1)

    return {
        "positions":    positions,
        "displacement": displacement,
        "msd":          msd,
        "t":            t,
        "n_walkers":    n_walkers,
        "n_steps":      n_steps,
        "step_size":    step_size,
    }
