"""The MOSS-Image training sigma sampler (technical report, Table 6 final stage + Eq. 4).

``sigma_j = j / T`` on the grid ``j = 1..T`` (``T = 1000``). With ``S = 1 - sigmoid(Z)``,
``Z ~ N(mu, s^2)`` and ``F`` the CDF of ``S``, the probability of grid point ``j`` is::

    p_j = (1 - rho) * (F(a * j / T) - F(a * (j - 1) / T)) / F(a) + rho / T

i.e. ``S`` truncated at the cutoff ``a`` and rescaled onto the whole grid, mixed with a ``rho``
discrete-uniform component. Defaults are the report's final stage: ``mu = -1.09``, ``s = 0.95``,
``a = 0.95``, ``rho = 0.01``, no shift, no forced ``sigma = 1`` mass.
"""

from __future__ import annotations

import torch

GRID = 1000
# MOSS-Image technical report, Table 6 (final stage, steps 274,001-310,000) / Eq. 4.
DEFAULT_MU = -1.09
DEFAULT_STD = 0.95
DEFAULT_CUTOFF = 0.95
DEFAULT_UNIFORM_MIX = 0.01


def _cdf(x: torch.Tensor, mu: float, std: float) -> torch.Tensor:
    """``F(x) = Pr(1 - sigmoid(Z) <= x) = Phi((logit(x) + mu) / s)`` for ``x`` in ``[0, 1)``."""
    return torch.special.ndtr((torch.logit(x) + mu) / std)


def grid_pmf(
    mu: float = DEFAULT_MU,
    std: float = DEFAULT_STD,
    cutoff: float = DEFAULT_CUTOFF,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    grid: int = GRID,
) -> torch.Tensor:
    """The pmf over ``sigma_j = j / grid`` (``j = 1..grid``, index ``j - 1``), float64, sums to 1."""
    if not (std > 0 and 0 < cutoff < 1 and 0 <= uniform_mix <= 1):
        raise ValueError("timestep sampler: need std > 0, 0 < cutoff < 1 and 0 <= uniform_mix <= 1")
    edges = torch.arange(grid + 1, dtype=torch.float64) / grid
    cdf = _cdf(cutoff * edges, mu, std)
    mass = (cdf[1:] - cdf[:-1]) / cdf[-1]
    return (1 - uniform_mix) * mass + uniform_mix / grid


def sample_sigmas(
    batch_size: int,
    device,
    quantile: float | None = None,
    *,
    mu: float = DEFAULT_MU,
    std: float = DEFAULT_STD,
    cutoff: float = DEFAULT_CUTOFF,
    uniform_mix: float = DEFAULT_UNIFORM_MIX,
    grid: int = GRID,
) -> torch.Tensor:
    """``(batch_size,)`` float32 sigmas on the grid. ``quantile`` returns that quantile of the pmf
    (deterministic, for validation probes) instead of sampling."""
    pmf = grid_pmf(mu, std, cutoff, uniform_mix, grid)
    if quantile is not None:
        index = int(torch.searchsorted(pmf.cumsum(0), torch.tensor(float(quantile), dtype=torch.float64)))
        index = min(index, grid - 1)
        return torch.full((batch_size,), (index + 1) / grid, dtype=torch.float32, device=device)
    index = torch.multinomial(pmf, batch_size, replacement=True)
    return ((index + 1).double() / grid).float().to(device)
