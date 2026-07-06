"""Shared calibration for the SRL tutorials.

One parameterization is shared across notebooks 0–4, so each of those describes
the *same* economy: a household problem that grows one feature at a time.
Keeping it here (rather than re-deriving it in each notebook) means they can
never drift out of calibration with one another. (Notebook 5, aggregate risk,
currently ships on the separate paper calibration; aligning it with this spine
is planned but not yet done.)

Pure NumPy (no JAX, no `srl`), so the partial-equilibrium notebooks stay
dependency-light.

Two notes on consistency with the `srl` library:
- **Income** is discretized with the **Rouwenhorst** method, which reproduces
  the reference income grid the general-equilibrium notebook validates against.
  `srl.utils.discrete_log_ar1` is a *Tauchen* discretization (a different
  method with different grids) and is **not** used for this 0–4 spine (notebook
  5's paper calibration does use it).
- **Assets** use the same geometric (double-exponential) spacing as
  `srl.utils.discrete_assets`, so the asset grids coincide.
"""
from math import comb

import numpy as np

# --- structural parameters (shared across tutorials 0–5) ---
SIGMA = 2.0        # CRRA risk aversion
BETA = 0.98        # discount factor
B_MIN = 0.0        # borrowing limit (no borrowing)
B_MAX = 50.0       # top of the asset grid
RHO = 0.966        # income persistence (AR(1) in logs)
SIGMA_Y = 0.5      # stationary standard deviation of log income
N_Y = 3            # number of income states

# Fixed bond price for the partial-equilibrium notebooks (0–2): r = 1/q − 1 =
# 0.503%, just below the general-equilibrium rate (0.562%) the market sets in
# notebook 4.
Q_FIXED = 0.995


def rouwenhorst(rho=RHO, sigma_y=SIGMA_Y, n=N_Y):
    """Discretize a log-AR(1) income process into an n-state Markov chain.

    Returns income levels ``e`` (normalized so mean income is 1 under the
    stationary distribution) and the transition matrix ``Pi``.
    """
    # Transition matrix, built recursively from the 2-state base case.
    p = (1 + rho) / 2
    Pi = np.array([[p, 1 - p], [1 - p, p]])
    for k in range(3, n + 1):
        Pn = np.zeros((k, k))
        Pn[:-1, :-1] += p * Pi
        Pn[:-1, 1:] += (1 - p) * Pi
        Pn[1:, :-1] += (1 - p) * Pi
        Pn[1:, 1:] += p * Pi
        Pn[1:-1] /= 2                  # interior rows double-count; halve them
        Pi = Pn
    # Evenly spaced log grid spanning +/- sqrt(n-1) stationary std devs.
    psi = sigma_y * np.sqrt(n - 1)
    log_e = np.linspace(-psi, psi, n)
    # Stationary distribution of a Rouwenhorst chain is Binomial(n-1, 1/2).
    pi_stat = np.array([comb(n - 1, i) * 0.5 ** (n - 1) for i in range(n)])
    e = np.exp(log_e)
    e = e / (pi_stat @ e)              # normalize: mean income = 1
    return e, Pi


def asset_grid(n, a_min=B_MIN, a_max=B_MAX, pivot=0.25):
    """Non-uniform asset grid on [a_min, a_max], dense near a_min.

    Matches ``srl.utils.discrete_assets`` (geometric / double-exponential
    spacing), which concentrates points near the borrowing constraint where the
    policy function is most curved.
    """
    a = np.geomspace(a_min + pivot, a_max + pivot, n) - pivot
    a[0] = a_min
    return a
