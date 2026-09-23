"""Shared setup for Tutorial 06: the Tutorial 05 economy and path machinery.

Tutorial 06 changes preferences, not the economy. This module carries
everything the notebook reuses from Tutorial 05 (the calibration, the
consumption-share policy, the one-period economy with in-loop market
clearing) plus the path-simulation plumbing that feeds a recursive
objective. Tutorial 05 walks through the economy code cell by cell, so the
notebook imports it instead of repeating it.

The economics that is new in Tutorial 06, the Epstein-Zin backward step,
is written in the notebook and passed to `EZSPGSolver` as a function
argument, the same way `solve` takes the environment function.
"""

from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from io import StringIO

import jax
import jax.numpy as jnp

from srl import SPGSolver
from srl.utils.discretize import discrete_assets, discrete_log_ar1
from srl.utils.safe_linalg import (
    apply_A_T,
    crra_util_func,
    find_clearing_point,
    interp_two_point_nonuniform,
)

# The grids below are built at import time, so precision is set here.
jax.config.update("jax_enable_x64", True)


# --- Tutorial 05 calibration ------------------------------------------------

nb, ny, nq, nz = 200, 3, 30, 5                      # grid sizes
borrowing_limit, bond_supply = -1.0, 0.0
consumption_floor, beta = 0.001, 0.96
rho = 2.0                                           # 1/EIS, for the initial guess

b_grid = discrete_assets(0, 50 - borrowing_limit, nb) + borrowing_limit
y_grid, y_trans = discrete_log_ar1(0.6, 0.2, ny)    # idiosyncratic income
z_grid, z_trans = discrete_log_ar1(0.9, 0.02, nz)   # aggregate productivity
q_grid = jnp.linspace(0.95, 0.999, nq)              # candidate bond prices

J = nb * ny
b_dist = jnp.repeat(b_grid, ny)
y_dist = jnp.tile(y_grid, nb)
state_indices = jnp.arange(J, dtype=jnp.int32)
y_indices = state_indices % ny


# --- Tutorial 05 setup: policy parameterization -----------------------------

def from_diff_to_cshare(policy):
    """Reconstruct s(b, y, q, z) from its first price column and differences."""
    cpolicy = jnp.zeros((nb, ny, nq, nz))
    cpolicy = cpolicy.at[..., 0, :].set(policy["first_col"])
    cpolicy = cpolicy.at[..., 1:, :].set(policy["diff_col"])
    return jnp.cumsum(cpolicy, axis=2)


def out_of_bounds_penalty(cpolicy):
    """Return a non-positive penalty for shares outside [0, 1]."""
    return (
        jnp.minimum(cpolicy, 1e-6)
        - jnp.maximum(cpolicy - 1.0, -1e-6)
        - 2e-6
    ).mean()


def wealth_monotonicity_penalty(cpolicy):
    """Return a non-positive penalty when the share falls with assets."""
    return jnp.minimum(cpolicy[1:] - cpolicy[:-1], 0.0).mean()


def initial_consumption_share():
    b = b_grid[:, None, None, None]
    y = y_grid[None, :, None, None]
    q = q_grid[None, None, :, None]
    z = z_grid[None, None, None, :]
    wealth = b + y * z - borrowing_limit
    gross_return = 1.0 / q
    numerator = jnp.maximum(
        (gross_return - (beta * gross_return) ** (1.0 / rho))
        * (wealth + 1.0 / (gross_return - 1.0)),
        0.0,
    )
    return jnp.clip(numerator / wealth, 0.001, 1.0)


init_cshare = initial_consumption_share()
initial_policy = {
    "first_col": init_cshare[..., 0, :],
    "diff_col": jnp.diff(init_cshare, axis=2),
}

state_space = {
    "b": ("markov", nb),
    "y": ("markov", ny),
    "q": ("non-markov", nq),
    "z": ("non-markov", nz),
}
action_space = {
    "first_col": (0.001, 1.0, 0.5),
    "diff_col": (-0.2, 0.2, 0.0),
}


# --- Tutorial 05 setup: household transition and market clearing -------------
# One change relative to Tutorial 05: the environment returns consumption
# rather than flow utility, because a recursive objective needs the
# consumption path itself.

def EZ_AUS_func(policy, mt, z_idx, key):
    """Return the sparse transition and consumption at aggregate state z_idx."""
    del key
    cpolicy = from_diff_to_cshare(policy)
    zt = z_grid[z_idx]
    wealth = b_dist + zt * y_dist

    # Market clearing sees the population policy but not its derivative.
    share_by_q = jax.lax.stop_gradient(cpolicy)[..., z_idx].reshape(J, nq).T
    consumption_by_q = jnp.clip(
        (wealth[None, :] - borrowing_limit) * share_by_q,
        consumption_floor,
        wealth[None, :] - (borrowing_limit + consumption_floor) * q_grid[:, None],
    )
    bond_demand = jnp.sum(
        ((wealth[None, :] - consumption_by_q) / q_grid[:, None]) * mt[None, :],
        axis=1,
    )
    q_lo, q_hi, w_lo, w_hi = find_clearing_point(bond_supply, bond_demand)
    q = q_grid[q_lo] * w_lo + q_grid[q_hi] * w_hi

    share = (
        cpolicy[:, :, q_lo, z_idx] * w_lo
        + cpolicy[:, :, q_hi, z_idx] * w_hi
    ).reshape(J)
    consumption = jnp.clip(
        (wealth - borrowing_limit) * share,
        consumption_floor,
        wealth - (borrowing_limit + consumption_floor) * q,
    )
    b_next = (wealth - consumption) / q
    b_lo, b_hi, b_w_lo, b_w_hi = interp_two_point_nonuniform(b_next, b_grid)

    income_weights = y_trans[y_indices]
    next_income = jnp.arange(ny, dtype=jnp.int32)[None, :]
    cols_lo = b_lo[:, None] * ny + next_income
    cols_hi = b_hi[:, None] * ny + next_income
    weights_lo = b_w_lo[:, None] * income_weights
    weights_hi = b_w_hi[:, None] * income_weights

    rows = jnp.broadcast_to(state_indices[:, None], (J, 2 * ny))
    columns = jnp.concatenate([cols_lo, cols_hi], axis=1)
    weights = jnp.concatenate([weights_lo, weights_hi], axis=1)
    return (rows, columns, weights), consumption, z_idx


def reset_func(key):
    """The path solver samples complete z paths, so this is only an API adapter."""
    return jax.random.choice(key, nz)


# --- Path simulation and solver adapter --------------------------------------

class EZSPGSolver(SPGSolver):
    """SPG with a path-sampled objective evaluated by backward recursion.

    The backward step itself is written in the notebook and passed in as
    `backward_step`, the same way `solve` takes the environment function.
    """

    def __init__(
        self,
        *,
        risk_aversion,
        eis,
        backward_step=None,
        monotonicity_weight=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.risk_aversion = float(risk_aversion)
        self.eis = float(eis)
        self.rho = 1.0 / self.eis
        self.theta = (1.0 - self.risk_aversion) / (1.0 - self.rho)
        self.backward_step = backward_step
        self.monotonicity_weight = float(monotonicity_weight)

    def _sample_z_paths(self, key):
        """Draw one aggregate Markov path for each simulated economy."""
        def one_path(path_key):
            key_initial, key_scan = jax.random.split(path_key)
            z_initial = jax.random.choice(key_initial, nz)

            def step(carry, _):
                z_previous, key_in = carry
                key_out, key_draw = jax.random.split(key_in)
                z_next = jax.random.choice(key_draw, nz, p=z_trans[z_previous])
                return (z_next, key_out), z_next

            (_, _), z_rest = jax.lax.scan(
                step, (z_initial, key_scan), None, length=int(self.trunc_len) - 1
            )
            return jnp.concatenate([z_initial[None], z_rest])

        return jax.vmap(one_path)(jax.random.split(key, self.sample_size))

    def _forward_paths(self, policy, key, initial_distributions):
        """Store transition weights and consumption along every aggregate path."""
        z_paths = self._sample_z_paths(key)

        def one_path(path_key, initial_distribution, z_path):
            def step(mt, z_idx):
                detached_mt = jax.lax.stop_gradient(mt)
                detached_mt = detached_mt / jnp.maximum(detached_mt.sum(), 1e-12)
                A, consumption, _ = self.env_AUS_fn(
                    policy, detached_mt, z_idx, path_key
                )
                mt_next = apply_A_T(A, mt)
                _, columns, weights = A
                return mt_next, (columns, weights, consumption)

            return jax.lax.scan(step, initial_distribution, z_path)

        keys = jax.random.split(key, self.sample_size)
        return jax.vmap(one_path)(keys, initial_distributions, z_paths)

    def _ez_value(self, outputs):
        """Evaluate the unpenalized EZ objective by backward recursion."""
        columns, weights, consumption = outputs

        def backward(log_power_next, inputs):
            step_columns, step_weights, step_consumption = inputs
            value = self.backward_step(
                log_power_next,
                step_columns,
                step_weights,
                step_consumption,
                self.risk_aversion,
                self.rho,
                self.theta,
            )
            return value, None

        def value_one_path(path_inputs):
            path_columns, path_weights, path_consumption = path_inputs
            root_log_power, _ = jax.lax.scan(
                backward,
                jnp.zeros(J),
                (path_columns, path_weights, path_consumption),
                reverse=True,
            )
            return jnp.expm1(root_log_power) / (1.0 - self.risk_aversion)

        root_values = jax.vmap(value_one_path)((columns, weights, consumption))
        initial_distribution = jnp.ones(J) / J
        return jnp.mean(root_values @ initial_distribution)

    def _policy_penalty(self, policy):
        cpolicy = from_diff_to_cshare(policy)
        return (
            out_of_bounds_penalty(cpolicy)
            + self.monotonicity_weight * wealth_monotonicity_penalty(cpolicy)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _neg_simulated_utility(self, policy, key, initial_distributions):
        final_distributions, outputs = self._forward_paths(
            policy, key, initial_distributions
        )
        objective = self._ez_value(outputs) + self._policy_penalty(policy)
        return -objective, final_distributions


def crra_path_value(outputs, risk_aversion):
    """Evaluate normalized additive CRRA utility on stored path transitions."""
    columns, weights, consumption = outputs
    rows = jnp.broadcast_to(state_indices[None, None, :, None], columns.shape)

    def one_path(path_inputs):
        path_rows, path_columns, path_weights, path_consumption = path_inputs

        def step(carry, inputs):
            distribution, discount = carry
            step_rows, step_columns, step_weights, step_consumption = inputs
            flow = discount * (1.0 - beta) * jnp.sum(
                crra_util_func(step_consumption, risk_aversion) * distribution
            )
            A = (step_rows, step_columns, step_weights)
            return (apply_A_T(A, distribution), discount * beta), flow

        initial = (jnp.ones(J) / J, jnp.array(1.0))
        _, flows = jax.lax.scan(
            step,
            initial,
            (path_rows, path_columns, path_weights, path_consumption),
        )
        return flows.sum()

    return jnp.mean(jax.vmap(one_path)((rows, columns, weights, consumption)))


class CRRAPathSolver(EZSPGSolver):
    """The same path solver with the additive CRRA objective."""

    def _ez_value(self, outputs):
        return crra_path_value(outputs, self.risk_aversion)


def solve_quietly(solver, policy_guess):
    """Run one policy solve without printing the per-iteration progress bar."""
    progress_output = StringIO()
    with redirect_stdout(progress_output), redirect_stderr(progress_output):
        return solver.solve(
            float(beta),
            state_space,
            action_space,
            EZ_AUS_func,
            reset_func,
            policy=policy_guess,
        )


# --- Diagnostics used by the notebook's checks and figures -------------------

def exact_nest_gaps(solver, policy, key):
    """Objective and gradient gaps between the EZ recursion and the CRRA sum.

    Both objectives are evaluated on the same simulated paths, so at
    theta = 1 any gap is pure floating-point error.
    """
    def ez_objective(p):
        _, outputs = solver._forward_paths(p, key, solver.m0)
        return solver._ez_value(outputs)

    def crra_objective(p):
        _, outputs = solver._forward_paths(p, key, solver.m0)
        return crra_path_value(outputs, solver.risk_aversion)

    ez_value, ez_gradient = jax.value_and_grad(ez_objective)(policy)
    crra_value, crra_gradient = jax.value_and_grad(crra_objective)(policy)
    gradient_gap = jnp.sqrt(
        sum(
            jnp.sum((ez_leaf - crra_leaf) ** 2)
            for ez_leaf, crra_leaf in zip(
                jax.tree_util.tree_leaves(ez_gradient),
                jax.tree_util.tree_leaves(crra_gradient),
            )
        )
    )
    gradient_norm = jnp.sqrt(
        sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(crra_gradient))
    )
    return {
        "ez_objective": float(ez_value),
        "crra_objective": float(crra_value),
        "objective_gap": float(jnp.abs(ez_value - crra_value)),
        "relative_gradient_gap": float(gradient_gap / gradient_norm),
    }


def supply_and_clearing(result, z_idx):
    """Return aggregate bond demand and the interpolated clearing price."""
    cpolicy = result["cpolicy"]
    mt = result["solver"].m0.mean(axis=0)
    zt = z_grid[z_idx]
    wealth = b_dist + zt * y_dist
    share_by_q = cpolicy[..., z_idx].reshape(J, nq).T
    consumption_by_q = jnp.clip(
        (wealth[None, :] - borrowing_limit) * share_by_q,
        consumption_floor,
        wealth[None, :] - (borrowing_limit + consumption_floor) * q_grid[:, None],
    )
    demand = jnp.sum(
        ((wealth[None, :] - consumption_by_q) / q_grid[:, None]) * mt[None, :],
        axis=1,
    )
    q_lo, q_hi, w_lo, w_hi = find_clearing_point(bond_supply, demand)
    q_star = q_grid[q_lo] * w_lo + q_grid[q_hi] * w_hi
    residual = w_lo * demand[q_lo] + w_hi * demand[q_hi] - bond_supply
    return demand, float(q_star), float(residual)


def consumption_at_price(result, q, z_idx, y_idx):
    """Read one consumption policy from the price grid by interpolation."""
    q_lo, q_hi, w_lo, w_hi = interp_two_point_nonuniform(q, q_grid)
    share = (
        result["cpolicy"][:, y_idx, q_lo, z_idx] * w_lo
        + result["cpolicy"][:, y_idx, q_hi, z_idx] * w_hi
    )
    wealth = b_grid + z_grid[z_idx] * y_grid[y_idx]
    return jnp.clip(
        (wealth - borrowing_limit) * share,
        consumption_floor,
        wealth - (borrowing_limit + consumption_floor) * q,
    )
