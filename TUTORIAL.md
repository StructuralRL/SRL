# SRL — Tutorial syllabus

This is the course map for the SRL tutorials. 

## Who this is for

We've written the tutorials with a second-year graduate student in mind. We don't assume much, but you should have seen the canonical household income-fluctuations problem before and solved it using a value function iteration method. We do **not** assume that you know anything about:
- policy gradients / reinforcement learning
- JAX and automatic differentiation 

These are the two new ideas in our paper. They are independent of each other, and the course introduces them one at a time.
In this sense, the tutorials are self-contained. Just jump in!

## How the course is built

Four principles shape the sequence. They're worth understanding up front, because they explain why the early notebooks start with a problem you already know.

1. **Start from the model you know, in the code you know.** Notebook 0 solves the household problem in plain NumPy with VFI, no JAX and no new method. It establishes the *ground-truth answer* using machinery you already trust.

2. **One new idea per notebook.** The two new innovations, JAX and the policy gradient, are never introduced together. Notebook 1 changes *only* the tooling (NumPy → JAX, same method). Notebook 2 changes *only* the method (VFI → policy gradient, same tooling).

3. **Every step checks against the previous one.** Because notebooks 0, 1, and 2 solve the *same* problem, they must produce the *same* answer. You confirm the new machinery reproduces a result you already believe before you trust it on a problem you don't. (These checks are also the repo's regression tests.)

4. **The model is held as constant as possible.** Notebooks 0–3 all use one household problem; it grows by exactly one feature at a time. You watch a single model develop rather than meeting four different ones.

## The two models

Everything is built on two models you already recognize:

- **The household problem** (partial equilibrium, so the household takes prices as given). Used in notebooks 0–3, starting from its simplest fixed-price form (which is just the income-fluctuation problem) and growing to a household that faces a stochastic *process* for prices.

- **The Huggett economy** (general equilibrium, the bond price is set by market clearing). Used in notebook 4, then extended with aggregate risk in notebook 5, the main result of the paper.

## The sequence

| # | Notebook | Model | The one new idea |
|---|----------|-------|------------------|
| 0 | [Household problem, in NumPy](tutorials_new/00_household_numpy.ipynb) | household, fixed price | the ground truth: VFI in code you already write |
| 1 | [The same problem, in JAX](tutorials_new/01_household_jax.ipynb) | household, fixed price | the tooling: `grad` / `vmap` / `jit`, taught by porting notebook 0 |
| 2 | [The same problem, by policy gradient](tutorials_new/02_household_policy_gradient.ipynb) | household, fixed price | the method: maximize lifetime utility directly by autodiff |
| 3 | [The household with moving prices](tutorials_new/03_household_moving_prices.ipynb) | household, stochastic price process | the household conditions on a *price process* |
| 4 | [The Huggett economy](tutorials_new/04_ge_huggett.ipynb) | stationary GE | the price is set by **market clearing**; price-taking as stop-gradient |
| 5 | [Huggett with aggregate risk](tutorials_new/05_ge_huggett_agg_risk.ipynb) | GE + aggregate shocks | prices replace the distribution as the state; learned from simulation |

---

## Per-notebook detail

Each notebook follows the same internal shape: **Overview** (motivation +
"what you should already know" + references) → **Model** (the economics) →
**Implementation** (the code) → **Exercises** (with worked solutions). The
household model is shown *inline* so you can read each notebook top to bottom;
the reusable solvers and helpers are imported from the `srl` package and
displayed where they matter.

### 0. The household problem, in NumPy

- **What you build.** The income-fluctuation problem at a single fixed interest
  rate: a household choosing consumption/savings against idiosyncratic income
  risk and a borrowing constraint. Solved by value function iteration in plain
  NumPy/SciPy. The object of interest is the **consumption policy**. These are
  partial-equilibrium notebooks, so there's no market clearing; the wealth
  distribution doesn't show up until notebook 4, where it's used for aggregation
  and market clearing.
- **The one new idea.** None, by design. This is the trusted answer the rest of
  the course is measured against.
- **What you should already know.** The income-fluctuation problem and VFI. If
  you have solved an Aiyagari household block, this is familiar territory.
- **Shows inline / imports.** Everything inline; no `srl` import. This is the
  "before SRL, in code you already write" baseline.
- **The check.** This *is* the reference. Save the consumption policy; notebooks
  1 and 2 must reproduce it.
- **Exercises.** Vary risk aversion and the borrowing limit; confirm the
  consumption policy moves the way intuition predicts.

### 1. The same problem, in JAX

- **What you build.** The exact model and method of notebook 0, re-expressed in
  JAX.
- **The one new idea: the tooling.** JAX's functional model: array
  immutability, pure functions, the explicit PRNG, and the three transforms
  `jit`, `vmap`, `grad`. Each is tied to something concrete: `vmap` over the
  asset grid, `jit` on the VFI step, and `grad` previewed as the object the next
  notebook is built on.
- **What you should already know.** Notebook 0. No prior JAX assumed. This *is*
  the JAX introduction, taught on a model you just solved.
- **Shows inline / imports.** Uses `VFISolver` from `srl`; the discretization
  and plotting helpers from `srl.utils`.
- **The check.** Same answer as notebook 0, to numerical tolerance. If it
  matches, you now trust the JAX tooling on familiar ground.
- **Exercises.** Take the gradient of a simple economic quantity with respect to
  a parameter and check it against a finite-difference estimate. Your first
  contact with autodiff, before it carries the method in notebook 2.

### 2. The same problem, by policy gradient

This is the notebook that matters most.

- **What you build.** The same fixed-price household problem, solved a third way:
  not by iterating a Bellman operator, but by **maximizing lifetime utility
  directly with gradient ascent**, using `SPGSolver` from `srl`.
- **The one new idea: the method.** Three steps, each starting from something
  you already use:
  1. *A value function is an expected value.* Instead of solving a Bellman
     equation, estimate `E[Σ βᵗ u(cₜ)]` by simulating paths and averaging.
  2. *The objective is differentiable in the policy.* A policy on the grid
     induces a transition matrix `A_π` (the same object as the histogram-method
     transition you built in notebook 0), so lifetime utility is a smooth
     function of the policy and its gradient exists in closed form.
     **Autodiff computes that gradient exactly**; unlike textbook reinforcement
     learning, nothing about the gradient is estimated by noisy sampling.
  3. *Climb it.* Stochastic gradient ascent on the policy.
- **What you should already know.** Notebooks 0–1. No reinforcement-learning
  background assumed.
- **Honest caveat (stated in the notebook).** On *this* problem the policy
  gradient is overkill, VFI already solves it cleanly. We use it here *because*
  you can check it against a known answer. You don't really need the policy
  gradient until notebook 5, where the methods you know start to struggle;
  notebook 2 points ahead to that so the detour doesn't look pointless. We also
  flag the assumption doing the work: exact differentiation is trustworthy
  because the grid keeps `A_π` smooth in the policy. With kinks or
  discontinuities these gradients can mislead.
- **The check.** Same consumption policy as notebooks 0–1.
- **Exercises.** Watch the loss climb across iterations; vary the learning rate
  and sample size and see the convergence/variance trade-off.

### 3. The household with moving prices

- **What you build.** The same household, but now the prices it faces (the bond
  price and aggregate productivity) follow **exogenous stochastic processes**
  that the household conditions on. Solved with the policy gradient, and checked
  against VFI on the expanded state.
- **The one new idea.** Prices are no longer a fixed number but a *process* the
  household takes as given. This is the rehearsal for how SRL will handle general
  equilibrium: even in GE, the household treats prices as an exogenous process it
  has learned.
- **What you should already know.** Notebooks 0–2.
- **Shows inline / imports.** The model grows inline from notebook 2 (the price
  dimension is switched on); solvers imported as before.
- **The check.** Policy agrees with a VFI solution on the same state space.
- **Exercises.** Inspect how the consumption policy shifts with the price state;
  relate it to the precautionary and intertemporal-substitution channels.

### 4. The Huggett economy

- **What you build.** The stationary Huggett general equilibrium: bonds in zero
  net supply, the **bond price set by market clearing**. Solved with the policy
  gradient.
- **The one new idea.** The price becomes *endogenous*. Two pieces:
  - market clearing happens *inside* the simulated environment: the household's
    policy doubles as its bond-supply schedule, and the market-clearing price is
    found on the path;
  - **price-taking is enforced as a stop-gradient on the price**: blocking the
    gradient from flowing through the market-clearing map *is* the
    competitive-equilibrium assumption.
- **What you should already know.** Notebooks 0–3, and the stationary Huggett/
  Aiyagari fixed point you have seen in coursework.
- **Shows inline / imports.** Market-clearing helper from `srl.utils`; solver
  from `srl`.
- **The check.** Consumption policy and the wealth distribution match the
  sequence-space-Jacobian reference solution shipped in `data/ssj_huggett/`.
- **Exercises.** Trace bond-market clearing as the policy converges; do
  comparative statics in the discount factor and bond supply.

### 5. Huggett with aggregate risk

- **What you build.** The Huggett economy with **aggregate productivity
  shocks**. This is the main result, and the model the classical methods find
  hard.
- **The one new idea.** With aggregate risk the cross-sectional distribution
  *would* become a state variable (this is the curse of dimensionality that makes
  the problem hard). SRL sidesteps it: the household carries a **low-dimensional
  price** in place of the distribution and **learns the price process from
  simulated paths**, rather than tracking the distribution or positing a law of
  motion for it.
- **What you should already know.** All prior notebooks; familiarity with why
  aggregate risk in heterogeneous-agent models is hard (Krusell–Smith and the
  approximate-aggregation idea) helps but is not required.
- **Shows inline / imports.** The aggregate-risk dynamics inline; solver imported.
- **The check.** The solution is validated against reference dynamics and the
  economic properties a correct solution must have.
- **Exercises.** Compare impulse responses to an aggregate shock; examine how
  much carrying only the price (not the distribution) costs in accuracy.

---

## Running the tutorials

Each notebook is self-contained and runs top to bottom, on a free Colab GPU
(click a badge below) or locally on a CPU (slower). The first cell of every
notebook handles the environment for you: on Colab it clones the repo and
installs `srl`; run locally it is a no-op and you just need the package installed
(`pip install -e .` from the repo root). There is no separate "setup" notebook.

| # | Notebook | Open in Colab |
|---|----------|:-------------:|
| 0 | [Household problem, in NumPy](tutorials_new/00_household_numpy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/00_household_numpy.ipynb) |
| 1 | [The same problem, in JAX](tutorials_new/01_household_jax.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/01_household_jax.ipynb) |
| 2 | [The same problem, by policy gradient](tutorials_new/02_household_policy_gradient.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/02_household_policy_gradient.ipynb) |
| 3 | [The household with moving prices](tutorials_new/03_household_moving_prices.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/03_household_moving_prices.ipynb) |
| 4 | [The Huggett economy](tutorials_new/04_ge_huggett.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/04_ge_huggett.ipynb) |
| 5 | [Huggett with aggregate risk](tutorials_new/05_ge_huggett_agg_risk.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StructuralRL/SRL/blob/main/tutorials_new/05_ge_huggett_agg_risk.ipynb) |


