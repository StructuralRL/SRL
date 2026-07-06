# data/

Pre-computed reference data used by the tutorial notebooks.

## ssj_huggett/

Steady-state solution of the Huggett (1993) economy computed with the
[sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) toolkit.
Used by the Huggett general-equilibrium tutorial (notebook 4) as a reference the
SRL policy-gradient solution is checked against.

Files (exactly the six that notebook loads):
- `a_grid.npy`: asset grid (size 200)
- `e_grid.npy`: income grid (size 3)
- `e_trans.npy`: income transition matrix
- `prices.npy`: steady-state {r, w}
- `savings.npy`: steady-state savings policy
- `distribution.npy`: steady-state wealth distribution by income state

Provenance: computed with the sequence-jacobian toolkit at the calibration the
tutorial uses.
