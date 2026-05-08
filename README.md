# Heat Equation Solver: Forward & Backward Euler

Numerically solves the 1D heat equation `u_t = k u_xx` using Forward and Backward Euler methods, and compares their accuracy and stability.

## Problem Setup

- **PDE:** `u_t = k u_xx`, with `k = 0.04` on `x ∈ [0, 1]`
- **Initial condition:** `u(x, 0) = 5 + 4 cos(3πx)`
- **Boundary conditions:** Neumann (`du/dx = 0`) at both ends
- **Exact solution:** `u(x, t) = 5 + 4 exp(-k(3π)² t) cos(3πx)`

## Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

## Usage

```bash
python Euler_MOL_Heat_EQ.py
```

## Output

The script produces several plots analyzing solver performance across a range of time step counts (`N = 50` to `50000`):

- **L2 and L∞ error vs. r** for both Forward and Backward Euler
- **Norm over time** showing unstable divergence (r > 0.5) vs. stable decay (r ≤ 0.5)
- **Stability comparison** of both methods on a single log-log plot
- **Convergence plot** in the stable regime (r ≤ 0.5) vs. number of time steps

## Notes

The stability parameter `r = k Δt / Δx²` governs Forward Euler stability. Forward Euler becomes unstable when `r > 0.5`; Backward Euler remains stable for all `r`.
