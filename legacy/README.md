# Legacy code (archived)

This folder contains the original research implementation from the `warp-2026` branch:

- **`warp_network.py`** — `ShiftNetwork` PyTorch module with Gaussian-shift discrete warping
- **`warp.py`** — NumPy discrete warp utilities (`matrix_decomp`, `shift`, `X_shift`)
- **`linear_basis.py`** — Piecewise-linear spline basis helpers

These files are kept for reference only. The active API lives in `src/warp_regression/`
(soft-warp path, dual likelihood training, unified `WarpModel`).
