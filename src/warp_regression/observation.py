"""Observation model (mean function μ) built from declarative additive terms."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

ActKind = Literal["gelu", "relu"]
TermKind = Literal["affine", "linear_multi", "mlp_on_warped", "log_trend", "envelope_sine"]


class _ScalarMLP(nn.Module):
    def __init__(self, hidden: int = 32, n_layers: int = 2, act: ActKind = "gelu") -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = 1
        act_cls = nn.GELU if act == "gelu" else nn.ReLU
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), act_cls()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z.unsqueeze(-1)).squeeze(-1)


class TimeMLP(nn.Module):
    """Scalar map f(t) from normalized calendar time."""

    def __init__(self, hidden: int = 32, n_layers: int = 2, out_init: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = 1
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            for module in self.net:
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(out_init)

    def forward(self, t_norm: Tensor) -> Tensor:
        return self.net(t_norm.unsqueeze(-1)).squeeze(-1)


class ObservationModel(nn.Module):
    """μ(warped covariates, calendar) as a sum of configured terms."""

    def __init__(self, terms: Sequence[Dict[str, Any]]) -> None:
        super().__init__()
        self.term_specs: List[Dict[str, Any]] = [dict(t) for t in terms]
        self._affine_A = nn.Parameter(torch.tensor(1.0))
        self._affine_C = nn.Parameter(torch.tensor(0.0))
        self._linear: Optional[nn.Linear] = None
        self.mlps = nn.ModuleDict()
        self.f_net: Optional[TimeMLP] = None
        self.trend_B: Optional[nn.Parameter] = None
        self.trend_C: Optional[nn.Parameter] = None
        self.zeta: Optional[nn.Parameter] = None
        self.d_raw: Optional[nn.Parameter] = None

        for i, spec in enumerate(self.term_specs):
            kind = spec["kind"]
            if kind == "affine":
                if "A_init" in spec:
                    self._affine_A = nn.Parameter(torch.tensor(float(math.log(max(spec["A_init"], 1e-6)))))
                    self._use_log_A = True
                else:
                    self._use_log_A = True
                if "C_init" in spec:
                    self._affine_C = nn.Parameter(torch.tensor(float(spec["C_init"])))
            elif kind == "linear_multi":
                n_in = len(spec.get("inputs", []))
                self._linear = nn.Linear(n_in, 1)
            elif kind == "mlp_on_warped":
                for name in spec.get("inputs", []):
                    key = f"{i}_{name}"
                    self.mlps[key] = _ScalarMLP(
                        int(spec.get("hidden", 32)),
                        int(spec.get("layers", 2)),
                        str(spec.get("act", "gelu")),  # type: ignore[arg-type]
                    )
            elif kind == "log_trend":
                self.trend_B = nn.Parameter(torch.tensor(float(spec.get("B_init", 1.0))))
                self.trend_C = nn.Parameter(torch.tensor(float(spec.get("C_init", 0.0))))
                z_init = float(min(spec.get("z_init", 0.0), 1.0 - 1e-4))
                zeta_init = float(np.log(max(1.0 - z_init, 1e-4)))
                self.zeta = nn.Parameter(torch.tensor(zeta_init, dtype=torch.float32))
            elif kind == "envelope_sine":
                self.f_net = TimeMLP(
                    int(spec.get("hidden", 32)),
                    int(spec.get("layers", 2)),
                    out_init=float(spec.get("f_init", 0.1)),
                )
                d_init = max(float(spec.get("d_init", 0.0)), 1e-6)
                self.d_raw = nn.Parameter(
                    torch.tensor(float(np.log(np.expm1(d_init))), dtype=torch.float32)
                )
            else:
                raise ValueError(f"unknown observation term kind: {kind}")

        if not hasattr(self, "_use_log_A"):
            self._use_log_A = True

    @property
    def d(self) -> Tensor:
        if self.d_raw is None:
            return torch.tensor(0.0)
        return torch.nn.functional.softplus(self.d_raw)

    @property
    def z_shift(self) -> Tensor:
        if self.zeta is None:
            return torch.tensor(0.0)
        return 1.0 - torch.exp(self.zeta)

    def closed_form_step(
        self,
        warped: Dict[str, Tensor],
        y: Tensor,
    ) -> None:
        """Least-squares update for affine / linear_multi terms (in-place)."""
        for spec in self.term_specs:
            kind = spec["kind"]
            if kind == "affine":
                name = spec.get("inputs", ["x"])[0]
                z = warped[name]
                Z = torch.stack([z, torch.ones_like(z)], dim=1)
                coef = torch.linalg.lstsq(Z, y.unsqueeze(1)).solution.squeeze(1)
                with torch.no_grad():
                    self._affine_A.copy_(torch.log(coef[0].clamp(min=1e-6)))
                    self._affine_C.copy_(coef[1])
            elif kind == "linear_multi" and self._linear is not None:
                names = list(spec.get("inputs", []))
                cols = [warped[n] for n in names]
                Z = torch.stack(cols + [torch.ones_like(cols[0])], dim=1)
                y_np = y.detach().cpu().numpy()
                Z_np = Z.detach().cpu().numpy()
                coef, _, _, _ = np.linalg.lstsq(Z_np, y_np, rcond=None)
                with torch.no_grad():
                    self._linear.weight.copy_(
                        torch.tensor(coef[: len(names)], dtype=torch.float32).view(1, -1)
                    )
                    self._linear.bias.copy_(torch.tensor([coef[len(names)]], dtype=torch.float32))

    def forward(
        self,
        warped: Dict[str, Tensor],
        calendar: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        calendar = calendar or {}
        parts: List[Tensor] = []
        for i, spec in enumerate(self.term_specs):
            kind = spec["kind"]
            if kind == "affine":
                name = spec.get("inputs", ["x"])[0]
                A = torch.exp(self._affine_A) if self._use_log_A else self._affine_A
                parts.append(A * warped[name] + self._affine_C)
            elif kind == "linear_multi":
                assert self._linear is not None
                names = list(spec.get("inputs", []))
                stacked = torch.stack([warped[n] for n in names], dim=-1)
                parts.append(self._linear(stacked).squeeze(-1))
            elif kind == "mlp_on_warped":
                acc = None
                for name in spec.get("inputs", []):
                    key = f"{i}_{name}"
                    term = self.mlps[key](warped[name])
                    acc = term if acc is None else acc + term
                assert acc is not None
                parts.append(acc)
            elif kind == "log_trend":
                assert self.trend_B is not None and self.trend_C is not None
                t_idx = calendar["t_idx"]
                parts.append(self.trend_C + self.trend_B * torch.log(t_idx - self.z_shift))
            elif kind == "envelope_sine":
                assert self.f_net is not None
                name = spec.get("inputs", ["z"])[0]
                t_norm = calendar["t_norm"]
                atten = 1.0 + self.d * (1.0 - t_norm)
                parts.append(self.f_net(t_norm) * atten * warped[name])
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out

    def extra_params(self) -> List[nn.Parameter]:
        """Parameters to optimize (excludes those updated only by closed-form LS)."""
        params: List[nn.Parameter] = []
        kinds = {s["kind"] for s in self.term_specs}
        if "affine" in kinds:
            # A,C set by LS each step — omit from Adam (parity with WarpParametricModel)
            pass
        if "linear_multi" in kinds and self._linear is not None:
            params.extend(list(self._linear.parameters()))
        if "mlp_on_warped" in kinds:
            params.extend(list(self.mlps.parameters()))
        if "log_trend" in kinds:
            assert self.trend_B is not None and self.trend_C is not None and self.zeta is not None
            params.extend([self.trend_B, self.trend_C, self.zeta])
        if "envelope_sine" in kinds:
            assert self.f_net is not None and self.d_raw is not None
            params.extend(list(self.f_net.parameters()))
            params.append(self.d_raw)
        return params

    def init_log_trend_from_fit(self, B: float, C: float, z: float) -> None:
        if self.trend_B is None or self.trend_C is None or self.zeta is None:
            return
        with torch.no_grad():
            self.trend_B.copy_(torch.tensor(float(B)))
            self.trend_C.copy_(torch.tensor(float(C)))
            z = float(min(z, 1.0 - 1e-4))
            self.zeta.copy_(torch.tensor(float(np.log(max(1.0 - z, 1e-4)))))

    def init_envelope_from_sine(self, f_init: float, d_init: float) -> None:
        if self.f_net is None or self.d_raw is None:
            return
        with torch.no_grad():
            for module in self.f_net.net:
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()
            last = self.f_net.net[-1]
            if isinstance(last, nn.Linear):
                last.bias.fill_(float(f_init))
            d_init = max(float(d_init), 1e-6)
            self.d_raw.copy_(torch.tensor(float(np.log(np.expm1(d_init)))))
