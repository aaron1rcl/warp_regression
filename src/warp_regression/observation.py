"""Observation mean μ as a sum of additive terms.

Each YAML ``terms:`` entry becomes one module. ``ObservationModel`` just adds
them up — there is no other logic here.

Term kinds
----------
``linear``  (aliases: ``affine``, ``linear_multi``)
    μ = w·z + b over one or more warped inputs. Fit by least squares each step
    (not Adam). One input is the usual A·z + C case.

``mlp_on_warped``
    μ = Σ_k f_k(z_k) with a small MLP per input. Trained by Adam.

``log_trend``
    Saturating log in calendar index t::

        u = t − z_shift
        μ = C + B · log(u) / (1 + γ · log(u))

``envelope_sine``
    Warped cycle with an exponential amplitude floor::

        atten(t) = a + (1 − a) · exp(−κ t)     # ∈ (a, 1]
        μ = f(t) · atten(t) · z_warped

    Here ``a`` is the long-run floor (``a_floor``) and ``κ`` is the decay rate
    (``kappa``) — not the dual-loss mixing weight ``fit_lambda``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

ActKind = Literal["gelu", "relu"]

# Softplus(raw) ≈ 0 when raw is this small (used to freeze non-negative params at 0).
_SOFTPLUS_ZERO = -20.0


def _softplus_inv(x: float) -> float:
    """raw such that softplus(raw) = x (x > 0)."""
    x = max(float(x), 1e-8)
    return float(np.log(np.expm1(x)))


def _positive_raw(value: float) -> float:
    """Unconstrained init for a softplus-parameterised non-negative quantity."""
    v = float(max(value, 0.0))
    return _softplus_inv(v) if v > 1e-8 else _SOFTPLUS_ZERO


def _a_floor_raw(a_floor: float) -> float:
    """Unconstrained init for a ∈ [0.05, 1] via 0.05 + 0.95·sigmoid(raw)."""
    u = float(np.clip((float(a_floor) - 0.05) / 0.95, 1e-4, 1.0 - 1e-4))
    return float(np.log(u / (1.0 - u)))


class _ScalarMLP(nn.Module):
    """Map a scalar series → scalar series (shape [n] → [n])."""

    def __init__(
        self,
        hidden: int = 32,
        n_layers: int = 2,
        act: ActKind = "gelu",
        *,
        bias_init: Optional[float] = None,
    ) -> None:
        super().__init__()
        act_cls = nn.GELU if act == "gelu" else nn.ReLU
        layers: List[nn.Module] = []
        width = 1
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), act_cls()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

        if bias_init is not None:
            with torch.no_grad():
                for mod in self.net:
                    if isinstance(mod, nn.Linear):
                        mod.weight.zero_()
                        mod.bias.zero_()
                last = self.net[-1]
                assert isinstance(last, nn.Linear)
                last.bias.fill_(float(bias_init))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.unsqueeze(-1)).squeeze(-1)

    def set_constant(self, value: float) -> None:
        """Reset to a constant map f(x) ≡ value."""
        with torch.no_grad():
            for mod in self.net:
                if isinstance(mod, nn.Linear):
                    mod.weight.zero_()
                    mod.bias.zero_()
            last = self.net[-1]
            assert isinstance(last, nn.Linear)
            last.bias.fill_(float(value))


class LinearTerm(nn.Module):
    """μ = w·z + b over one or more warped inputs (closed-form least squares).

    One input with optional ``A_init`` / ``C_init`` is the usual affine case.
    """

    def __init__(self, spec: Dict[str, Any]) -> None:
        super().__init__()
        inputs = spec.get("inputs")
        if inputs is None:
            inputs = ["x"]
        self.inputs = [str(n) for n in inputs]
        self.linear = nn.Linear(len(self.inputs), 1)
        if len(self.inputs) == 1 and ("A_init" in spec or "C_init" in spec):
            with torch.no_grad():
                self.linear.weight.fill_(float(spec.get("A_init", 1.0)))
                self.linear.bias.fill_(float(spec.get("C_init", 0.0)))

    @property
    def A(self) -> Tensor:
        """Weight on the first input (affine shorthand)."""
        return self.linear.weight[0, 0]

    @property
    def C(self) -> Tensor:
        return self.linear.bias[0]

    def forward(self, warped: Dict[str, Tensor], calendar: Dict[str, Tensor]) -> Tensor:
        del calendar
        cols = torch.stack([warped[n] for n in self.inputs], dim=-1)
        return self.linear(cols).squeeze(-1)

    def closed_form_step(self, warped: Dict[str, Tensor], y: Tensor) -> None:
        cols = [warped[n] for n in self.inputs]
        design = torch.stack(cols + [torch.ones_like(cols[0])], dim=1)
        coef, _, _, _ = np.linalg.lstsq(
            design.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            rcond=None,
        )
        k = len(self.inputs)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(coef[:k], dtype=torch.float32).view(1, -1))
            self.linear.bias.copy_(torch.tensor([coef[k]], dtype=torch.float32))

    def train_params(self) -> List[nn.Parameter]:
        return []


class MlpWarpedTerm(nn.Module):
    """μ = Σ_k f_k(z_k) — one scalar MLP per warped input."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = [str(n) for n in spec.get("inputs", [])]
        hidden = int(spec.get("hidden", 32))
        layers = int(spec.get("layers", 2))
        act: ActKind = str(spec.get("act", "gelu"))  # type: ignore[assignment]
        self.mlps = nn.ModuleDict(
            {name: _ScalarMLP(hidden, layers, act) for name in self.inputs}
        )

    def forward(self, warped: Dict[str, Tensor], calendar: Dict[str, Tensor]) -> Tensor:
        del calendar
        out = self.mlps[self.inputs[0]](warped[self.inputs[0]])
        for name in self.inputs[1:]:
            out = out + self.mlps[name](warped[name])
        return out

    def closed_form_step(self, warped: Dict[str, Tensor], y: Tensor) -> None:
        del warped, y

    def train_params(self) -> List[nn.Parameter]:
        return list(self.parameters())


class LogTrendTerm(nn.Module):
    """Saturating log trend: C + B · log(u) / (1 + γ · log(u)), u = t − z_shift."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        super().__init__()
        self.B = nn.Parameter(torch.tensor(float(spec.get("B_init", 1.0))))
        self.C = nn.Parameter(torch.tensor(float(spec.get("C_init", 0.0))))
        z0 = float(min(spec.get("z_init", 0.0), 1.0 - 1e-4))
        self.zeta = nn.Parameter(torch.tensor(math.log(max(1.0 - z0, 1e-4))))
        self.gamma_raw = nn.Parameter(
            torch.tensor(_positive_raw(float(spec.get("gamma_init", 0.0))))
        )

    @property
    def gamma(self) -> Tensor:
        return torch.nn.functional.softplus(self.gamma_raw)

    @property
    def z_shift(self) -> Tensor:
        return 1.0 - torch.exp(self.zeta)

    def forward(self, warped: Dict[str, Tensor], calendar: Dict[str, Tensor]) -> Tensor:
        del warped
        return self.trend_forward(calendar["t_idx"])

    def trend_forward(self, t_idx: Tensor) -> Tensor:
        log_u = torch.log(t_idx - self.z_shift)
        denom = (1.0 + self.gamma * log_u).clamp(min=1e-6)
        return self.C + self.B * (log_u / denom)

    def closed_form_step(self, warped: Dict[str, Tensor], y: Tensor) -> None:
        del warped, y

    def train_params(self) -> List[nn.Parameter]:
        return [self.B, self.C, self.zeta, self.gamma_raw]

    def init_from_fit(self, B: float, C: float, z: float, gamma: float = 0.0) -> None:
        with torch.no_grad():
            self.B.copy_(torch.tensor(float(B)))
            self.C.copy_(torch.tensor(float(C)))
            z = float(min(z, 1.0 - 1e-4))
            self.zeta.copy_(torch.tensor(math.log(max(1.0 - z, 1e-4))))
            self.gamma_raw.copy_(torch.tensor(_positive_raw(gamma)))


class EnvelopeSineTerm(nn.Module):
    """μ = f(t) · atten(t) · z_warped, atten = a + (1−a)·exp(−κ t)."""

    def __init__(self, spec: Dict[str, Any]) -> None:
        super().__init__()
        self.input = str(spec.get("inputs", ["z"])[0])
        self.f_net = _ScalarMLP(
            int(spec.get("hidden", 32)),
            int(spec.get("layers", 2)),
            act="relu",
            bias_init=float(spec.get("f_init", 0.1)),
        )
        a0 = float(spec.get("a_floor_init", spec.get("a_inf_init", 1.0)))
        k0 = float(spec.get("kappa_init", spec.get("lam_init", 0.0)))
        self.a_floor_raw = nn.Parameter(torch.tensor(_a_floor_raw(a0)))
        self.kappa_raw = nn.Parameter(torch.tensor(_positive_raw(k0)))

    @property
    def a_floor(self) -> Tensor:
        return 0.05 + 0.95 * torch.sigmoid(self.a_floor_raw)

    @property
    def kappa(self) -> Tensor:
        return torch.nn.functional.softplus(self.kappa_raw)

    def atten(self, t_norm: Tensor) -> Tensor:
        a = self.a_floor
        return a + (1.0 - a) * torch.exp(-self.kappa * t_norm)

    def cyclic_forward(self, t_norm: Tensor, z_w: Tensor) -> Tensor:
        return self.f_net(t_norm) * self.atten(t_norm) * z_w

    def forward(self, warped: Dict[str, Tensor], calendar: Dict[str, Tensor]) -> Tensor:
        return self.cyclic_forward(calendar["t_norm"], warped[self.input])

    def closed_form_step(self, warped: Dict[str, Tensor], y: Tensor) -> None:
        del warped, y

    def train_params(self) -> List[nn.Parameter]:
        return list(self.f_net.parameters()) + [self.a_floor_raw, self.kappa_raw]

    def init_from_sine(
        self, f_init: float, a_floor_init: float = 1.0, kappa_init: float = 0.0
    ) -> None:
        with torch.no_grad():
            self.f_net.set_constant(f_init)
            self.a_floor_raw.copy_(torch.tensor(_a_floor_raw(a_floor_init)))
            self.kappa_raw.copy_(torch.tensor(_positive_raw(kappa_init)))


# ``affine`` / ``linear_multi`` kept as YAML aliases for ``linear``.
_TERM_BUILDERS = {
    "linear": LinearTerm,
    "affine": LinearTerm,
    "linear_multi": LinearTerm,
    "mlp_on_warped": MlpWarpedTerm,
    "log_trend": LogTrendTerm,
    "envelope_sine": EnvelopeSineTerm,
}


def _build_term(spec: Dict[str, Any]) -> nn.Module:
    kind = spec["kind"]
    try:
        return _TERM_BUILDERS[kind](spec)
    except KeyError as exc:
        raise ValueError(f"unknown observation term kind: {kind}") from exc


class ObservationModel(nn.Module):
    """μ(warped, calendar) = Σ_i term_i(warped, calendar)."""

    def __init__(self, terms: Sequence[Dict[str, Any]]) -> None:
        super().__init__()
        self.term_specs: List[Dict[str, Any]] = [dict(t) for t in terms]
        self.terms = nn.ModuleList([_build_term(spec) for spec in self.term_specs])

    def _find(self, cls: type) -> Optional[nn.Module]:
        for term in self.terms:
            if isinstance(term, cls):
                return term
        return None

    @property
    def _linear(self) -> Optional[LinearTerm]:
        return self._find(LinearTerm)  # type: ignore[return-value]

    @property
    def _log_trend(self) -> Optional[LogTrendTerm]:
        return self._find(LogTrendTerm)  # type: ignore[return-value]

    @property
    def _envelope(self) -> Optional[EnvelopeSineTerm]:
        return self._find(EnvelopeSineTerm)  # type: ignore[return-value]

    @property
    def trend_B(self) -> Optional[nn.Parameter]:
        term = self._log_trend
        return None if term is None else term.B

    @property
    def trend_C(self) -> Optional[nn.Parameter]:
        term = self._log_trend
        return None if term is None else term.C

    @property
    def zeta(self) -> Optional[nn.Parameter]:
        term = self._log_trend
        return None if term is None else term.zeta

    @property
    def gamma_raw(self) -> Optional[nn.Parameter]:
        term = self._log_trend
        return None if term is None else term.gamma_raw

    @property
    def gamma(self) -> Tensor:
        term = self._log_trend
        return torch.tensor(0.0) if term is None else term.gamma

    @property
    def z_shift(self) -> Tensor:
        term = self._log_trend
        return torch.tensor(0.0) if term is None else term.z_shift

    @property
    def f_net(self) -> Optional[_ScalarMLP]:
        term = self._envelope
        return None if term is None else term.f_net

    @property
    def a_floor_raw(self) -> Optional[nn.Parameter]:
        term = self._envelope
        return None if term is None else term.a_floor_raw

    @property
    def kappa_raw(self) -> Optional[nn.Parameter]:
        term = self._envelope
        return None if term is None else term.kappa_raw

    @property
    def a_floor(self) -> Tensor:
        term = self._envelope
        return torch.tensor(1.0) if term is None else term.a_floor

    @property
    def kappa(self) -> Tensor:
        term = self._envelope
        return torch.tensor(0.0) if term is None else term.kappa

    def trend_forward(self, t_idx: Tensor) -> Tensor:
        term = self._log_trend
        if term is None:
            raise RuntimeError("no log_trend term")
        return term.trend_forward(t_idx)

    def atten(self, t_norm: Tensor) -> Tensor:
        term = self._envelope
        if term is None:
            raise RuntimeError("no envelope_sine term")
        return term.atten(t_norm)

    def cyclic_forward(self, t_norm: Tensor, z_w: Tensor) -> Tensor:
        term = self._envelope
        if term is None:
            raise RuntimeError("no envelope_sine term")
        return term.cyclic_forward(t_norm, z_w)

    def forward(
        self,
        warped: Dict[str, Tensor],
        calendar: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        calendar = calendar or {}
        out = self.terms[0](warped, calendar)
        for term in self.terms[1:]:
            out = out + term(warped, calendar)
        return out

    def closed_form_step(self, warped: Dict[str, Tensor], y: Tensor) -> None:
        for term in self.terms:
            term.closed_form_step(warped, y)  # type: ignore[operator]

    def extra_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for term in self.terms:
            params.extend(term.train_params())  # type: ignore[operator]
        return params

    def init_log_trend_from_fit(
        self, B: float, C: float, z: float, gamma: float = 0.0
    ) -> None:
        term = self._log_trend
        if term is not None:
            term.init_from_fit(B, C, z, gamma)

    def init_envelope_from_sine(
        self,
        f_init: float,
        a_floor_init: float = 1.0,
        kappa_init: float = 0.0,
        *,
        a_inf_init: Optional[float] = None,
        lam_init: Optional[float] = None,
    ) -> None:
        if a_inf_init is not None:
            a_floor_init = a_inf_init
        if lam_init is not None:
            kappa_init = lam_init
        term = self._envelope
        if term is not None:
            term.init_from_sine(f_init, a_floor_init, kappa_init)
