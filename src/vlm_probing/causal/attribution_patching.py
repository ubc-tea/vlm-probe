"""Gradient estimates of an explicitly directed activation replacement."""

import torch
from torch import Tensor

from ..core.precision import analysis_difference
from ..core.types import ProbeResult
from .base import BaseCausal, Runner, scalar_metric, selection_mask, validate_activation, validate_pair


class AttributionPatching(BaseCausal):
    """Compute ``(source - receiver) * grad(metric, receiver)``.

    Supply either a precomputed gradient evaluated at the receiver, or a scalar
    differentiable metric that replays computation from the receiver activation.
    Scores approximate source-into-receiver metric change; they are not exact
    patch effects. Positive values predict an increase in the caller's metric.

    With ``steps`` (and a replay ``metric``), the gradient is instead averaged
    along the straight path from the receiver to the masked replacement with a
    trapezoidal rule over ``steps`` intervals (``steps + 1`` gradients). The two
    path endpoints are actual replays, so the result also carries the measured
    replacement ``effect`` and ``completeness_error = estimated_effect - effect``,
    which tends to zero as ``steps`` grows.
    """

    name = "attribution_patching"

    def run(
        self,
        receiver: Tensor,
        source: Tensor,
        *,
        gradient: Tensor | None = None,
        metric: Runner | None = None,
        mask: Tensor | None = None,
        steps: int | None = None,
    ) -> ProbeResult:
        validate_pair(receiver, source)
        if (gradient is None) == (metric is None):
            raise ValueError("supply exactly one of gradient or metric")
        if steps is not None:
            if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
                raise ValueError("steps must be a positive integer")
            if metric is None:
                raise ValueError("path integration (steps) requires a replay metric, not a gradient")
            return self._path(receiver, source, metric, selection_mask(receiver, mask), steps)
        if metric is not None:
            with torch.enable_grad():
                point = receiver.detach().clone().requires_grad_(True)
                gradient = torch.autograd.grad(scalar_metric(metric, point), point)[0]
        validate_activation(gradient, "gradient")
        validate_pair(receiver, gradient)
        selected = selection_mask(receiver, mask)
        # Difference, products and their sum use at least FP32 (FP64 is
        # preserved); reduced-precision rounding would bias the estimate.
        dtype = torch.promote_types(torch.float32, receiver.dtype)
        receiver_wide = receiver.to(dtype)
        delta = torch.where(selected, source.to(dtype) - receiver_wide, torch.zeros_like(receiver_wide))
        attribution = delta * gradient.to(dtype)
        return ProbeResult(
            method=self.name,
            tensors={"attribution": attribution, "estimated_effect": attribution.sum(),
                     "gradient": gradient, "delta": delta, "mask": selected},
            metadata={"approximation": "first_order", "gradient_at": "receiver",
                      "sign": "source_metric - receiver_metric", "analysis_dtype": str(dtype)},
        )

    def _path(self, receiver: Tensor, source: Tensor, metric: Runner, selected: Tensor,
              steps: int) -> ProbeResult:
        # Path points are formed in at least FP32 and replayed in the native
        # dtype. Endpoints are selected, not reconstructed, so the final replay
        # is exactly the masked replacement even when the subtraction rounds.
        dtype = torch.promote_types(torch.float32, receiver.dtype)
        baseline = receiver.detach().to(dtype)
        delta = torch.where(selected, source.detach().to(dtype) - baseline, torch.zeros_like(baseline))
        endpoint = torch.where(selected, source.detach(), receiver.detach())
        gradient_sum = torch.zeros_like(baseline)
        scores = []
        with torch.enable_grad():
            for index in range(steps + 1):
                if index == 0:
                    value = receiver.detach()
                elif index == steps:
                    value = endpoint
                else:
                    value = (baseline + (index / steps) * delta).to(receiver.dtype)
                point = value.clone().requires_grad_(True)
                score = scalar_metric(metric, point)
                gradient = torch.autograd.grad(score, point)[0]
                if not torch.isfinite(gradient).all():
                    raise ValueError("metric produced nonfinite gradients on the path")
                weight = 0.5 if index in (0, steps) else 1.0
                gradient_sum = gradient_sum + weight * gradient.to(dtype)
                if index in (0, steps):
                    scores.append(score.detach())
        average_gradient = gradient_sum / steps
        attribution = delta * average_gradient
        estimate = attribution.sum()
        effect = analysis_difference(scores[1], scores[0])
        return ProbeResult(
            method=self.name,
            tensors={"attribution": attribution, "estimated_effect": estimate,
                     "gradient": average_gradient, "delta": delta, "mask": selected,
                     "baseline_score": scores[0], "intervention_score": scores[1],
                     "effect": effect, "completeness_error": estimate - effect},
            metadata={"approximation": "path_integrated", "gradient_at": "straight_path",
                      "steps": steps, "quadrature": "trapezoid",
                      "sign": "source_metric - receiver_metric", "analysis_dtype": str(dtype)},
        )
