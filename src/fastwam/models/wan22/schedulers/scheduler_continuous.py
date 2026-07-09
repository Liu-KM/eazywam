import torch


_AYS_SIGMA_ANCHORS = (1.0, 0.845, 0.73, 0.587, 0.443, 0.31, 0.193, 0.116, 0.053, 0.013)
_SCHEDULE_PRESETS = {"shifted_flowmatch", "karras", "ays"}


def _as_1d_float_tensor(
    values: object,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(values, str):
        items = [float(item.strip()) for item in values.split(",") if item.strip()]
    elif isinstance(values, torch.Tensor):
        items = values
    else:
        items = [float(item) for item in values]
    tensor = torch.as_tensor(items, device=device, dtype=torch.float32).flatten()
    if tensor.numel() == 0:
        raise ValueError(f"`{name}` must contain at least one value.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"`{name}` must contain only finite values.")
    return tensor.to(dtype=dtype)


def _round_float(value: float) -> float:
    return round(float(value), 8)


def _normalize_schedule_preset(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    if text in {"", "none", "null"}:
        return None
    if text not in _SCHEDULE_PRESETS:
        supported = ", ".join(sorted(_SCHEDULE_PRESETS))
        raise ValueError(f"`schedule_preset` must be one of: {supported}; got {value!r}.")
    return text


def _summarize_1d_tensor(values: torch.Tensor, *, max_items: int = 64) -> dict[str, object]:
    detached = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    count = int(detached.numel())
    payload: dict[str, object] = {"count": count}
    if count == 0:
        payload.update({"first": None, "last": None, "min": None, "max": None})
        return payload

    float_values = [_round_float(item) for item in detached.tolist()]
    payload.update(
        {
            "first": float_values[0],
            "last": float_values[-1],
            "min": _round_float(detached.min().item()),
            "max": _round_float(detached.max().item()),
        }
    )
    if count <= max_items:
        payload["values"] = float_values
    else:
        payload["head"] = float_values[:8]
        payload["tail"] = float_values[-8:]
    return payload


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
        timesteps: object | None = None,
        sigmas: object | None = None,
        schedule_preset: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preset = _normalize_schedule_preset(schedule_preset)
        if preset is not None and (timesteps is not None or sigmas is not None):
            raise ValueError("`schedule_preset`, `timesteps`, and `sigmas` are mutually exclusive.")
        if timesteps is not None and sigmas is not None:
            raise ValueError("`timesteps` and `sigmas` are mutually exclusive.")
        if sigmas is not None:
            sigma_steps = _as_1d_float_tensor(sigmas, name="sigmas", device=device, dtype=dtype)
            if sigma_steps.numel() != num_inference_steps:
                raise ValueError(
                    "`sigmas` length must match `num_inference_steps`: "
                    f"{sigma_steps.numel()} != {num_inference_steps}"
                )
            return self._schedule_from_sigmas(sigma_steps, dtype=dtype)
        if timesteps is not None:
            timestep_steps = _as_1d_float_tensor(
                timesteps,
                name="timesteps",
                device=device,
                dtype=dtype,
            )
            if timestep_steps.numel() != num_inference_steps:
                raise ValueError(
                    "`timesteps` length must match `num_inference_steps`: "
                    f"{timestep_steps.numel()} != {num_inference_steps}"
                )
            sigma_steps = timestep_steps / float(self.num_train_timesteps)
            _, deltas = self._schedule_from_sigmas(sigma_steps, dtype=dtype)
            return timestep_steps.to(dtype=dtype), deltas

        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        if preset == "karras":
            return self._schedule_from_sigmas(
                self._karras_sigmas(num_inference_steps, device=device, dtype=dtype),
                dtype=dtype,
            )
        if preset == "ays":
            return self._schedule_from_sigmas(
                self._ays_sigmas(num_inference_steps, device=device, dtype=dtype),
                dtype=dtype,
            )

        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    def _schedule_from_sigmas(
        self,
        sigma_steps: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.any(sigma_steps < 0) or torch.any(sigma_steps > 1):
            raise ValueError("`sigmas` must be in the inclusive range [0, 1].")
        terminal = torch.zeros(1, device=sigma_steps.device, dtype=sigma_steps.dtype)
        sigma_path = torch.cat([sigma_steps, terminal])
        deltas = sigma_path[1:] - sigma_path[:-1]
        timesteps = sigma_steps * float(self.num_train_timesteps)
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    def _karras_sigmas(
        self,
        num_inference_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        rho: float = 7.0,
    ) -> torch.Tensor:
        sigma_min = 1.0 / float(self.num_train_timesteps)
        sigma_max = 1.0
        ramp = torch.linspace(0.0, 1.0, num_inference_steps, device=device, dtype=torch.float32)
        min_inv_rho = sigma_min ** (1.0 / rho)
        max_inv_rho = sigma_max ** (1.0 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas.to(dtype=dtype)

    def _ays_sigmas(
        self,
        num_inference_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        anchors = torch.tensor(_AYS_SIGMA_ANCHORS, device=device, dtype=torch.float32)
        if num_inference_steps == anchors.numel():
            return anchors.to(dtype=dtype)

        anchor_x = torch.linspace(0.0, 1.0, anchors.numel(), device=device, dtype=torch.float32)
        target_x = torch.linspace(0.0, 1.0, num_inference_steps, device=device, dtype=torch.float32)
        right = torch.searchsorted(anchor_x, target_x, right=True).clamp(1, anchors.numel() - 1)
        left = right - 1
        span = anchor_x[right] - anchor_x[left]
        weight = (target_x - anchor_x[left]) / span
        sigmas = anchors[left] + weight * (anchors[right] - anchors[left])
        return sigmas.to(dtype=dtype)

    def inference_schedule_metadata(
        self,
        *,
        num_inference_steps: int,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        shift_override: float | None = None,
        schedule_type: str = "shifted_flowmatch",
        schedule_source: str = "generated",
        schedule_preset: object | None = None,
    ) -> dict[str, object]:
        shift = self.shift if shift_override is None else float(shift_override)
        preset = _normalize_schedule_preset(schedule_preset)
        sigma_steps = timesteps.detach().to(device="cpu", dtype=torch.float64) / float(
            self.num_train_timesteps
        )
        delta_steps = deltas.detach().to(device="cpu", dtype=torch.float64)
        return {
            "scheduler_name": "fastwam_flowmatch_euler",
            "solver": "euler",
            "schedule_type": schedule_type,
            "schedule_preset": preset,
            "schedule_source": schedule_source,
            "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": shift,
            "timestep_count": int(timesteps.numel()),
            "timesteps": _summarize_1d_tensor(timesteps),
            "sigmas": _summarize_1d_tensor(sigma_steps),
            "deltas": _summarize_1d_tensor(delta_steps),
        }

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
