import torch


def _stats_per_sample(x: torch.Tensor) -> dict:
    """
    Compute scalar statistics for each sample in a batch.
    """
    x = x.detach().float()
    flat = x.flatten(start_dim=1)

    return {
        "mean": flat.mean(dim=1),
        "std": flat.std(dim=1, unbiased=False),
        "l1": flat.abs().mean(dim=1),
        "l2": torch.sqrt((flat ** 2).mean(dim=1) + 1e-12),
        "max": flat.abs().max(dim=1).values,
    }


def _estimate_clean_latent(
    wrapper,
    z_t: torch.Tensor,
    epsilon_hat: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate clean latent from noisy latent and predicted noise.
    """
    if hasattr(wrapper, "estimate_clean_latent_from_noise"):
        try:
            return wrapper.estimate_clean_latent_from_noise(
                z_t=z_t,
                epsilon_hat=epsilon_hat,
                gamma=gamma,
            )
        except TypeError:
            return wrapper.estimate_clean_latent_from_noise(
                z_t,
                epsilon_hat,
                gamma,
            )

    gamma = gamma.to(device=z_t.device, dtype=z_t.dtype)

    if gamma.ndim == 0:
        gamma = gamma.view(1)

    if gamma.numel() == 1:
        gamma = gamma.expand(z_t.shape[0])

    gamma = gamma.view(z_t.shape[0], *([1] * (z_t.ndim - 1)))
    gamma = gamma.clamp(1e-6, 1.0 - 1e-6)

    z_hat = (z_t - torch.sqrt(1.0 - gamma) * epsilon_hat) / torch.sqrt(gamma)

    return z_hat


def _frequency_band_masks(
    height: int,
    width: int,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build low/mid/high frequency masks after fftshift.

    Low frequencies are near the center.
    High frequencies are near the borders.
    """
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / (radius.max() + 1e-12)

    low_mask = radius <= 0.25
    mid_mask = (radius > 0.25) & (radius <= 0.50)
    high_mask = radius > 0.50

    low_mask = low_mask.to(dtype=dtype).view(1, 1, height, width)
    mid_mask = mid_mask.to(dtype=dtype).view(1, 1, height, width)
    high_mask = high_mask.to(dtype=dtype).view(1, 1, height, width)

    return low_mask, mid_mask, high_mask


def _masked_energy_per_sample(
    power: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute average energy inside a frequency band for each sample.
    """
    channels = power.shape[1]
    denom = mask.sum() * channels + 1e-12

    return (power * mask).sum(dim=(1, 2, 3)) / denom


@torch.no_grad()
def extract_frequency_reconstruction_single(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    return_maps: bool = True,
) -> dict:
    """
    Single-step frequency reconstruction error for T2I-ImageNet / CAD-I.

    Important:
        The frequency analysis is done in image space,
        but the reconstruction is obtained through the CAD-I latent pipeline.

    Pipeline:
        image x
        -> latent z
        -> noisy latent z_gamma
        -> predicted noise epsilon_hat
        -> clean latent z_hat
        -> reconstructed image x_hat
        -> residual r = x - x_hat
        -> FFT(r)
        -> low/mid/high frequency energies

    """
    x = x.to(wrapper.device)

    z = wrapper.encode_image(x, sample=False)

    z_gamma, epsilon, gamma_tensor = wrapper.add_noise(
        z=z,
        noise=noise,
        gamma=gamma,
    )

    epsilon_hat = wrapper.predict_noise(
        z_t=z_gamma,
        gamma=gamma_tensor,
        prompt=prompt,
    )

    z_hat = _estimate_clean_latent(
        wrapper=wrapper,
        z_t=z_gamma,
        epsilon_hat=epsilon_hat,
        gamma=gamma_tensor,
    )

    x_hat = wrapper.decode_latent(z_hat).clamp(-1, 1)

    residual = x - x_hat

    fft = torch.fft.fft2(
        residual.float(),
        dim=(-2, -1),
        norm="ortho",
    )

    fft = torch.fft.fftshift(fft, dim=(-2, -1))

    magnitude = torch.abs(fft)
    power = magnitude ** 2
    frequency_map = torch.log1p(magnitude)

    _, _, height, width = residual.shape

    low_mask, mid_mask, high_mask = _frequency_band_masks(
        height=height,
        width=width,
        device=residual.device,
        dtype=power.dtype,
    )

    low_energy = _masked_energy_per_sample(power, low_mask)
    mid_energy = _masked_energy_per_sample(power, mid_mask)
    high_energy = _masked_energy_per_sample(power, high_mask)
    total_energy = power.mean(dim=(1, 2, 3))

    residual_scores = _stats_per_sample(residual.abs())

    scores = {
        **residual_scores,
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "total_energy": total_energy,
        "high_low_ratio": high_energy / (low_energy + 1e-12),
    }

    output = {
        "scores": scores,
    }

    if return_maps:
        output.update(
            {
                "x_hat": x_hat,
                "residual": residual,
                "frequency_map": frequency_map,
                "z": z,
                "z_gamma": z_gamma,
                "z_hat": z_hat,
                "epsilon": epsilon,
                "epsilon_hat": epsilon_hat,
            }
        )

    return output
def _stack_score_dict(score_dicts: list[dict]) -> dict:
    """
    Stack score dictionaries along the gamma dimension.

    Each score tensor has shape [B].
    Output tensors have shape [B, K].
    """
    keys = score_dicts[0].keys()

    return {
        key: torch.stack([scores[key] for scores in score_dicts], dim=1)
        for key in keys
    }


@torch.no_grad()
def extract_frequency_reconstruction_trajectory(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | tuple[float, ...] = (0.95, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10),
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    fixed_noise: bool = True,
    return_maps: bool = False,
) -> dict:
    """
    Multi-step / multi-gamma frequency reconstruction trajectory.

    For each gamma:
        image x
        -> latent z
        -> noisy latent z_gamma
        -> predicted noise epsilon_hat
        -> clean latent z_hat
        -> reconstructed image x_hat
        -> residual r = x - x_hat
        -> FFT(r)
        -> frequency energies

    Returns:
        scores with shape [B, K].
    """
    device = getattr(wrapper, "device", x.device)
    x = x.to(device)

    z = wrapper.encode_image(x, sample=False)

    if noise is not None:
        base_noise = noise.to(device=z.device, dtype=z.dtype)
    elif fixed_noise:
        base_noise = torch.randn_like(z)
    else:
        base_noise = None

    score_list = []

    if return_maps:
        x_hats = []
        residuals = []
        frequency_maps = []
        z_hats = []

    for gamma in gammas:
        current_noise = base_noise if fixed_noise or noise is not None else None

        z_gamma, epsilon, gamma_tensor = wrapper.add_noise(
            z=z,
            noise=current_noise,
            gamma=gamma,
        )

        epsilon_hat = wrapper.predict_noise(
            z_t=z_gamma,
            gamma=gamma_tensor,
            prompt=prompt,
        )

        z_hat = _estimate_clean_latent(
            wrapper=wrapper,
            z_t=z_gamma,
            epsilon_hat=epsilon_hat,
            gamma=gamma_tensor,
        )

        x_hat = wrapper.decode_latent(z_hat).clamp(-1, 1)

        residual = x - x_hat

        fft = torch.fft.fft2(
            residual.float(),
            dim=(-2, -1),
            norm="ortho",
        )

        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        magnitude = torch.abs(fft)
        power = magnitude ** 2
        frequency_map = torch.log1p(magnitude)

        _, _, height, width = residual.shape

        low_mask, mid_mask, high_mask = _frequency_band_masks(
            height=height,
            width=width,
            device=residual.device,
            dtype=power.dtype,
        )

        low_energy = _masked_energy_per_sample(power, low_mask)
        mid_energy = _masked_energy_per_sample(power, mid_mask)
        high_energy = _masked_energy_per_sample(power, high_mask)
        total_energy = power.mean(dim=(1, 2, 3))

        residual_scores = _stats_per_sample(residual.abs())

        scores = {
            **residual_scores,
            "low_energy": low_energy,
            "mid_energy": mid_energy,
            "high_energy": high_energy,
            "total_energy": total_energy,
            "high_low_ratio": high_energy / (low_energy + 1e-12),
        }

        score_list.append(scores)

        if return_maps:
            x_hats.append(x_hat.detach().cpu())
            residuals.append(residual.detach().cpu())
            frequency_maps.append(frequency_map.detach().cpu())
            z_hats.append(z_hat.detach().cpu())

    output = {
        "gammas": torch.tensor(gammas, dtype=torch.float32),
        "scores": _stack_score_dict(score_list),
    }

    if return_maps:
        output.update(
            {
                "x_hats": torch.stack(x_hats, dim=1),
                "residuals": torch.stack(residuals, dim=1),
                "frequency_maps": torch.stack(frequency_maps, dim=1),
                "z_hats": torch.stack(z_hats, dim=1),
            }
        )

    return output