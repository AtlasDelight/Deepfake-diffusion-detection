import torch
import torch.nn.functional as F
import math


def _stats_per_sample(x: torch.Tensor) -> dict:
    """
    Compute scalar statistics for each sample in a batch.

    Input:
        x: tensor with shape [B, ...]

    Output:
        dictionary of tensors with shape [B]
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
    Estimate the clean latent from the noisy latent and predicted noise.

    Priority:
        1. Use wrapper.estimate_clean_latent_from_noise if available.
        2. Otherwise use the standard formula:

           z_hat = (z_t - sqrt(1-gamma) * epsilon_hat) / sqrt(gamma)

    This is adapted to the T2I-ImageNet / CAD-I latent diffusion setting.
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


def _haar_dwt2(x: torch.Tensor) -> dict:
    """
    Compute a one-level 2D Haar wavelet decomposition.

    Input:
        x: tensor with shape [B, C, H, W]

    Output:
        dictionary containing:
            LL: low-low band
            LH: low-high band
            HL: high-low band
            HH: high-high band

    Notes:
        - This implementation uses torch only.
        - No pywt dependency is required.
        - The same Haar filters are applied independently to each channel.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape [B, C, H, W], got {x.shape}")

    b, c, h, w = x.shape

    pad_h = h % 2
    pad_w = w % 2

    if pad_h != 0 or pad_w != 0:
        x = F.pad(
            x,
            pad=(0, pad_w, 0, pad_h),
            mode="reflect",
        )

    device = x.device
    dtype = x.dtype

    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    h0 = torch.tensor(
        [inv_sqrt2, inv_sqrt2],
        device=device,
        dtype=dtype,
    )

    h1 = torch.tensor(
        [inv_sqrt2, -inv_sqrt2],
        device=device,
        dtype=dtype,
    )

    ll = torch.outer(h0, h0)
    lh = torch.outer(h0, h1)
    hl = torch.outer(h1, h0)
    hh = torch.outer(h1, h1)

    base_filters = torch.stack(
        [ll, lh, hl, hh],
        dim=0,
    ).unsqueeze(1)

    filters = base_filters.repeat(c, 1, 1, 1)

    coeffs = F.conv2d(
        x,
        filters,
        stride=2,
        padding=0,
        groups=c,
    )

    _, _, h2, w2 = coeffs.shape

    coeffs = coeffs.view(b, c, 4, h2, w2)

    return {
        "LL": coeffs[:, :, 0],
        "LH": coeffs[:, :, 1],
        "HL": coeffs[:, :, 2],
        "HH": coeffs[:, :, 3],
    }


def _energy_per_sample(x: torch.Tensor) -> torch.Tensor:
    """
    Compute average squared energy for each sample.

    Input:
        x: tensor with shape [B, ...]

    Output:
        tensor with shape [B]
    """
    return (x.float() ** 2).flatten(start_dim=1).mean(dim=1)


@torch.no_grad()
def extract_wavelet_reconstruction_single(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    return_maps: bool = True,
) -> dict:
    """
    Single-step wavelet reconstruction error for T2I-ImageNet / CAD-I.

    Pipeline:
        image x
        -> encode_image(x)
        -> latent z
        -> add_noise(z, gamma)
        -> noisy latent z_gamma
        -> predict_noise(z_gamma, gamma, prompt)
        -> epsilon_hat
        -> estimate clean latent z_hat
        -> decode_latent(z_hat)
        -> reconstructed image x_hat
        -> residual r = x - x_hat
        -> Haar wavelet decomposition of r
        -> energies in LL, LH, HL, HH bands


    """
    device = getattr(wrapper, "device", x.device)
    x = x.to(device)

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
    abs_residual = residual.abs()

    bands = _haar_dwt2(residual.float())

    ll_energy = _energy_per_sample(bands["LL"])
    lh_energy = _energy_per_sample(bands["LH"])
    hl_energy = _energy_per_sample(bands["HL"])
    hh_energy = _energy_per_sample(bands["HH"])

    detail_energy = (lh_energy + hl_energy + hh_energy) / 3.0
    total_wavelet_energy = (ll_energy + lh_energy + hl_energy + hh_energy) / 4.0

    detail_ll_ratio = detail_energy / (ll_energy + 1e-12)
    hh_ll_ratio = hh_energy / (ll_energy + 1e-12)

    residual_scores = _stats_per_sample(abs_residual)

    scores = {
        **residual_scores,
        "ll_energy": ll_energy,
        "lh_energy": lh_energy,
        "hl_energy": hl_energy,
        "hh_energy": hh_energy,
        "detail_energy": detail_energy,
        "total_wavelet_energy": total_wavelet_energy,
        "detail_ll_ratio": detail_ll_ratio,
        "hh_ll_ratio": hh_ll_ratio,
    }

    output = {
        "scores": scores,
    }

    if return_maps:
        output.update(
            {
                "x_hat": x_hat,
                "residual": residual,
                "abs_residual": abs_residual,
                "wavelet_LL": bands["LL"],
                "wavelet_LH": bands["LH"],
                "wavelet_HL": bands["HL"],
                "wavelet_HH": bands["HH"],
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
def extract_wavelet_reconstruction_trajectory(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | tuple[float, ...] = (0.95, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10),
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    fixed_noise: bool = True,
    return_maps: bool = False,
) -> dict:
    """
    Multi-step / multi-gamma wavelet reconstruction trajectory.

    For each gamma:
        image x
        -> latent z
        -> noisy latent z_gamma
        -> predicted noise epsilon_hat
        -> clean latent z_hat
        -> reconstructed image x_hat
        -> residual r = x - x_hat
        -> Haar wavelet decomposition
        -> LL, LH, HL, HH energies

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
        wavelet_ll = []
        wavelet_lh = []
        wavelet_hl = []
        wavelet_hh = []
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
        abs_residual = residual.abs()

        bands = _haar_dwt2(residual.float())

        ll_energy = _energy_per_sample(bands["LL"])
        lh_energy = _energy_per_sample(bands["LH"])
        hl_energy = _energy_per_sample(bands["HL"])
        hh_energy = _energy_per_sample(bands["HH"])

        detail_energy = (lh_energy + hl_energy + hh_energy) / 3.0
        total_wavelet_energy = (ll_energy + lh_energy + hl_energy + hh_energy) / 4.0

        detail_ll_ratio = detail_energy / (ll_energy + 1e-12)
        hh_ll_ratio = hh_energy / (ll_energy + 1e-12)

        residual_scores = _stats_per_sample(abs_residual)

        scores = {
            **residual_scores,
            "ll_energy": ll_energy,
            "lh_energy": lh_energy,
            "hl_energy": hl_energy,
            "hh_energy": hh_energy,
            "detail_energy": detail_energy,
            "total_wavelet_energy": total_wavelet_energy,
            "detail_ll_ratio": detail_ll_ratio,
            "hh_ll_ratio": hh_ll_ratio,
        }

        score_list.append(scores)

        if return_maps:
            x_hats.append(x_hat.detach().cpu())
            residuals.append(residual.detach().cpu())
            wavelet_ll.append(bands["LL"].detach().cpu())
            wavelet_lh.append(bands["LH"].detach().cpu())
            wavelet_hl.append(bands["HL"].detach().cpu())
            wavelet_hh.append(bands["HH"].detach().cpu())
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
                "wavelet_LL": torch.stack(wavelet_ll, dim=1),
                "wavelet_LH": torch.stack(wavelet_lh, dim=1),
                "wavelet_HL": torch.stack(wavelet_hl, dim=1),
                "wavelet_HH": torch.stack(wavelet_hh, dim=1),
                "z_hats": torch.stack(z_hats, dim=1),
            }
        )

    return output