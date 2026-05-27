import torch


def _stats_per_sample(x: torch.Tensor) -> dict:
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
        1. Use wrapper.estimate_clean_latent_from_noise if it exists.
        2. Otherwise use the standard formula:
           z_hat = (z_t - sqrt(1-gamma) * epsilon_hat) / sqrt(gamma)

    This keeps the code compatible with your T2IImageNetWrapper.
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


@torch.no_grad()
def extract_pixel_reconstruction_single(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    return_maps: bool = True,
) -> dict:
    """
    Single-step pixel reconstruction error for T2I-ImageNet / CAD-I.

    This version is adapted to your wrapper.

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
        -> pixel residual |x - x_hat|

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
    error_map = residual.abs()

    output = {
        "scores": _stats_per_sample(error_map),
    }

    if return_maps:
        output.update(
            {
                "x_hat": x_hat,
                "residual": residual,
                "error_map": error_map,
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
    Stack a list of score dictionaries along the gamma dimension.

    Input:
        score_dicts: list of dictionaries.
        Each dictionary contains tensors of shape [B].

    Output:
        dictionary containing tensors of shape [B, K],
        where K is the number of gammas.
    """
    keys = score_dicts[0].keys()

    return {
        key: torch.stack([scores[key] for scores in score_dicts], dim=1)
        for key in keys
    }


@torch.no_grad()
def extract_pixel_reconstruction_trajectory(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | tuple[float, ...] = (0.95, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10),
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    fixed_noise: bool = True,
    return_maps: bool = False,
) -> dict:
    """
    Multi-step / multi-gamma pixel reconstruction trajectory.

    For each gamma:
        image x
        -> latent z
        -> noisy latent z_gamma
        -> predicted noise epsilon_hat
        -> clean latent z_hat
        -> reconstructed image x_hat
        -> pixel error |x - x_hat|

    Args:
        x:
            Batch of images, shape [B, 3, H, W].
        wrapper:
            T2I-ImageNet / CAD-I wrapper.
        gammas:
            List of noise levels.
        prompt:
            Text conditioning.
        noise:
            Optional fixed noise tensor with same shape as z.
        fixed_noise:
            If True, use the same epsilon for all gammas.
            If False, a new noise is sampled at each gamma.
        return_maps:
            If True, returns maps for each gamma.
            Keep False for dataset extraction to avoid memory issues.

    Returns:
        {
            "gammas": tensor [K],
            "scores": {
                "mean": tensor [B, K],
                "std": tensor [B, K],
                "l1": tensor [B, K],
                "l2": tensor [B, K],
                "max": tensor [B, K],
            }
        }
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
        error_maps = []
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
        error_map = residual.abs()

        score_list.append(_stats_per_sample(error_map))

        if return_maps:
            x_hats.append(x_hat.detach().cpu())
            residuals.append(residual.detach().cpu())
            error_maps.append(error_map.detach().cpu())
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
                "error_maps": torch.stack(error_maps, dim=1),
                "z_hats": torch.stack(z_hats, dim=1),
            }
        )

    return output