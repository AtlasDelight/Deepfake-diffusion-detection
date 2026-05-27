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
    Estimate the clean latent.

    This function first tries to use your wrapper method:
        wrapper.estimate_clean_latent_from_noise(...)

    If the method is not available, it falls back to the standard formula.
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
def extract_latent_reconstruction_single(
    x: torch.Tensor,
    wrapper,
    gamma: float = 0.5,
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    return_maps: bool = True,
) -> dict:
    """
    Single-step latent reconstruction error for T2I-ImageNet / CAD-I.

    Pipeline:
        image x
        -> latent z
        -> noisy latent z_gamma
        -> predicted noise epsilon_hat
        -> reconstructed clean latent z_hat
        -> latent error |z - z_hat|


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

    latent_error = torch.abs(z - z_hat)

    output = {
        "scores": _stats_per_sample(latent_error),
    }

    if return_maps:
        output.update(
            {
                "z": z,
                "z_gamma": z_gamma,
                "z_hat": z_hat,
                "epsilon": epsilon,
                "epsilon_hat": epsilon_hat,
                "latent_error": latent_error,
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
def extract_latent_reconstruction_trajectory(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | tuple[float, ...] = (0.95, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10),
    prompt: str = "a photo",
    noise: torch.Tensor | None = None,
    fixed_noise: bool = True,
    return_maps: bool = False,
) -> dict:
    """
    Multi-step / multi-gamma latent reconstruction trajectory.

    For each gamma:
        image x
        -> latent z
        -> noisy latent z_gamma
        -> predicted noise epsilon_hat
        -> reconstructed latent z_hat
        -> latent error |z - z_hat|

    Returns:
        scores with shape [B, K],
        where K = number of gammas.
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
        z_gammas = []
        z_hats = []
        latent_errors = []
        epsilons = []
        epsilon_hats = []

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

        latent_error = torch.abs(z - z_hat)

        score_list.append(_stats_per_sample(latent_error))

        if return_maps:
            z_gammas.append(z_gamma.detach().cpu())
            z_hats.append(z_hat.detach().cpu())
            latent_errors.append(latent_error.detach().cpu())
            epsilons.append(epsilon.detach().cpu())
            epsilon_hats.append(epsilon_hat.detach().cpu())

    output = {
        "gammas": torch.tensor(gammas, dtype=torch.float32),
        "scores": _stack_score_dict(score_list),
    }

    if return_maps:
        output.update(
            {
                "z": z.detach().cpu(),
                "z_gammas": torch.stack(z_gammas, dim=1),
                "z_hats": torch.stack(z_hats, dim=1),
                "latent_errors": torch.stack(latent_errors, dim=1),
                "epsilons": torch.stack(epsilons, dim=1),
                "epsilon_hats": torch.stack(epsilon_hats, dim=1),
            }
        )

    return output