import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_implicit_detector_like_features(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | None = None,
    prompt=None,
    include_noise: bool = True,
    include_response: bool = True,
    include_error: bool = True,
) -> torch.Tensor:
    """
    Extract implicit detector-like features from model responses.

    Principle:
        x -> z
        z_gamma = sqrt(gamma) z + sqrt(1 - gamma) epsilon
        epsilon_hat = model_response(z_gamma, gamma, prompt)

    For each gamma, we collect:
        - epsilon
        - epsilon_hat
        - |epsilon - epsilon_hat|

    The final feature tensor is the concatenation of these responses.

    Returns:
        features: [B, C_features, H_latent, W_latent]
    """

    if gammas is None:
        gammas = [0.2, 0.5, 0.8]

    z = wrapper.encode_image(x, sample=False)

    feature_list = []

    for gamma in gammas:
        z_gamma, epsilon, gamma_tensor = wrapper.add_noise(
            z=z,
            noise=None,
            gamma=gamma,
        )

        epsilon_hat = wrapper.predict_noise(
            z_t=z_gamma,
            gamma=gamma_tensor,
            prompt=prompt,
        )

        error = torch.abs(epsilon - epsilon_hat)

        if include_noise:
            feature_list.append(epsilon)

        if include_response:
            feature_list.append(epsilon_hat)

        if include_error:
            feature_list.append(error)

    features = torch.cat(feature_list, dim=1)

    return features


@torch.no_grad()
def extract_implicit_detector_like_gray_map(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | None = None,
    prompt=None,
) -> torch.Tensor:
    """
    Convert implicit detector-like features into a single-channel map.

    Returns:
        gray_map: [B, 1, H_latent, W_latent]
    """

    features = extract_implicit_detector_like_features(
        x=x,
        wrapper=wrapper,
        gammas=gammas,
        prompt=prompt,
    )

    gray_map = features.abs().mean(dim=1, keepdim=True)

    return gray_map


@torch.no_grad()
def extract_implicit_detector_like_upsampled_map(
    x: torch.Tensor,
    wrapper,
    gammas: list[float] | None = None,
    prompt=None,
    size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Extract an implicit detector-like map and upsample it to image resolution.

    Returns:
        upsampled_map: [B, 1, H, W]
    """

    gray_map = extract_implicit_detector_like_gray_map(
        x=x,
        wrapper=wrapper,
        gammas=gammas,
        prompt=prompt,
    )

    if size is None:
        size = x.shape[-2:]

    upsampled_map = F.interpolate(
        gray_map,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    return upsampled_map