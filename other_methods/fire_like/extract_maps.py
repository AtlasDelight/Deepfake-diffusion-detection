import torch


def _fft_magnitude(x: torch.Tensor, log_scale: bool = True) -> torch.Tensor:
    """
    Compute centered Fourier magnitude.

    Args:
        x: image tensor [B, C, H, W]
        log_scale: whether to apply log(1 + magnitude)

    Returns:
        magnitude spectrum [B, C, H, W]
    """

    fft = torch.fft.fft2(x, dim=(-2, -1))
    fft = torch.fft.fftshift(fft, dim=(-2, -1))

    magnitude = torch.abs(fft)

    if log_scale:
        magnitude = torch.log1p(magnitude)

    return magnitude


def _frequency_mask(
    height: int,
    width: int,
    device,
    band: str = "all",
) -> torch.Tensor:
    """
    Create a radial frequency mask.

    Bands:
        all  -> keep all frequencies
        low  -> keep low frequencies
        mid  -> keep middle frequencies
        high -> keep high frequencies
    """

    if band == "all":
        return torch.ones(1, 1, height, width, device=device)

    y = torch.linspace(-0.5, 0.5, height, device=device)
    x = torch.linspace(-0.5, 0.5, width, device=device)

    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx**2 + yy**2)

    if band == "low":
        mask = radius <= 0.15
    elif band == "mid":
        mask = (radius > 0.15) & (radius <= 0.35)
    elif band == "high":
        mask = radius > 0.35
    else:
        raise ValueError(
            f"Unknown frequency band: {band}. "
            "Expected one of: all, low, mid, high."
        )

    return mask.float().unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def extract_fire_like_map(
    x: torch.Tensor,
    wrapper,
    band: str = "all",
    log_scale: bool = True,
) -> torch.Tensor:
    """
    Extract a FIRE-like frequency reconstruction error map.

    Principle:
        x -> x_hat
        M(x) = |F(x) - F(x_hat)|

    Args:
        x:
            Input image tensor [B, 3, H, W], normalized in [-1, 1].

        wrapper:
            Lightweight pretrained generative model wrapper.
            It must implement reconstruct_image(x).

        band:
            Frequency band to keep: all, low, mid, or high.

        log_scale:
            Whether to use log-magnitude spectra.

    Returns:
        frequency_error_map:
            Tensor [B, 3, H, W] in the frequency domain.
    """

    x = x.to(wrapper.device)

    x_hat = wrapper.reconstruct_image(x)
    x_hat = x_hat.clamp(-1, 1)

    fx = _fft_magnitude(x, log_scale=log_scale)
    fx_hat = _fft_magnitude(x_hat, log_scale=log_scale)

    freq_error = torch.abs(fx - fx_hat)

    _, _, h, w = freq_error.shape
    mask = _frequency_mask(h, w, device=freq_error.device, band=band)

    freq_error = freq_error * mask

    return freq_error


@torch.no_grad()
def extract_fire_like_gray_map(
    x: torch.Tensor,
    wrapper,
    band: str = "all",
    log_scale: bool = True,
) -> torch.Tensor:
    """
    Convert the FIRE-like frequency error map into a single-channel map.

    Returns:
        gray_map: [B, 1, H, W]
    """

    freq_error = extract_fire_like_map(
        x=x,
        wrapper=wrapper,
        band=band,
        log_scale=log_scale,
    )

    gray_map = freq_error.mean(dim=1, keepdim=True)

    return gray_map