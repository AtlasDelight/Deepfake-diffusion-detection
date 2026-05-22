# FIRE-like

This folder contains a lightweight adaptation inspired by FIRE.

The goal is not to reproduce the official FIRE implementation.  
Instead, we keep the central idea of comparing the original image and its reconstruction in the frequency domain.

## Principle

Given an input image \(x\), a lightweight pretrained generative model produces a reconstructed image \(\hat{x}\).

The image and its reconstruction are then transformed into the frequency domain:

```math
\mathcal{F}(x),
\qquad
\mathcal{F}(\hat{x})
```

The FIRE-like signal is computed as:

```math
M(x)
=
\left|
\mathcal{F}(x)
-
\mathcal{F}(\hat{x})
\right|
```

## Pipeline

```text
input image x
    -> lightweight pretrained model
    -> reconstructed image x_hat
    -> Fourier transform of x
    -> Fourier transform of x_hat
    -> frequency error map |F(x) - F(x_hat)|
```

## Frequency bands

This implementation can optionally focus on different frequency bands:

```text
all   -> all frequencies
low   -> low-frequency components
mid   -> middle-frequency components
high  -> high-frequency components
```

## Difference from official FIRE

Official FIRE uses a frequency-guided reconstruction error strategy designed specifically for robust detection of diffusion-generated images.

This implementation is a simplified FIRE-inspired approximation. It uses the reconstruction produced by the lightweight T2I-ImageNet-based wrapper and computes the frequency error between the input image and the reconstructed image.