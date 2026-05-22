# LaRE-like

This folder contains a lightweight adaptation inspired by LaRE².

The goal is not to reproduce the official LaRE² implementation.  
Instead, we keep the central idea of exploiting a latent reconstruction error.

## Principle

Given an input image `x`, the VAE encoder maps it to a latent representation `z`.

Then, the latent is decoded into a reconstructed image `x_hat`, and this reconstructed image is encoded again into a reconstructed latent `z_hat`.

The detection signal is the latent reconstruction error:

```math
M(x) = |z - \hat{z}|
```

# Pipeline
```text
input image x
    -> VAE encoder
    -> latent z
    -> VAE decoder
    -> reconstructed image x_hat
    -> VAE encoder
    -> reconstructed latent z_hat
    -> latent error map |z - z_hat|
```