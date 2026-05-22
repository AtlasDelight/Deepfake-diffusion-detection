# DNF-like

This folder contains a lightweight adaptation inspired by DNF.

The goal is not to reproduce the official DNF implementation.  
Instead, we keep the central idea of exploiting the noise predicted by a generative model as a forensic signal.

## Principle

Given an input image `x`, we first encode it into a latent representation `z`.

Then, Gaussian noise is added at a given timestep or noise level:

```math
z_t = \sqrt{\alpha_t}z + \sqrt{1-\alpha_t}\varepsilon

```
# Pipeline
```text
input image x
    -> VAE encoder
    -> latent z
    -> add Gaussian noise
    -> noisy latent z_t
    -> lightweight pretrained model
    -> predicted noise epsilon_hat
    -> noise error map |epsilon - epsilon_hat|
```
