# DIRE-like

This folder contains a lightweight adaptation inspired by DIRE.

The goal is not to reproduce the official DIRE implementation.  
Instead, we keep the central idea of the method: using an image reconstruction error as a forensic signal.

## Principle

Given an input image `x`, a lightweight pretrained generative model produces a reconstructed image `x_hat`.

The detection signal is the reconstruction error map:

```math
M(x) = |x - \hat{x}|
```
## Pipeline
```text
input image x
    -> lightweight pretrained model
    -> reconstructed image x_hat
    -> error map |x - x_hat|
```