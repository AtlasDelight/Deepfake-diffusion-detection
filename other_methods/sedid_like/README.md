# SeDID-like

This folder contains a lightweight adaptation inspired by SeDID.

The goal is not to reproduce the official SeDID implementation.  
Instead, we keep the central idea of SeDID: comparing a deterministic reverse path and a deterministic denoising path at the same intermediate timestep.

## Principle

## SeDID-like image-space signal

A noisier latent state is first built at a source noise level:

```math
z_{\gamma_s}
=
\sqrt{\gamma_s}z_0
+
\sqrt{1-\gamma_s}\varepsilon
```

Two paths are then constructed.

### 1. Reverse / diffusion path

The reverse path moves the clean latent state \(z_0\) to the target noise level \(\gamma_t\):

```math
z_{\gamma_t}^{rev}
=
\sqrt{\gamma_t}z_0
+
\sqrt{1-\gamma_t}\varepsilon
```

### 2. Denoising path

The model predicts a noise-like response from \(z_{\gamma_s}\):

```math
\hat{\varepsilon}_{\theta}
=
f_{\theta}(z_{\gamma_s},\gamma_s)
```

A clean latent is then estimated:

```math
\hat{z}_0
=
\frac{
z_{\gamma_s}
-
\sqrt{1-\gamma_s}\hat{\varepsilon}_{\theta}
}{
\sqrt{\gamma_s}
}
```

This estimated clean latent is projected to the same target noise level:

```math
z_{\gamma_t}^{denoise}
=
\sqrt{\gamma_t}\hat{z}_0
+
\sqrt{1-\gamma_t}\hat{\varepsilon}_{\theta}
```

Both target states are then decoded back to image space:

```math
x_{\gamma_t}^{rev}
=
D(z_{\gamma_t}^{rev})
```

```math
x_{\gamma_t}^{denoise}
=
D(z_{\gamma_t}^{denoise})
```

The final SeDID-like signal is computed in image space:

```math
M(x)
=
\left|
x_{\gamma_t}^{rev}
-
x_{\gamma_t}^{denoise}
\right|
```

### Important note

The final error map is computed in the original image space.  
Internally, the model still uses latent variables because T2I-ImageNet operates in a latent representation, but the comparison used for detection is performed after decoding both paths back to image space.