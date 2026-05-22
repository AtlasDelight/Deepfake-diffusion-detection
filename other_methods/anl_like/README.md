# ANL-like

This folder contains a lightweight adaptation inspired by Attention-guided Noise Learning.

The attention used here is not handcrafted.  
It is learned during the training of the detector.

## Principle

Given an input image \(x\), we extract two signals:

```math
E(x)=|x-\hat{x}|
```

A noise-related feature is extracted from the pretrained model:

```math
\hat{\varepsilon}_{\theta}
=
f_{\theta}(z_{\gamma}, \gamma)
```

A learnable attention module \(g_{\eta}\) then predicts an attention map:

```math
A_{\eta}(x)
=
g_{\eta}(\hat{\varepsilon}_{\theta})
```

The image reconstruction error is defined as:

```math
E(x)
=
|x-\hat{x}|
```

This reconstruction error is then guided by the learned attention:

```math
M(x)
=
A_{\eta}(x)
\odot
E(x)
```

The final classifier is trained end-to-end using this attention-guided error map.

## Important note

Unlike a heuristic attention map based directly on the magnitude of the predicted noise, this implementation uses a trainable attention module.

The attention map is learned during training and should only be interpreted after the detector has been trained.