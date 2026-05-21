from t2i_imagenet import T2IPipeline

MODEL_NAME = "Lucasdegeorge/CAD-I"

pipe = T2IPipeline(MODEL_NAME)

components = {
    "network": pipe.network,
    "scheduler": pipe.scheduler,
    "sampler": pipe.sampler,
    "postprocessing": pipe.postprocessing,
    "preconditioning": pipe.preconditioning,
}

for name, obj in components.items():
    print("\n" + "=" * 80)
    print(f"{name.upper()}")
    print("=" * 80)

    print("Type:")
    print(type(obj))

    print("\nAttributs utiles possibles:")
    keywords = [
        "vae", "encoder", "decoder", "decode", "encode",
        "latent", "noise", "sample", "step",
        "forward", "model", "unet", "scheduler",
        "alpha", "sigma", "timestep", "denoise"
    ]

    for attr in dir(obj):
        low = attr.lower()
        if any(k in low for k in keywords):
            print("-", attr)

    print("\nMéthodes publiques:")
    for attr in dir(obj):
        if not attr.startswith("_"):
            value = getattr(obj, attr)
            if callable(value):
                print("-", attr)
                