from t2i_imagenet import T2IPipeline

MODEL_NAME = "Lucasdegeorge/CAD-I"

print("Chargement du pipeline...")
pipe = T2IPipeline(MODEL_NAME)

print("\nType du pipeline :")
print(type(pipe))

print("\nAttributs importants possibles :")
keywords = [
    "vae", "autoencoder", "encoder", "decoder",
    "unet", "model", "network",
    "scheduler", "sampler", "diffusion",
    "tokenizer", "text_encoder",
    "latent", "noise"
]

for name in dir(pipe):
    lname = name.lower()
    if any(k in lname for k in keywords):
        print("-", name)

print("\nDictionnaire interne du pipeline :")
for key in pipe.__dict__.keys():
    print("-", key)