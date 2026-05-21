import inspect
from t2i_imagenet import T2IPipeline

MODEL_NAME = "Lucasdegeorge/CAD-I"

pipe = T2IPipeline(MODEL_NAME)

print("\n================ VAE ================")
vae = pipe.postprocessing.vae
print("Type VAE:", type(vae))

for name in ["encode", "decode", "forward"]:
    if hasattr(vae, name):
        fn = getattr(vae, name)
        print(f"\n{name} signature:")
        print(inspect.signature(fn))

print("\n================ POSTPROCESSING ================")
print("Type:", type(pipe.postprocessing))
print("forward signature:")
print(inspect.signature(pipe.postprocessing.forward))

print("\n================ NETWORK ================")
print("Type:", type(pipe.network))
print("forward signature:")
print(inspect.signature(pipe.network.forward))

print("\n================ PRECONDITIONING ================")
print("Type:", type(pipe.preconditioning))
print("forward signature:")
print(inspect.signature(pipe.preconditioning.forward))

print("\n================ SAMPLER ================")
print("Type:", type(pipe.sampler))
print("sampler signature:")
print(inspect.signature(pipe.sampler))