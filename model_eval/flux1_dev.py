import torch
from diffusers import FluxPipeline
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import gc

#global pipe
import torch.nn.functional as F
original_scaled_dot_product_attention = F.scaled_dot_product_attention

def patched_scaled_dot_product_attention(*args, **kwargs):
    kwargs.pop('enable_gqa', None)
    return original_scaled_dot_product_attention(*args, **kwargs)

F.scaled_dot_product_attention = patched_scaled_dot_product_attention

pipe = FluxPipeline.from_pretrained("/path/to/Flux/black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

def eval_flux1_dev(image, prompt, edit=False):
    global pipe
    if edit:
        #del pipe
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        pipe = FluxKontextPipeline.from_pretrained("/path/to/Flux1_Kontext_dev/", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        init_image = load_image(image)
        image = pipe(
            image=init_image,
            prompt=prompt,
            guidance_scale=2.5,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]
        return image

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]
    return image

if __name__ == "__main__":
    pass
