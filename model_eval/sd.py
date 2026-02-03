import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image
import gc

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

pipe = StableDiffusion3Pipeline.from_pretrained("/path/to/model_ckpts/sd-3.5-large", torch_dtype=torch.bfloat16, trust_remote_code=True)
pipe = pipe.to("cuda")

def count_params(module):
    return sum(p.numel() for p in module.parameters())

#print(pipe)
#total = count_params(pipe)

#print(f"Total parameters: {total:,}")

def eval_sd35(org_image, prompt, edit=False):
    global pipe
    if edit == True:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            "/path/to/model_ckpts/sd-3.5-large",
            torch_dtype=torch.float16
        ).to("cuda")
        #init_image = Image.open(org_image).convert("RGB")
        org_image = load_image(org_image)
        image = pipe(
            prompt,
            image=org_image,
            num_inference_steps=28,
            strength=0.7,
            guidance_scale=9.0,
        ).images[0]
    else:
        image = pipe(
            prompt,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]
    return image

if __name__ == "__main__":
    img = eval_sd35('/path/to/uni_bench/test.png', 'Add a cat.', True)
    img.save('sd2.png')
