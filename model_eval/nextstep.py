import torch
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.append('/data2/user/junxianli/')
from PIL import Image

# NextStep_1_Large_Edit
# NextStep_1_Large
from model_ckpts.NextStep_1_Large_Edit.models.gen_pipeline import NextStepPipeline
from model_ckpts.NextStep_1_Large_Edit.utils.aspect_ratio import center_crop_arr_with_buckets

HF_HUB = "/data2/user/junxianli/model_ckpts/NextStep_1_Large"
# /data2/user/junxianli/model_ckpts/NextStep_1_Large_Edit
# /data2/user/junxianli/model_ckpts/NextStep_1_Large

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
pipeline = NextStepPipeline(tokenizer=tokenizer, model=model).to(device="cuda", dtype=torch.bfloat16)

# set prompts

def gen_image(prompt):
    positive_prompt = "masterpiece, film grained, best quality."
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
    example_prompt = prompt

    # generate image from text
    IMG_SIZE = 512
    image = pipeline.generate_image(
        example_prompt,
        hw=(IMG_SIZE, IMG_SIZE),
        num_images_per_caption=1,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        cfg=7.5,
        cfg_img=1.0,
        cfg_schedule="constant",
        use_norm=False,
        num_sampling_steps=28,
        timesteps_shift=1.0,
        seed=3407,
    )[0]
    return image

def edit_image(image, prompt):
    positive_prompt = None
    negative_prompt = "Copy original image."
    example_prompt = "<image>" + prompt

    IMG_SIZE = 512
    ref_image = Image.open(image).convert('RGB')
    ref_image = center_crop_arr_with_buckets(ref_image, buckets=[IMG_SIZE])

    # generate edited image
    image = pipeline.generate_image(
        example_prompt,
        images=[ref_image],
        hw=(IMG_SIZE, IMG_SIZE),
        num_images_per_caption=1,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        cfg=7.5,
        cfg_img=2,
        cfg_schedule="constant",
        use_norm=True,
        num_sampling_steps=50,
        timesteps_shift=3.2,
        seed=42,
    )[0]

    return image

def eval_nextstep(image, prompt, edit=False):
    if edit == True:
        img = edit_image(image, prompt)
        return img
    else:
        img = gen_image(image)
        return img

if __name__ == "__main__":
    img = gen_image('a lovely cat')
    img.save('nextstep.png')
