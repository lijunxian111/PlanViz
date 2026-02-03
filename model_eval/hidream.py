import torch
import argparse
import sys 
import gc
from HiDream_I1.hi_diffusers import HiDreamImagePipeline
from HiDream_I1.hi_diffusers import HiDreamImageTransformer2DModel
from HiDream_I1.hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from HiDream_I1.hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate import infer_auto_device_map, dispatch_model



#device = accelerator.device

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="full")
args = parser.parse_args()
model_type = args.model_type
MODEL_PREFIX = "/path/to/model_ckpts"
LLAMA_MODEL_NAME = "/path/to/LLama-3.1-8B-Instruct"

# Model configurations
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# Resolution options
RESOLUTION_OPTIONS = [
    "1024 × 1024 (Square)",
    "768 × 1360 (Portrait)",
    "1360 × 768 (Landscape)",
    "880 × 1168 (Portrait)",
    "1168 × 880 (Landscape)",
    "1248 × 832 (Landscape)",
    "832 × 1248 (Portrait)"
]

# Load models
def load_models(model_type):
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    scheduler = MODEL_CONFIGS[model_type]["scheduler"](num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)
    
    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        LLAMA_MODEL_NAME,
        use_fast=False)
    
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        #output_hidden_states=True,
        #output_attentions=True,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16)
    text_encoder_4.eval()

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer", 
        device_map='auto',
        torch_dtype=torch.bfloat16)
    transformer.eval()

    print(text_encoder_4.device)
    print(transformer.device)

    pipe = HiDreamImagePipeline.from_pretrained(
        pretrained_model_name_or_path, 
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16).to('cuda', torch.bfloat16)
    #).to('cuda', torch.bfloat16)
    pipe.transformer = transformer
    #pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    
    return pipe, config

# Parse resolution string to get height and width
def parse_resolution(resolution_str):
    if "1024 × 1024" in resolution_str:
        return 1024, 1024
    elif "768 × 1360" in resolution_str:
        return 768, 1360
    elif "1360 × 768" in resolution_str:
        return 1360, 768
    elif "880 × 1168" in resolution_str:
        return 880, 1168
    elif "1168 × 880" in resolution_str:
        return 1168, 880
    elif "1248 × 832" in resolution_str:
        return 1248, 832
    elif "832 × 1248" in resolution_str:
        return 832, 1248
    else:
        return 1024, 1024  # Default fallback

# Generate image function
def generate_image(pipe, model_type, prompt, resolution, seed):
    # Get configuration for current model
    config = MODEL_CONFIGS[model_type]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    
    # Parse resolution
    height, width = parse_resolution(resolution)

    # height, width = 512, 512
    
    # Handle seed
    if seed == -1:
        seed = 42
    
    generator = torch.Generator("cuda").manual_seed(seed)
    with torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator
        ).images
    
    return images[0], seed

def eval_hidream(image, prompt, edit=False):
    if edit == True:
        pass 
    else:
        print("Loading default model (full)...")
        pipe, _ = load_models(model_type)
        print("Model loaded successfully!")
        resolution = "1024 × 1024 (Square)"
        seed = -1
        image, seed = generate_image(pipe, model_type, prompt, resolution, seed)
        return image

# Initialize with default model
print("Loading default model (full)...")
pipe, _ = load_models(model_type)
print("Model loaded successfully!")
prompt = "A cat holding a sign that says \"Hi-Dreams.ai\"." 
resolution = "1024 × 1024 (Square)"
seed = -1
image, seed = generate_image(pipe, model_type, prompt, resolution, seed)
image.save("output.png")
