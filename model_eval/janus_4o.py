import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor

# Load model and processor
class VLChatProcessorOutput:
    def __init__(self, sft_format: str, input_ids: torch.Tensor, pixel_values: torch.Tensor, num_image_tokens: torch.IntTensor):
        self.sft_format = sft_format
        self.input_ids = input_ids
        self.pixel_values = pixel_values
        self.num_image_tokens = num_image_tokens

    def __len__(self):
        return len(self.input_ids)

import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model_path = "/path/to/model_ckpts/Janus-4o"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True,torch_dtype=torch.bfloat16
)
vl_gpt = vl_gpt.cuda().eval()

def process_image(image_paths,vl_chat_processor):
    images = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
    images_outputs = vl_chat_processor.image_processor(images, return_tensors="pt")
    return images_outputs['pixel_values']

# Define text-to-image generation function
def text_to_image_generate(input_prompt, output_path, vl_chat_processor, vl_gpt, temperature = 1.0, parallel_size = 2, cfg_weight = 5):

    torch.cuda.empty_cache()

    conversation = [
            {
             "role": "<|User|>",
            "content": input_prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    prompt = sft_format + vl_chat_processor.image_start_tag

    mmgpt = vl_gpt

    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    with torch.inference_mode():
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec
        
        return PIL.Image.fromarray(visual_img[0])
        """
        os.makedirs(output_path, exist_ok=True)
        output_images = []
        for i in range(parallel_size):
            save_path = output_path.replace('.png','') + f'_{i}.png'
            PIL.Image.fromarray(visual_img[i]).save(save_path)
            output_images.append(save_path)
        return output_images
        """

def text_and_image_to_image_generate(input_prompt, input_image_path, output_path, vl_chat_processor, vl_gpt, temperature = 1.0, parallel_size = 2, cfg_weight = 5, cfg_weight2 = 5):
    torch.cuda.empty_cache()

    input_img_tokens = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag + vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    output_img_tokens = vl_chat_processor.image_start_tag 

    pre_data = []
    input_images = [input_image_path]
    img_len = len(input_images)
    prompts = input_img_tokens * img_len + input_prompt
    conversation = [
                    {"role": "<|User|>","content": prompts},
                    {"role": "<|Assistant|>", "content": ""}
                ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    sft_format = sft_format + output_img_tokens

    mmgpt = vl_gpt

    image_token_num_per_image = 576
    img_size = 384
    patch_size = 16

    with torch.inference_mode():
        input_image_pixel_values = process_image(input_images,vl_chat_processor).to(torch.bfloat16).cuda()
        quant_input, emb_loss_input, info_input = mmgpt.gen_vision_model.encode(input_image_pixel_values)
        image_tokens_input = info_input[2].detach().reshape(input_image_pixel_values.shape[0], -1)
        image_embeds_input = mmgpt.prepare_gen_img_embeds(image_tokens_input)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        
        encoder_pixel_values = process_image(input_images,vl_chat_processor).cuda()
        tokens = torch.zeros((parallel_size*3, len(input_ids)), dtype=torch.long)
        for i in range(parallel_size*3):
            tokens[i, :] = input_ids
            if i % 3 == 2:
                tokens[i, 1:-1] = vl_chat_processor.pad_id
                pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=encoder_pixel_values, input_ids=tokens[i-2], num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
                pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=encoder_pixel_values, input_ids=tokens[i-1], num_image_tokens=[vl_chat_processor.num_image_tokens] * img_len))
                pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=None, input_ids=tokens[i], num_image_tokens=[]))

        prepare_inputs = vl_chat_processor.batchify(pre_data)

        inputs_embeds = mmgpt.prepare_inputs_embeds(
                    input_ids=tokens.cuda(),
                    pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).cuda(),
                    images_emb_mask=prepare_inputs['images_emb_mask'].cuda(),
                    images_seq_mask=prepare_inputs['images_seq_mask'].cuda()
                )

        image_gen_indices = (tokens == vl_chat_processor.image_end_id).nonzero()

        for ii, ind in enumerate(image_gen_indices):
            if ii % 4 == 0:
                offset = ind[1] + 2
                inputs_embeds[ind[0],offset: offset+image_embeds_input.shape[1],:] = image_embeds_input[(ii // 2) % img_len]

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state

            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond_full = logits[0::3, :]
            logit_cond_part = logits[1::3, :]
            logit_uncond = logits[2::3, :]

            logit_cond = (logit_cond_full + cfg_weight2 * (logit_cond_part)) / (1 + cfg_weight2)
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return PIL.Image.fromarray(visual_img[0])
        

# Run
def eval_janus4o(image, prompt, edit=False):
    #prompt = "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair"
    image_output_path = "./test.png"
    if edit == True:
        output_image = text_and_image_to_image_generate(prompt, image, image_output_path, vl_chat_processor, vl_gpt, parallel_size = 1)
    else:
        output_image = text_to_image_generate(prompt, image_output_path, vl_chat_processor, vl_gpt, parallel_size = 1)
    return output_image

if __name__ == "__main__":
    prompt = "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair"
    image_output_path = "./test.png"
    output_image = text_to_image_generate(prompt, image_output_path, vl_chat_processor, vl_gpt, parallel_size = 1)

