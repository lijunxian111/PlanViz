import json
import torch
from transformers import modeling_utils
import argparse
import os

import base64

from template import *

from openai import OpenAI
import base64
import re, json
import pandas as pd

api_key = "sk-xxxx"
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def eval_qa(messages):
    #image_base64 = image_to_base64(image)
    completion = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-instruct",  # model names
        messages=messages
        )
    print(completion.model_dump_json())
    return completion


def eval_one(prompt, img_paths):

    query = prompt
    try:
        content_lst = []
        for path in img_paths:
            if os.path.exists(path):
                pass 
            else:
                path = path.replace('png', 'jpg')
            b64 = image_to_base64(path)
            mime_type = "image/jpeg" 
            img_str = f"data:{mime_type};base64,{b64}"
            content_lst.append({'type': 'image_url', 'image_url': {
                        "url": img_str
                    }})
        content_lst.append({"type": "text", "text": query})
        messages = [
            {
                "role": "user",
                "content": content_lst
            }
        ]

        # Preparation for inference
        completion = eval_qa(messages)
        resp_text = completion.choices[0].message.content
        return resp_text
    except:
        return "No response"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choices for MLLM-as-a-judge")

    parser.add_argument('--task', type=str, help="task you want to evaluate", default="gen", choices=['gen', 'edit'])
    parser.add_argument('--path', type=str, help="result_json_path", default="")
    parser.add_argument('--mode', type=str, help="evaluation_dimension", default="correctness")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")

    args = parser.parse_args()
    
    data = open(args.path, 'r')
    data = json.load(data)
    score_lst = []
    for i, line in enumerate(data):
        if 'res_path' not in line:
            continue
        if args.mode == 'correctness':
            if 'correctness_score_points' in line:
                score_points = line['correctness_score_points']
            else:
                score_points = line['correctness_key_points']
            score_points_str = ''
            for j in range(len(score_points)):
                score_points_str += f'{j+1}. {score_points[j]} '
            if args.task == 'gen':
                prompt = TEMPLATE_CORRECTNESS_GENERATION.format(
                        line['prompt'],
                        score_points_str
                )   
                output_text = eval_one(prompt, [line['res_path']])
                line['evaluation_out'] = output_text
                score_lst.append(line)
            else:
                prompt = TEMPLATE_CORRECTNESS_EDITING.format(
                    line['prompt'],
                    score_points_str
                )
                if "res_path" not in line:
                    output_text = "0"
                else:
                    output_text = eval_one(prompt, [line['image_path'], line['res_path'], line['ref_image_path']])
                line['evaluation_out'] = output_text
                score_lst.append(line)
        elif args.mode == 'visual':
            if args.task == 'gen':
                prompt = TEMPLATE_VISUAL_GENERATION
                output_text = eval_one(prompt, [line['res_path']])
            else:
                prompt = TEMPLATE_VISUAL_EDIT
                output_text = eval_one(prompt, [line['image_path'], line['res_path']])
            line['evaluation_out'] = output_text
            score_lst.append(line)
        elif args.mode == 'efficiency':
            if args.task == 'gen':
                prompt = TEMPLATE_EFFICIENCY_GEN.format(line['prompt'])
                output_text = eval_one(prompt, [line['res_path']])
            else:
                prompt = TEMPLATE_EFFICIENCY_EDIT.format(line['prompt'])
                output_text = eval_one(prompt, [line['image_path'], line['res_path']])
            line['evaluation_out'] = output_text
            score_lst.append(line)
        print(f"Eval Finished for [{i}]")

    store_path = args.path.split('/')[-1].replace('.json',f'_{args.mode}_after_score.json')
    with open('output_jsons/'+store_path, 'w') as writer:
        json.dump(score_lst, writer, ensure_ascii=False, indent=4)
