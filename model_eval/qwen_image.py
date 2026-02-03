import base64
import mimetypes
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import dashscope
from dashscope import MultiModalConversation
import requests
from dashscope import ImageSynthesis
import os
import json
from io import BytesIO

from PIL import Image


dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


api_key = "sk-xxx"
model = 'qwen-image-edit-2509'  

def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("not supported")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"



def eval_qwen_image(image, prompt, model_name):
    if image is not None:
        image = encode_file(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image},
                    {"text": prompt}
                ]
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt}
                ]
            }
        ]
    response = MultiModalConversation.call(
        api_key=api_key,
        model=model_name,
        messages=messages,
        stream=False,
        n=1,
        watermark=False,
        negative_prompt=" ",
        prompt_extend=True,

    )
    if response.status_code == 200:
        success_contents = []
        print(json.dumps(response, ensure_ascii=False))
        for i, content in enumerate(response.output.choices[0].message.content):
            print(f"URL:{content['image']}")
            success_contents.append(content['image'])
        return success_contents
    else:
        print(f"HTTP Code：{response.status_code}")
        print(f"Error Code：{response.code}")
        print(f"Error Information：{response.message}")
        print("Doc：https://help.aliyun.com/zh/model-studio/error-code")
        return []

def eval_wan(image, prompt, model_name):
    if image is not None:
        image_url_1 = "file://" + image     # Linux/macOS
    
        print(image_url_1)

    print('----sync call, please wait a moment----')
    if 'i2i' in model_name:
        rsp = ImageSynthesis.call(api_key=api_key,
                          model=model_name, #wan2.5-i2i-preview or wan2.5-t2i-preview
                          prompt=prompt,
                          images=[image_url_1],
                          negative_prompt="",
                          n=1,
                          # size="1280*1280",
                          prompt_extend=True,
                          watermark=False,
                          seed=12345)
    else:
        rsp = ImageSynthesis.call(api_key=api_key,
                          model=model_name, #wan2.5-i2i-preview or wan2.5-t2i-preview
                          prompt=prompt,
                          #images=[image_url_1],
                          negative_prompt="",
                          n=1,
                          # size="1280*1280",
                          prompt_extend=True,
                          watermark=False,
                          seed=12345)
    print('response: %s' % rsp)
    if rsp.status_code == HTTPStatus.OK:
        for result in rsp.output.results:
            file_name = f'/path/to/uni_bench/results/{model_name}/tmp.png'
            #file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            with open(file_name, 'wb+') as f:
                f.write(requests.get(result.url).content)
        img = Image.open(f'/path/to/uni_bench/results/{model_name}/tmp.png').convert('RGB')
        return img
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
            (rsp.status_code, rsp.code, rsp.message))
