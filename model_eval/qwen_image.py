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

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
api_key = "sk-badb475655b044c082f4748111f03427"
model = 'qwen-image-edit-2509'  # 可选值：wan-2.5-i2i-preview, qwen-image-edit-plus, qwen-image-edit
# --- 输入图片：使用 Base64 编码 ---
# base64编码格式为 data:{MIME_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

"""
图像输入方式说明：
以下提供了三种图片输入方式，三选一即可

1. 使用公网URL - 适合已有公开可访问的图片
2. 使用本地文件 - 适合本地开发测试
3. 使用Base64编码 - 适合私有图片或需要加密传输的场景
"""

# 【方式一】使用公网图片 URL
#image_url_1 = "https://img.alicdn.com/imgextra/i3/O1CN0157XGE51l6iL9441yX_!!6000000004770-49-tps-1104-1472.webp"
#image_url_2 = "https://img.alicdn.com/imgextra/i3/O1CN01SfG4J41UYn9WNt4X1_!!6000000002530-49-tps-1696-960.webp"

# 【方式二】使用本地文件（支持绝对路径和相对路径）
# 格式要求：file:// + 文件路径
# 示例（绝对路径）：
"""
if model == 'wan-2.5-i2i-preview':
    image_url_1 = "file://" + "/data2/user/junxianli/uni_bench/map/editing/org_images/paris.png"     # Linux/macOS

    print('----sync call, please wait a moment----')
    rsp = ImageSynthesis.call(api_key=api_key,
                          model="qwen-image-edit-plus", #wan2,5-i2i-preview or qwen-image-edit 
                          prompt="Plan a tourist guide for me. I want to visit more than three famous landmarks. Draw the route on the map.",
                          images=[image_url_1],
                          negative_prompt="",
                          n=1,
                          # size="1280*1280",
                          prompt_extend=True,
                          watermark=False,
                          seed=12345)
    print('response: %s' % rsp)
    if rsp.status_code == HTTPStatus.OK:
    # 在当前目录下保存图片
        for result in rsp.output.results:
            file_name = '/data2/user/junxianli/uni_bench/results/wan-2.5-preview/try.png'
            #file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            with open('./%s' % file_name, 'wb+') as f:
                f.write(requests.get(result.url).content)
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
            (rsp.status_code, rsp.code, rsp.message))


elif 'qwen-image-edit' in model:
    image = encode_file("/data2/user/junxianli/uni_bench/map/editing/org_images/paris.png")

    messages = [
        {
            "role": "user",
            "content": [
                #{"image": image},
                {"text": "Can you design a food-focused travel route for exploring Tokyo's best local dishes and markets and draw it?"}
            ]
        }
    ]

    response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-plus",
    messages=messages,
    stream=False,
    n=1,
    watermark=False,
    negative_prompt=" ",
    prompt_extend=True,
    # 仅当输出图像数量n=1时支持设置size参数，否则会报错
    # size="2048*1024",
)
    if response.status_code == 200:
        # 如需查看完整响应，请取消下行注释
        print(json.dumps(response, ensure_ascii=False))
        for i, content in enumerate(response.output.choices[0].message.content):
            print(f"输出图像{i+1}的URL:{content['image']}")
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")
"""


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
        # 仅当输出图像数量n=1时支持设置size参数，否则会报错
        # size="2048*1024",
    )
    if response.status_code == 200:
        # 如需查看完整响应，请取消下行注释
        success_contents = []
        print(json.dumps(response, ensure_ascii=False))
        for i, content in enumerate(response.output.choices[0].message.content):
            print(f"输出图像{i+1}的URL:{content['image']}")
            success_contents.append(content['image'])
        return success_contents
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/error-code")
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
    # 在当前目录下保存图片
        for result in rsp.output.results:
            file_name = f'/data2/user/junxianli/uni_bench/results/{model_name}/tmp.png'
            #file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
            with open(file_name, 'wb+') as f:
                f.write(requests.get(result.url).content)
        img = Image.open(f'/data2/user/junxianli/uni_bench/results/{model_name}/tmp.png').convert('RGB')
        return img
    else:
        print('sync_call Failed, status_code: %s, code: %s, message: %s' %
            (rsp.status_code, rsp.code, rsp.message))