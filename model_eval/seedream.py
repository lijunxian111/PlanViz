import os
# Install SDK:  pip install 'volcengine-python-sdk[ark]'
from volcenginesdkarkruntime import Ark 
import base64
from PIL import Image
import json
from io import BytesIO
import mimetypes

api_key = "5d1705be-9a72-4ade-a9a3-72410d720727"

def save_base64_image_from_json(json_input, image_key='image_base64'):
    # 解析 JSON 字符串（如果传入的是 str）
    #print(type(json_input))
    """
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided.")
    elif isinstance(json_input, dict):
        data = json_input
    else:
        raise TypeError("json_input must be a str or dict")
    """
    data = json_input
    b64_string = data
    # 获取 base64 内容
    """
    if image_key not in data:
        raise ValueError(f"Key '{image_key}' not found in JSON.")
    
    b64_string = data[image_key]
    """

    # 如包含头部（data:image/png;base64,...），移除它
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    # 解码并保存图像
    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except base64.binascii.Error:
        raise ValueError("Failed to decode base64 string.")

def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"

client = Ark(
    # The base URL for model invocation
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # Get API Key：https://console.volcengine.com/ark/region:ark+cn-beijing/apikey
    api_key=api_key, 
)
 
def eval_seedream(image, prompt, edit=False):
    if image is not None:
        b64_image = encode_file(image)

    if edit == True:
        imagesResponse = client.images.generate( 
        # Replace with Model ID
            model="doubao-seedream-4-5-251128",
            image=b64_image,
            prompt=prompt,
            size="2K",
            response_format="b64_json",
            watermark=False
        )
    else:
        imagesResponse = client.images.generate( 
            # Replace with Model ID
            model="doubao-seedream-4-5-251128",
            prompt=prompt,
            size="2K",
            response_format="b64_json",
            watermark=False
        ) 
 
    #print(imagesResponse.data[0].b64_json)
    output_image = save_base64_image_from_json(imagesResponse.data[0].b64_json)
    return output_image

if __name__ == "__main__":
    img = eval_seedream(None, "Generate a cute cat for me.")
    img.save('seedream.png')
