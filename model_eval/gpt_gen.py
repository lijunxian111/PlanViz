from openai import OpenAI
import base64
import os
from PIL import Image
from io import BytesIO

#os.environ['OPENAI_API_KEY'] = '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'

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7893"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7893"

client = OpenAI(api_key='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')

def eval_gpt_gen(image, prompt, edit=False):
    
    if edit == True:
        result = client.images.edit(
            model="gpt-image-1",
            image=open(image, "rb"),
            prompt=prompt
        )
    else:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024"
        )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image_bytes))
    
    return img

if __name__ == "__main__":
    prompt = "a cute cat"

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )

# 返回的是 base64 编码，需要解码保存
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image_bytes))

    with open("cat.png", "wb") as f:
        f.write(image_bytes)

    print("图片已生成并保存为 cat.png")