from openai import OpenAI
import base64
import os
from PIL import Image
from io import BytesIO


client = OpenAI(api_key='sk-proj-xxxx')

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
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image_bytes))

    with open("cat.png", "wb") as f:
        f.write(image_bytes)
