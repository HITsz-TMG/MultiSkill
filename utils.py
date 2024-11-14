import json
import time
import base64
from dashscope import MultiModalConversation
import PIL.Image
import os
from io import BytesIO
import logging
from PIL import Image
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        logging.info(f'create directory: {dir}')
    else:
        logging.info(f'existed: {dir}')


def read_jsonl(file):
    with open(file, 'r', encoding='utf-8') as f:
        datas = [json.loads(item) for item in f]
    return datas


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    return datas


def write_jsonl(file, results):
    with open(file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + '\n')


def compact_image(img_path, model):
    max_size = {"qwen-vl-max": 15 * 1024 * 1024, "models/gemini-1.5-pro-latest": 3 * 1024 * 1024}
    img = Image.open(img_path)
    # 将图像保存到 BytesIO 对象以计算其大小
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_size = img_byte_arr.tell()

    while img_size > max_size[model]:
        # 缩小图像尺寸（保持纵横比）
        img.thumbnail((img.width * 0.9, img.height * 0.9), Image.Resampling.LANCZOS)
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_size = img_byte_arr.tell()
    if model == "models/gemini-1.5-pro-latest":
        model = "gemini-1.5-pro-latest"
    temp_file_path = f'./temp/compressed_{model}_{img_path[9:]}'
    img.save(temp_file_path, format='PNG')

    return temp_file_path


def generate_answer(args, messages):
    error_cnt = 0
    while True:
        try:
            if 'gpt' in args.model:
                response = args.client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                )
                num_prompt_tokens = response.usage.prompt_tokens
                num_completion_tokens = response.usage.completion_tokens
                content = response.choices[0].message.content

            elif 'gemini' in args.model:
                response = args.client.generate_content(
                    contents=messages,
                    stream=True,
                    generation_config={"temperature": args.temperature, "max_output_tokens": 1024},
                )
                response.resolve()
                num_prompt_tokens = args.client.count_tokens(messages[0]).total_tokens
                num_completion_tokens = args.client.count_tokens(response.text).total_tokens
                content = response.text

            elif 'claude' in args.model:
                response = args.client.messages.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=args.temperature,
                )
                num_prompt_tokens = response.usage.input_tokens
                num_completion_tokens = response.usage.output_tokens
                content = response.content[0].text

            elif 'qwen' in args.model:
                response = MultiModalConversation.call(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                )
                num_prompt_tokens = None
                num_completion_tokens = None
                content = response.output.choices[0].message.content[0]["text"]

            else:
                raise Exception('Unknown model!')

            break
        except Exception as e:
            error_cnt += 1
            if error_cnt == 5:
                raise Exception(f'retry 5 times. {e}')
            logging.warning(f"[{error_cnt}] retrying due to an error: {e}")
            time.sleep(5)

    return response, content, num_prompt_tokens, num_completion_tokens


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def media_type(image):
    if 'jpg' in image.lower() or 'jpeg' in image.lower():
        return 'image/jpeg'
    if 'png' in image.lower():
        return 'image/png'
    if 'gif' in image.lower():
        return 'image/gif'
    if 'webp' in image.lower():
        return 'image/webp'
    return 'image/jpeg'


def need_compact_qwen(img_path):
    img = Image.open(img_path)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_size = img_byte_arr.tell()
    if img_size > 15 * 1024 * 1024:
        return True
    return False


def need_compact_gemini(img_path):
    img = Image.open(img_path)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_size = img_byte_arr.tell()
    if img_size > 3 * 1024 * 1024:
        return True
    return False


def construct_mess(model, question, image):
    if 'gpt' in model:
        content = [{"type": "text", "text": question},]
        if isinstance(image, list):
            for img in image:
                base64_img = encode_image(img)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
        elif isinstance(image, str):
            base64_image = encode_image(image)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        else:
            raise Exception('Unknown image format!')
        messages = [{"role": "user", "content": content}, ]

    elif 'gemini' in model:
        if isinstance(image, list):
            messages = [question, ]
            for img in image:
                if ('VisionGraph' in img or 'KVQA' in img) and need_compact_gemini(img):
                    messages.append(compact_image(img, model))
                elif 'gif' in img or 'GIF' in img:
                    messages.append(PIL.Image.open(img).convert('RGB'))
                else:
                    messages.append(PIL.Image.open(img))
        elif isinstance(image, str):
            if ('VisionGraph' in image or 'KVQA' in image) and need_compact_gemini(image):
                messages = [question, compact_image(image, model)]
            elif 'gif' in image or 'GIF' in image:
                messages = [question, PIL.Image.open(image).convert('RGB')]
            else:
                messages = [question, PIL.Image.open(image)]
        else:
            raise Exception('Unknown image format!')

    elif 'claude' in model:
        content = []
        if isinstance(image, list):
            for img in image:
                base64_img = encode_image(img)
                content.append({"type": "image", "source": {"type": "base64", "media_type": media_type(img), "data": base64_img}})
        elif isinstance(image, str):
            base64_image = encode_image(image)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type(image), "data": base64_image}})
        else:
            raise Exception('Unknown image format!')
        content.append({"type": "text", "text": question},)
        messages = [{"role": "user", "content": content}, ]

    elif 'qwen' in model:
        content = []
        if isinstance(image, list):
            for img in image:
                if need_compact_qwen(img):
                    content.append({"image": compact_image(img, model)})
                else:
                    content.append({"image": img})
        elif isinstance(image, str):
            if need_compact_qwen(image):
                content.append({"image": compact_image(image, model)})
            else:
                content.append({"image": image})
        else:
            raise Exception('Unknown image format!')
        content.append({"text": question})
        messages = [{"role": "user", "content": content}, ]

    else:
        raise Exception('Unknown model!')
    return messages





if __name__ == '__main__':
    pass