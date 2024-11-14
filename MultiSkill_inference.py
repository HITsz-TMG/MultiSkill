import argparse
import os
import anthropic
from openai import OpenAI
from tqdm import tqdm
import google.generativeai as genai
from utils import read_jsonl, construct_mess, generate_answer, write_jsonl, check_dir
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import openai








def inference(args):
    logging.info(f'model: {args.model}')

    input_file = os.path.join(args.data_path, 'MultiSkill.jsonl')
    datas = read_jsonl(input_file)
    logging.info(f'data length: {len(datas)}')

    check_dir(args.result_path)
    output_file = os.path.join(args.result_path, f'MultiSkill_inference_{args.model_name}.jsonl')
    if os.path.exists(output_file):
        result = read_jsonl(output_file)
        logging.info(f'reload results: {len(result)}')
    else:
        result = []

    for i, item in enumerate(tqdm(datas)):
        if i < len(result):
            continue

        id, question, label, image, task = item["id"], item['question'], item['label'], item['image'], item['task']

        messages = construct_mess(args.model, question, image)
        _, content, in_token_num, out_token_num = generate_answer(args, messages)

        result.append({
            'id': id,
            'question': question,
            'image': image,
            'model_answer': content,
            'label': label,
            'skill': item['skill'],
            'domain': item['domain'],
            'difficulty': item['difficulty'],
            'task': task,
            'model': args.model,
        })

        write_jsonl(output_file, result)


def get_args():
    args_parser = argparse.ArgumentParser(description='MultiSkill')

    args_parser.add_argument('--data_path', type=str, default='./')
    args_parser.add_argument('--result_path', type=str, default='./result/')

    args_parser.add_argument('--model', type=str, default='gpt-4-turbo-2024-04-09',
                             choices=['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13', 'models/gemini-1.5-pro-latest', 'claude-3-opus-20240229', 'qwen-vl-max'],)
    args_parser.add_argument('--temperature', type=float, default=0)
    # args_parser.add_argument('--max_new_tokens', type=int, default=100)
    args_parser.add_argument('--api_key', type=str, default='*********YOUR API KEY*********')
    # openai.organization = ""
    # args_parser.add_argument('--base_url', type=str, default='')

    args = args_parser.parse_args()
    if 'gpt' in args.model:
        args.client = OpenAI(api_key=args.api_key,) #  base_url=args.base_url
    elif 'gemini' in args.model:
        genai.configure(api_key=args.api_key, transport='rest')
        args.client = genai.GenerativeModel(args.model)
    elif 'claude' in args.model:
        args.client = anthropic.Anthropic(api_key=args.api_key)
    elif 'qwen' in args.model:
        os.environ["DASHSCOPE_API_KEY"] = args.api_key
    else:
        raise Exception('Unknown model!')
    args.model_name = {'gpt-4o-2024-05-13': 'gpt-4o', 'models/gemini-1.5-pro-latest': 'gemini-1.5-pro',
                       'claude-3-opus-20240229': 'claude-3-opus', 'qwen-vl-max': 'qwen-vl-max',
                       'gpt-4-turbo-2024-04-09': 'gpt-4-turbo'}[args.model]
    return args


if __name__ == '__main__':
    args = get_args()

    inference(args)
