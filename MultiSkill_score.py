import argparse
import os
import random
import time
import openai
import re
from openai import OpenAI
from tqdm import tqdm
from utils import read_json, read_jsonl, generate_answer, write_jsonl, check_dir, encode_image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



SYSTEM_PROMPT = """You are a helpful and precise assistant in labeling the score of the instruction."""

SCORE_ANNOTATION_PROMPT = """We would like to request your feedback on the performance of the response of the assistant to the user instruction displayed below. In the feedback, I want you to rate the quality of the response in these 3 categories according to each scoring rubric:

{skill_prompt}

[Instruction]
{question}

[Ground truth Answer]
{label}

[Assistant's Response]
{model_answer}

[The End of Assistant's Response]

Please provide feedback on the assistant's responses. Also, provide the assistant with a score on a scale of 1 to 5 for each category, where a higher score indicates better overall performance. Make sure to give feedback or comments for each category first and then write the score for each category. Only include feedback corresponding to the scoring rubric for each category. The scores for each category should be independent, meaning 'Logical Correctness' should not be considered when rating 'Readability,' for example.

Note that solving the instruction requires visual information from the image. To evaluate perception abilities (i.e., fine-grained perception, coarse-grained perception, and OCR), carefully analyze the assistant's response and determine what the assistant has seen based on its response. By comparing your perception of the image with the perception reflected in the assistant's response, rate its perception ability. Do NOT use "N/A" or "None" in your scoring results.

Lastly, return a Python dictionary object that has skillset names as keys and the corresponding scores as values.

[System]
"""


def build_skill_prompt(skill, skill_info):
    target_skill = []
    for item in skill_info:
        if item['Skill'] in skill:
            target_skill.append(item)

    skill_prompt = ""
    assert len(target_skill) <= 3
    for s in target_skill:
        skill_prompt += f"{s['Skill']}: {s['Criteria']}"
        for index in ['1', '2', '3', '4', '5']:
            skill_prompt += f"\nScore {index}: {s['Scoring'][index]}"
        if s != target_skill[-1]:
            skill_prompt += '\n\n'
    return skill_prompt


def construct_score_mess(question, model_answer, label, image, skill, skill_info):
    skill_prompt = build_skill_prompt(skill, skill_info)
    content = [{"type": "text", "text": SCORE_ANNOTATION_PROMPT.format(question=question, model_answer=model_answer, label=label, skill_prompt=skill_prompt)},]
    if isinstance(image, list):
        for img in image:
            base64_img = encode_image(img)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}})
    elif isinstance(image, str):
        base64_image = encode_image(image)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    else:
        raise Exception('Unknown image format!')
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages


def parse_score(content):
    try:
        match = re.findall(r'{[^}]+}', content)
        if len(match) > 0:
            dictionary_part = match[-1].replace("\n", "").replace('_', " ").lower()
            dictionary_part = re.sub(r'#.*?(\n|})', r'\1', dictionary_part)
            lines = eval(dictionary_part)
            for key, value in lines.items():
                if value == 'na':
                    lines[key] = 'N/A'
                elif value == 'n/a':
                    lines[key] = 'N/A'
                elif value == 'not applicable':
                    lines[key] = 'N/A'
            return lines
        else:
            return {}

    except Exception as e:
        raise Exception(f'{e}\nContent: {content}\n\nYou must manually fix the score pair.')


def annotation_score(args):
    logging.info(f'generate result model: {args.result_model}')
    logging.info(f'eval model: {args.eval_model}')

    input_file = os.path.join(f'./result/MultiSkill_inference_{args.result_model}.jsonl')
    logging.info(f'input_file: {input_file}')
    datas = read_jsonl(input_file)
    logging.info(f'data length: {len(datas)}')

    skill_file = os.path.join(args.data_path, f'skillset_description_multimodal.json')
    skill_info = read_json(skill_file)
    assert len(skill_info) == 12

    check_dir(args.result_path)
    output_file = os.path.join(args.result_path, f'MultiSkill_score_result{args.result_model}_eval{args.eval_model}.jsonl')
    logging.info(f'output file: {output_file}')
    if os.path.exists(output_file):
        result = read_jsonl(output_file)
        logging.info(f'reload results: {len(result)}')
    else:
        result = []

    for i, item in enumerate(tqdm(datas)):
        if i < len(result):
            continue
        question, model_answer, label, image, skill = item['question'], item['model_answer'], item['label'], item['image'], item['skill']

        messages = construct_score_mess(question, model_answer, label, image, skill, skill_info)
        _, content, in_token_num, out_token_num = generate_answer(args, messages)
        annotation_score = parse_score(content)

        result.append({
            **item,
            'score': annotation_score,
        })

        write_jsonl(output_file, result)


def calculate(args):
    check_dir(args.result_path)
    output_file = os.path.join(args.result_path, f'MultiSkill_score_result{args.result_model}_eval{args.eval_model}.jsonl')
    result = read_jsonl(output_file)
    logging.info(f'reload results: {len(result)}')
    skill_score_sum = {}
    skill_count = {}
    for i, item in enumerate(result):
        skill_score = item['score']
        for s, score in skill_score.items():
            if score == 'N/A':
                continue
            if s in skill_score_sum:
                skill_score_sum[s] += score
                skill_count[s] += 1
            else:
                skill_score_sum[s] = score
                skill_count[s] = 1

    for s in skill_score_sum:
        count = skill_count[s]
        print(f"{s}: {skill_score_sum[s] / count:.2f}")



def get_args():
    args_parser = argparse.ArgumentParser(description='MultiSkill')

    args_parser.add_argument('--data_path', type=str, default='./')
    args_parser.add_argument('--result_path', type=str, default='./result/')

    args_parser.add_argument('--result_model', type=str, default='gpt-4-turbo') # gpt-4-turbo qwen-vl-max gemini-1.5-pro
    args_parser.add_argument('--eval_model', type=str, default='gpt-4o') # gpt-4-turbo-2024-04-09
    args_parser.add_argument('--temperature', type=float, default=0)
    # args_parser.add_argument('--max_new_tokens', type=int, default=100)
    args_parser.add_argument('--api_key', type=str, default='*********YOUR API KEY*********')
    # openai.organization = ""

    args = args_parser.parse_args()
    args.model = args.eval_model
    args.client = OpenAI(api_key=args.api_key)

    return args


if __name__ == '__main__':
    args = get_args()

    annotation_score(args)

    calculate(args)