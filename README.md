# MultiSkill: Evaluating Large Multimodal Models for Fine-grained Alignment Skills




## :sparkles: Overview

This repository contains the code and data for "MultiSkill: Evaluating Large Multimodal Models for Fine-grained Alignment Skills" (EMNLP 2024 Findings).

## :rotating_light: Usage


### Data
Download data from [here](https://huggingface.co/datasets/HIT-TMG/MultiSkill) and `unzip` images.zip.
```
MultiSkill.jsonl
skillset_description_multimodal.json
images
result
temp
MultiSkill_inference.py
MultiSkill_score.py
utils.py
```

### Inference
1) Set your api_key and choose the base model.
2) Run `MultiSkill_inference.py`.

### Score
1) Set your api_key and choose the base model, evaluation file.
2) Run `MultiSkill_score.py`.




