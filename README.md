# Self-Attention Recovery for QAT(SARQ) Implementation
This Repository contains SARQ code for **Self-Attention Map is All You Need for QAT of Finetuned Transformers**

<img width="867" alt="스크린샷 2022-02-07 오후 2 03 19" src="https://user-images.githubusercontent.com/54992207/152727350-4553fc95-efc8-43dc-beff-9ad74898e421.png">

## Environments
```
pip install -r requirements.txt
```

## Model
You can get GLUE task specific fine-tuned BERT base model using huggingface code. 
https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification


## GLUE Dataset 
### Data
Download GLUE. 
https://github.com/nyu-mll/GLUE-baselines

## Self-Attention Recovery for QAT (SARQ)
Proposed SARQ method consists of Two Steps. (See Figure for Two Step SARQ)

1. Teacher Intervention is employed to finetune quantized weights of attention propagation (PROP)
2. Quantization is applied to the entire weights of Transformer layers for QAT

You can easily try SARQ two step Training using bash scripts.
```
# For SARQ Two Step Training (w/o DA)
bash run_SARQ_two_step.sh {GPU Num} {GLUE Task} # bash run_SARQ_two_step.sh 0 sts-b

# For SARQ 1 Step Training (w/o DA)
bash run_SARQ_1step.sh {GPU Num} {GLUE Task} {DA option} {DA N param} # bash run_SARQ-1step.sh 0 sts-b 0 0

# For TernaryBERT Training for comparison
bash run_glue.sh {GPU Num} {GLUE Task} # bash run_glue.sh 0 sts-b

```

For Data Augmentation (DA) Option, use TinyBERT Data Augmentation for getting expanded GLUE Dataset.

https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT


## Arguments
To be Updated.


