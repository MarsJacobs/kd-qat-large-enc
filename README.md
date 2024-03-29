# [Understanding and Improving KD for QAT of Large Transformer Encoders (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.450)

<img width="2087" alt="스크린샷 2023-01-22 오후 7 37 29" src="https://user-images.githubusercontent.com/54992207/213911541-95e99d90-832e-4582-bb6f-a6e330a1048c.png">

Code for "Understanding and Improving Knowledge Distailltion for Quantization-Aware-Training of Large Transformer Encoders"   
[Proceeding](https://aclanthology.org/2022.emnlp-main.450), [Arxiv](https://arxiv.org/abs/2211.11014)

- This paper provides in-depth analysis of the mechanism of Knowledge Distillation(KD) on Attention recovery of quantized large Transformer encoders.
- Based on this analysis, we propose new sets of KD loss functions for better QAT of ultra-low bit precision (Weight Ternarization of Transformer Encoders).

Our implementation is based on the Huawei-Noah TernaryBERT Pytorch code. ([link](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TernaryBERT))



## Setup


```
pip install -r requirements.txt
```
First, you need task-specific fine-tuned full-precision BERT models for initialize model for QAT.
You can fine-tune BERT-base/large pre-trained model using huggingface example code with following link. 

- https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification)

Or you could download fine-tuned BERT-base/large model with Google Cloud link. (sst-2, rte provided)
- https://drive.google.com/drive/folders/1OjCeLP2tmirZAhP2_ZX8pN3N2ytQBbn2?usp=share_link

Fine-tuned BERT model file should be plaece in "models" folder with its GLUE task name. (ex. models/rte)

## Training (QAT)
This repository provides multiple KD options for Ternary QAT of BERT-base/large.

#### Attenion map/output loss
For attention map/output loss QAT,
```
bash run_kd_qat_map.sh $GPU_NUM # map loss
bash run_kd_qat_output.sh $GPU_NUM # output loss
```

#### KD options exploration
For exploration of KD options for QAT with BERT-base/large over GLUE tasks, use run_kd_qat_exploration.sh.
For example, let's run attention-map loss QAT of BERT-base with CoLA task.
```
task_name=cola
bert=base
map_coeff=1
output_coeff=0
bash run_kd_qat_exploration.sh $task_name $bert $map_coeff $output_coeff 
```
#### Find mixing parameters
For explorating mixing parameters for attention-map/output losses, run run_mixing_param_sweep.sh

## Experiments (Analysis)

### Model Directory
For experimental notebooks, you need QAT model file from Training section.
Note that fine-tuned full-precision model files should be placed in models folder, and QAT model files should be placed in output folder.
For example,

```
teacher_model_dir = "models/BERT_base/sst-2"
student_model_dir = "output/BERT_large/rte/exploration/$EXP_NAME"
```
Please set full-precision/QAT model directory name properly in notebook :)

### Exp1. Attention Map Distance (Figure 4)
Exp 1 shows how to measure self-attention map distance between teacher model (full-precision model) and student model (ternary quantized model)
This notebook provides attention map distance plot as follows.

![스크린샷 2022-12-26 오후 3 27 49](https://user-images.githubusercontent.com/54992207/209511787-6d7e7867-eb17-4e43-888e-7abc15761d39.png)

### Exp2. Hessian Eigen Max Spectra Analysis (Figure 3)
Exp2 provides hessian max eigenvalue spectra of QAT model.
This implementation is based on the Pyhessian and repository of "Park et al, How do Vision Transformer Work?, ICLR 2022"

Pyhessian : https://github.com/amirgholami/PyHessian
How-vits-work : https://github.com/xxxnell/how-do-vits-work/issues/12

### Exp3. Attention Output Analysis (Figure 5-6)
This experiements provide analysis of attention output's min-max dynamic range and attention norm.
Once you load model file properly, you can find the model's attention output dynamic range per layer and conduct norm based analysis per layer/head.

Per Layer Attnetion output min-max dynamic range (Left), Norm based analysis per layer (Right)


![스크린샷 2022-12-26 오후 3 51 28](https://user-images.githubusercontent.com/54992207/209514174-1c82db7d-549f-4288-9a0a-e33c7bef9af6.png)

Per Head attnetion probability, transformed output heat map visualization and difference between student and teacher model visualizataion with heat map. (with Attention-map/output loss QAT)


![스크린샷 2022-12-26 오후 3 54 50](https://user-images.githubusercontent.com/54992207/209514547-abdc9297-f8fb-45c4-b932-a4ea6d8868d2.png)

Attentoin Norm based analysis is based on "Kobayashi et al Attention is Not Only a Weight: Analyzing Transformers with Vector Norms, EMNLP 2020" code [link](https://github.com/gorokoba560/norm-analysis-of-transformer/tree/master/emnlp2020)

For further question, contact me anytime (minsoo2333@hanyang.ac.kr) or kindly leave questions in issues tab.
