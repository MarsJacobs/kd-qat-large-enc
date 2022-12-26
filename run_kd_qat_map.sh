# Quantization Option
act_quant=1
weight_quant=1
weight_bits=2 # 8, 4, 2
input_bits=8

# KD options
gt_loss=0
pred_distill=1
rep_distill=1
attn_distill=0
attnmap_distill=1
context_distill=0
output_distill=0
sa_output_distill=0

# BERT option (base, large)
bert=$3

# DA options
aug_train=0
aug_N=30

# Training Options
learning_rate=2E-5
num_train_epochs=3
task_name=$2
exp_name=${task_name}_${bert}_map
neptune=0
save_quantized_model=1
seed=42

# Unification Attention Loss (Map + Output Loss)
map_coeff=1
output_coeff=1
# ============================================= #

CUDA_VISIBLE_DEVICES=$1 python main.py --data_dir data --task_name $task_name --output_dir output --num_train_epochs $num_train_epochs --bert ${bert} \
--weight_bits ${weight_bits} --input_bits ${input_bits} --gpu 1 --act_quant ${act_quant} --weight_quant ${weight_quant} --aug_train ${aug_train} --aug_N ${aug_N} \
--sa_output_distill $sa_output_distill --output_distill ${output_distill} --context_distill ${context_distill} --gt_loss ${gt_loss} --pred_distill ${pred_distill} --rep_distill ${rep_distill} --attn_distill ${attn_distill} --attnmap_distill ${attnmap_distill} \
--map_coeff $map_coeff --output_coeff $output_coeff \
--exp_name ${exp_name} --save_quantized_model ${save_quantized_model} \
--neptune ${neptune} --seed $seed --learning_rate ${learning_rate} 