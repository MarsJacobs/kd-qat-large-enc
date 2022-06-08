# Quantization Range
quantize=1
act_quant=1
weight_quant=1
q_qkv=1
q_ffn_1=1
q_ffn_2=1
q_emb=1
q_cls=1
layer_num=-1

# KD & Ternary Option
kd_layer_num=-1
mean_scale=0.7

# LSQ Options
gradient_scaling=1 
init_scaling=1 

# PACT Options
clip_method=std 
clip_ratio=1 
clip_wd=0.5
lr_scaling=1
index_ratio=0.01
map=0

#===========================================================#
bert=large
version_2_with_negative=0
loss_SM=0
sm_temp=0
quantizer=ternary # ternary, pact, lsq
act_quantizer=ternary
weight_bits=2 # 8, 2
input_bits=8 # 8, 2
clipping=0

parks=0
stop_grad=0
qk_FP=0

# Logging Option
exp_name=1SB_no_rep
step1_option=LSM
neptune=0
save_quantized_model=1


prob_log=0
log_metric=0
log_map=0

# Distill Option
gt_loss=0
pred_distill=1
rep_distill=0

attn_distill=$3
attnmap_distill=$4
context_distill=$5
output_distill=$6

word_distill=0
val_distill=0

# Teacher Intervention (TI)
teacher_attnmap=0
teacher_input=0
layer_thres_num=15

# Training Type (downstream, qat_normal, qat_step1, qat_step2)
training_type=qat_normal

# Loss Coeff
attnmap_coeff=1
word_coeff=1
cls_coeff=1
att_coeff=1
rep_coeff=1
val_coeff=1
context_coeff=1
output_coeff=1

# DA Options
aug_train=0
aug_N=0

learning_rate=2E-5
other_lr=2E-5
# ===========================================================#

CUDA_VISIBLE_DEVICES=$1 python quant_task_squad_non.py --data_dir data --task_name $2 --output_dir output --num_train_epochs 3 --bert ${bert} \
--weight_bits ${weight_bits} --input_bits ${input_bits} --kd_layer_num ${kd_layer_num} \
--gpu 1 --quantize ${quantize} --act_quant ${act_quant} --weight_quant ${weight_quant} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
--layer_num ${layer_num} \
--aug_train ${aug_train} \
--output_distill ${output_distill} --context_distill ${context_distill} --val_distill ${val_distill} --word_distill ${word_distill} --gt_loss ${gt_loss} --pred_distill ${pred_distill} --rep_distill ${rep_distill} --attn_distill ${attn_distill} --attnmap_distill ${attnmap_distill} \
--output_coeff ${output_coeff} --context_coeff ${context_coeff} --val_coeff ${val_coeff} --attnmap_coeff ${attnmap_coeff} --cls_coeff ${cls_coeff} --att_coef ${att_coeff} --rep_coeff ${rep_coeff} --word_coeff ${word_coeff} \
--training_type ${training_type} \
--clipping ${clipping} \
--mean_scale ${mean_scale} \
--quantizer ${quantizer} --act_quantizer ${act_quantizer} \
--init_scaling ${init_scaling} \
--gradient_scaling ${gradient_scaling} \
--lr_scaling ${lr_scaling} --index_ratio ${index_ratio} \
--clip_wd ${clip_wd} \
--clip_ratio ${clip_ratio} --clip_method ${clip_method} \
--exp_name ${exp_name} \
--version_2_with_negative $version_2_with_negative \
--save_quantized_model ${save_quantized_model} \
--log_map ${log_map} --log_metric ${log_metric} \
--neptune ${neptune} \
--aug_N ${aug_N} \
--prob_log ${prob_log} \
--teacher_input ${teacher_input} --teacher_attnmap ${teacher_attnmap} --layer_thres_num ${layer_thres_num} \
--other_lr ${other_lr} \
--seed $7 --sm_temp ${sm_temp} --loss_SM ${loss_SM} \
--step1_option ${step1_option} \
--learning_rate ${learning_rate} --parks ${parks} --stop_grad ${stop_grad} --qk_FP ${qk_FP}
