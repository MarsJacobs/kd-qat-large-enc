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
index_ratio=0.005

#===========================================================#
bert=large
quantizer=ternary # ternary, pact, lsq
act_quantizer=ternary # pact, lsq, clipping
weight_bits=2 # 8, 2
input_bits=8 # 8, 2
clipping=0

act_method=clipping # rounding, clipping

parks=0
khshim=0
khshim_FP=0

# Logging Option
exp_name=1SB_output
neptune=1
save_quantized_model=1

prob_log=0
log_metric=0
log_map=0

# Distill Option
gt_loss=0
pred_distill=1
rep_distill=1
attn_distill=0
attnmap_distill=0
word_distill=0
val_distill=0
context_distill=1

value_relation=0
teacher_attnmap=0

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

# DA Options
aug_train=0
aug_N=0

# LR
learning_rate=2E-5
other_lr=2E-5 # for step 2

# ===========================================================#

CUDA_VISIBLE_DEVICES=$1 python quant_task_glue.py --data_dir data --task_name $2 --output_dir output --num_train_epochs 3 --bert ${bert} \
--weight_bits ${weight_bits} --input_bits ${input_bits} --kd_layer_num ${kd_layer_num} --gpus 1 \
--quantize ${quantize} --act_quant ${act_quant} --weight_quant ${weight_quant} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
--layer_num ${layer_num} \
--aug_train ${aug_train} \
--context_distill ${context_distill} --val_distill ${val_distill} --word_distill ${word_distill} --gt_loss ${gt_loss} --pred_distill ${pred_distill} --rep_distill ${rep_distill} --attn_distill ${attn_distill} --attnmap_distill ${attnmap_distill} \
--context_coeff ${context_coeff} --val_coeff ${val_coeff} --attnmap_coeff ${attnmap_coeff} --cls_coeff ${cls_coeff} --att_coef ${att_coeff} --rep_coeff ${rep_coeff} --word_coeff ${word_coeff} \
--training_type ${training_type} \
--clipping ${clipping} --act_method ${act_method} \
--mean_scale ${mean_scale} \
--quantizer ${quantizer} --act_quantizer ${act_quantizer} \
--init_scaling ${init_scaling} \
--gradient_scaling ${gradient_scaling} \
--lr_scaling ${lr_scaling} --index_ratio ${index_ratio} \
--clip_wd ${clip_wd} \
--clip_ratio ${clip_ratio} --clip_method ${clip_method} \
--exp_name ${exp_name} \
--save_quantized_model ${save_quantized_model} \
--log_map ${log_map} --log_metric ${log_metric} \
--neptune ${neptune} \
--aug_N ${aug_N} \
--prob_log ${prob_log} \
--num_train_epochs 3 \
--teacher_attnmap ${teacher_attnmap} \
--other_lr ${other_lr} \
--seed 42 \
--learning_rate ${learning_rate} --parks ${parks}
