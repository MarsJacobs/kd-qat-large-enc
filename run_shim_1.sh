# Quantization Range
quantize=1
act_quant=1
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
lr_scaling=10
map=0

#===========================================================#
quantizer=ternary # ternary, pact, lsq
clipping=0

parks=0
khshim=1
khshim_FP=1

# Logging Option
exp_name=step_1_mse_kl
neptune=1
save_quantized_model=1

prob_log=0
log_metric=0
log_map=0

# Distill Option
gt_loss=0
pred_distill=1
rep_distill=1
attn_distill=1
attnmap_distill=1

value_relation=0
teacher_attnmap=1

# Training Type (downstream, qat_normal, qat_step1, qat_step2)
training_type=qat_step1

# Loss Coeff
attnmap_coeff=1
cls_coeff=1
att_coeff=1
rep_coeff=1

# DA Options
aug_train=0
aug_N=5
clip_teacher=0

learning_rate=2E-5
other_lr=2E-5
# ===========================================================#

CUDA_VISIBLE_DEVICES=$1 python quant_task_glue.py --data_dir data --task_name $2 --output_dir output --num_train_epochs 3 \
--weight_bits 2 --input_bits 8 --kd_layer_num ${kd_layer_num} \
--gpu 1 --quantize ${quantize} --act_quant ${act_quant} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
--layer_num ${layer_num} \
--aug_train ${aug_train} \
--gt_loss ${gt_loss} --pred_distill ${pred_distill} --rep_distill ${rep_distill} --attn_distill ${attn_distill} --attnmap_distill ${attnmap_distill} --value_relation ${value_relation} \
--training_type ${training_type} \
--clipping ${clipping} \
--mean_scale ${mean_scale} \
--quantizer ${quantizer} \
--init_scaling ${init_scaling} \
--gradient_scaling ${gradient_scaling} \
--lr_scaling ${lr_scaling} \
--clip_wd ${clip_wd} \
--clip_ratio ${clip_ratio} --clip_method ${clip_method} \
--exp_name ${exp_name} \
--save_quantized_model ${save_quantized_model} \
--log_map ${log_map} --log_metric ${log_metric} \
--neptune ${neptune} \
--aug_N ${aug_N} \
--prob_log ${prob_log} \
--clip_teacher ${clip_teacher} \
--num_train_epochs 3 \
--teacher_attnmap ${teacher_attnmap} \
--other_lr ${other_lr} \
--attnmap_coeff ${attnmap_coeff} --cls_coeff ${cls_coeff} --att_coef ${att_coeff} --rep_coeff ${rep_coeff} \
--seed 126 \
--learning_rate ${learning_rate} --parks ${parks} --khshim ${khshim} --khshim_FP ${khshim_FP}