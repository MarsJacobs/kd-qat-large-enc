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

# ===========================================================#
quantizer=ternary # ternary, pact, lsq
clipping=0

neptune=1
save_quantized_model=0
aug_train=0
prob_log=0
attn_test=0

# Distill Option
rep_distill=1
attn_distill=1
attnmap_distill=0

# DA Options
aug_N=30
clip_teacher=0
# ===========================================================#

CUDA_VISIBLE_DEVICES=$1 python quant_task_glue.py --data_dir data --task_name $2 --output_dir output --learning_rate 2e-5 --num_train_epochs 3 \
--weight_bits 2 --input_bits 8 --kd_layer_num ${kd_layer_num} \
--gpu 1 --quantize ${quantize} --act_quant ${act_quant} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
--layer_num ${layer_num} \
--aug_train ${aug_train} \
--pred_distill --intermediate_distill \
--rep_distill ${rep_distill} --attn_distill ${attn_distill} --attnmap_distill ${attnmap_distill} \
--clipping ${clipping} \
--mean_scale ${mean_scale} \
--quantizer ${quantizer} \
--init_scaling ${init_scaling} \
--gradient_scaling ${gradient_scaling} \
--lr_scaling ${lr_scaling} \
--clip_wd ${clip_wd} \
--clip_ratio ${clip_ratio} --clip_method ${clip_method} \
--exp_name all_da_0103 \
--save_quantized_model ${save_quantized_model} \
--neptune ${neptune} \
--aug_N ${aug_N} \
--prob_log ${prob_log} \
--clip_teacher ${clip_teacher} \
--attn_test ${attn_test} \
--num_train_epochs 3

#--pred_distill --intermediate_distill \
#--neptune--save_quantized_model \