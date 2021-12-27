# Quantization Range
quantize=1
act_quant=1
q_qkv=1
q_ffn_1=1
q_ffn_2=1
q_emb=1
q_cls=1
layer_num=-1

# Training Option
clipping=0

# KD & Ternary Option
kd_layer_num=-1
mean_scale=0.7
gradient_scaling=1

# ===========================================================#
quantizer=pact # ternary, pact, lsq
init_scaling=1 # only for LSQ Clip val init

clip_wd=0.5

clip_method=std
clip_ratio=1 # for PACT
lr_scaling=10
neptune=1
aug_train=0
# ===========================================================#


CUDA_VISIBLE_DEVICES=$1 python quant_task_glue.py --data_dir data --task_name cola --output_dir output --learning_rate 2e-5 --num_train_epochs 3 \
--weight_bits 2 --input_bits 8 --kd_layer_num ${kd_layer_num} \
--gpu 1 --quantize ${quantize} --act_quant ${act_quant} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
--layer_num ${layer_num} \
--aug_train ${aug_train} \
--pred_distill --intermediate_distill --gt_loss \
--clipping ${clipping} \
--mean_scale ${mean_scale} \
--quantizer ${quantizer} \
--init_scaling ${init_scaling} \
--gradient_scaling ${gradient_scaling} \
--lr_scaling ${lr_scaling} \
--clip_wd ${clip_wd} \
--clip_ratio ${clip_ratio} --clip_method ${clip_method} \
--exp_name 1215 \
--neptune ${neptune} 


# \
#--pred_distill --intermediate_distill \
#--neptune