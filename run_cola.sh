quantize=1
act_quant=1
q_qkv=1
q_ffn_1=1
q_ffn_2=1
q_emb=1
q_cls=1
clipping=0
aug_train=1
layer_num=-1
kd_layer_num=-1
mean_scale=0.7

for kd_layer_num in -1
do
    CUDA_VISIBLE_DEVICES=$1 python quant_task_glue.py --data_dir data --task_name cola --output_dir output --learning_rate 2e-5 --num_train_epochs 3 \
    --weight_bits 2 --input_bits 8 --kd_layer_num ${kd_layer_num} \
    --gpu 1 --quantize ${quantize} --act_quant ${act_quant} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
    --layer_num ${layer_num} \
    --aug_train ${aug_train} \
    --pred_distill --intermediate_distill \
    --clipping ${clipping} \
    --mean_scale ${mean_scale}
done
#--save_quantized_model \
#--pred_distill --intermediate_distill \
