quantize=1
act_quant=0
q_qkv=0
q_ffn_1=1
q_ffn_2=1
q_emb=0
q_cls=0
aug_train=0
layer_num=-1
kd_layer_num=-1
mean_scale=0.7

for kd_layer_num in -1
do
    CUDA_VISIBLE_DEVICES=$1 python eval_code.py --data_dir data --task_name cola --output_dir output --learning_rate 2e-5 --num_train_epochs 3 \
    --weight_bits 2 --input_bits 8 --kd_layer_num ${kd_layer_num} \
    --gpu 1 --quantize ${quantize} --act_quant ${act_quant} --qkv ${q_qkv} --ffn_1 ${q_ffn_1} --ffn_2 ${q_ffn_2} --emb ${q_emb} --cls ${q_cls} \
    --layer_num ${layer_num} \
    --aug_train ${aug_train} \
    --pred_distill --intermediate_distill \
    --mean_scale ${mean_scale} \
    --seed 50
done