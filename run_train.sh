quantize=1
act_quant=0
q_qkv=0
q_ffn=1
q_emb=0
q_cls=0
layer_num=-1

CUDA_VISIBLE_DEVICES=$1 python quant_task_glue.py --data_dir data --task_name $2 --output_dir output --learning_rate 2e-5 --num_train_epochs 3 \
--weight_bits 2 --input_bits 8 --pred_distill --intermediate_distill --save_fp_model --save_quantized_model --aug_train \
--gpu 1 \
--quantize ${quantize} --act_quant ${act_quant} \
--qkv ${q_qkv} \
--ffn ${q_ffn} \
--emb ${q_emb} \
--cls ${q_cls} \
--layer_num ${layer_num} \
#--neptune




