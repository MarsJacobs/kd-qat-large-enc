python quant_task_glue.py --data_dir data --task_name $1 --output_dir output --learning_rate 2e-5 --num_train_epochs 3 \
--weight_bits 2 --input_bits 8 --pred_distill --intermediate_distill --save_fp_model --save_quantized_model \
--aug_train --gpu 1