python quant_task_glue.py --data_dir /home/ms/workspace/git/Pretrained-Language-Model/TernaryBERT/data --task_name sst-2 --output_dir output --learning_rate 2e-5 --num_train_epochs 3 \
--weight_bits 2 --input_bits 8 --pred_distill --intermediate_distill --save_fp_model --save_quantized_model \
--aug_train --gpu 1
