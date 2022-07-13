file_name=$4
bert=base
# SARQ step1 : sarq_step1_S_M 
# Ternary step1 : 1SB_1epoch_S 
# Ternary : 1SB_S 1SB : 1SB_S_M 2SB : step_2_S_M
model_name=$3
quant_model_name=1SB_O
init=0

CUDA_VISIBLE_DEVICES=$1 python max_eigenvalue.py --task $2 \
                                                --bert $bert \
                                                --file_name $file_name \
                                                --model_name $model_name \
                                                --quant_model_name $quant_model_name \
                                                --init $init