task_name=$2
bert_size=$3
model_name=$4
kd_loss=1
kd_loss_type=$5

CUDA_VISIBLE_DEVICES=$1 python run_lls.py --task_name $task_name --bert_size $bert_size \
                                          --model_name $model_name \
                                          --kd_loss $kd_loss \
                                          --kd_loss_type $kd_loss_type