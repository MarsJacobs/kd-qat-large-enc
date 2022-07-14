task=$2
model_name=$3
bert_size=$4
data_percentage=0.05
tol=0.01

for seed in 1 2 3 4 5 6 7 8 9 10
do
    CUDA_VISIBLE_DEVICES=$1 python run_lw_hessian.py --task_name $task \
                                                     --model_name $model_name \
                                                     --data_percentage $data_percentage \
                                                     --bert_size $bert_size \
                                                     --seed $seed \
                                                     --tol $tol 
done                        

