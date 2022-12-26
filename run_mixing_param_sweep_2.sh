task=$2
bert=$3

# Mixing Param 2 
# Map-Loss * alpha + Output-Loss

output_coeff=1
for map_coeff in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for seed in 42 52 62
    do
    CUDA_VISIBLE_DEVICES=$1 bash run_kd_qat_exploration.sh $task $bert $map_coeff $output_coeff $seed
    done
done
