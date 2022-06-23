
output_coeff=1

for map_coeff in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    for seed in 42 52 62
    do                             
        bash run_kd_sweep.sh $1 $2 $output_coeff $map_coeff $seed $3
    done
done
