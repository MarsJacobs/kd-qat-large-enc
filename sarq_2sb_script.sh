task=$2
for size in tiny-6l
do
# bash run_SARQ_step_1.sh $1 $task $size 1 0 0
    for seed in 1 2 3 4 5 6 7 8 9 10
    do          
        bash run_SARQ_step_2.sh $1 $task $seed $size 0 0 0 MIXED 3 # OI step2 - epoch
    done
done

# bash run_SARQ_step_1.sh $1 $task $size 0 1 0
#     for seed in 1 2 3 4 5 6 7 8 9 10
#     do          
#         bash run_SARQ_step_2.sh $1 $task $seed $size 1 0 1 STOCHASTIC 3 # OI step2 - epoch
#     done

# bash run_SARQ_step_1.sh $1 $task $size 1 0 0
#     for seed in 1 2 3 4 5 6 7 8 9 10
#     do          
#         bash run_SARQ_step_2.sh $1 $task $seed $size 1 0 1 MIXED 3 # OI step2 - epoch
#     done