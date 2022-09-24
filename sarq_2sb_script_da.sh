# distill (4-6) TI (7-9) map-con-output | MI-CI-OI
# bash run_SARQ_step_1.sh $1 $2 $3 1 0 0 1 0 0 # MI
# bash run_SARQ_step_1.sh $1 $2 $3 0 1 0 0 1 0 # CI
# bash run_SARQ_step_1.sh $1 $2 $3 0 0 1 0 0 1 # OI

# bash run_SARQ_step_2.sh $1 $2 $seed $size 1 0 0 MI # MI step2
# bash run_SARQ_step_2.sh $1 $2 $seed $size 0 1 0 CI # CI step2
# bash run_SARQ_step_2.sh $1 $2 $seed $size 0 0 1 OI # OI step2
# bash run_SARQ_step_1.sh $1 $task $size 1 0 0



for size in tiny-4l tiny-6l
do
    for task in rte sts-b mrpc
    do
        for epoch in 1 12
        do                                                  # Epoch-DA-N
            for seed in 1 2 3
            do
                bash run_SARQ_step_2.sh $1 $task $seed $size MIXED $epoch 1 30 # OI step2 - epoch
            done
        done
    done
done