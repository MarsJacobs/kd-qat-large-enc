
# distill (4-6) TI (7-9) map-con-output | MI-CI-OI
# bash run_SARQ_step_1.sh $1 $2 $3 1 0 0 1 0 0 # MI
# bash run_SARQ_step_1.sh $1 $2 $3 0 1 0 0 1 0 # CI
# bash run_SARQ_step_1.sh $1 $2 $3 0 0 1 0 0 1 # OI

# bash run_SARQ_step_2.sh $1 $2 $seed $size 1 0 0 MI # MI step2
# bash run_SARQ_step_2.sh $1 $2 $seed $size 0 1 0 CI # CI step2
# bash run_SARQ_step_2.sh $1 $2 $seed $size 0 0 1 OI # OI step2


for size in base large
do
# bash run_SARQ_step_1.sh $1 $2 $size 0 0 1 0 0 1 # OI
    for seed in 1 2 3 4 5 6 7 8 9 10
    do                             # M O SA SEED    
        bash run_SARQ_step_2.sh $1 $2 $seed $size 0 0 1 OI 3 # OI step2 - epoch
    done
done

for size in tiny-4l tiny-6l
do
bash run_SARQ_step_1.sh $1 $2 $size 0 0 1 0 0 1 # OI
    for seed in 1 2 3 4 5 6 7 8 9 10
    do                             # M O SA SEED    
        bash run_SARQ_step_2.sh $1 $2 $seed $size 0 0 1 OI 10 # OI step2
    done
done
