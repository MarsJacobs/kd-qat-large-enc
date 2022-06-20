
                           
for seed in 72 82 12 22 32
do                             # M O SEED    
    bash run_SARQ-1step.sh $1 $2 0 0 $seed $3
    bash run_SARQ-1step.sh $1 $2 1 0 $seed $3
    bash run_SARQ-1step.sh $1 $2 0 1 $seed $3
done
