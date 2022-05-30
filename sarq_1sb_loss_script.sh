
                           
for seed in 72 82 92 12 22 32
do                             # S M C O              
    bash run_SARQ-1step.sh $1 $2 1 0 0 0 $seed
    bash run_SARQ-1step.sh $1 $2 0 1 0 0 $seed
    bash run_SARQ-1step.sh $1 $2 0 0 1 0 $seed
    bash run_SARQ-1step.sh $1 $2 0 0 0 1 $seed
    bash run_SARQ-1step.sh $1 $2 0 1 0 1 $seed
    bash run_SARQ-1step.sh $1 $2 0 1 1 0 $seed
done
