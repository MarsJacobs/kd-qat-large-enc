
                           
for seed in 42 # 52 62 
do                                             
    bash run_SARQ-1step.sh $1 $2 $3 3 1 1
    # bash run_SARQ-1step.sh $1 $2 large ${seed} ${temp} 0 1 0 
    # bash run_SARQ-1step.sh $1 $2 large ${seed} ${temp} 0 0 1
done
