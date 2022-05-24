for epoch in 3
do                                             
    bash run_SARQ-1step.sh $1 $2 $3 $epoch 1 0
    # bash run_SARQ-1step.sh $1 $2 large ${seed} ${temp} 0 1 0 
    # bash run_SARQ-1step.sh $1 $2 large ${seed} ${temp} 0 0 1
done
