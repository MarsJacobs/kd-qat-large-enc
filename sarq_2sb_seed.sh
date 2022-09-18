
                           
for seed in 1 2 3 4 5 6 7 8 9 10
do                             # M O SA SEED    
    bash run_SARQ_step_2.sh $1 $2 $seed base
done

for seed in 1 2 3 4 5 6 7 8 9 10
do                             # M O SA SEED    
    bash run_SARQ_step_2.sh $1 $2 $seed large
done