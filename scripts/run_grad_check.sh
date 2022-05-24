bash run_Ternary-1step.sh $1 $2 base
bash run_SARQ-1step.sh $1 $2 base 42

bash run_SARQ_step_1_ci_o.sh $1 $2 base
bash run_SARQ_step_2.sh $1 $2 base 42

bash run_Ternary-1step.sh $1 $2 large
bash run_SARQ-1step.sh $1 $2 large 42

bash run_SARQ_step_1_ci_o.sh $1 $2 large
bash run_SARQ_step_2.sh $1 $2 large 42
