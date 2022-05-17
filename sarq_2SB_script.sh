#----------------------------# (map | cc | co) | (Attn Score | Attnmap | Context | Output)
# map : Teacher Map Intervention
# cc : Teacher Context intervention (w/ Context Loss)
# co : Teacher Context intervention (w/ Attention Output Loss)
# S : Attention Score | M : Attention Map | C : Attention Context | O : Attention Output

# STEP 1 PREPARATION
bash run_SARQ_step_1_ci_o.sh $1 $2 large
# bash run_SARQ_step_1_ci_o.sh $1 $2 large

# STEP 2 TRAINING                            
for seed in 42 52 62
do
# bash run_SARQ_step_2.sh $1 $2 base 0.4 ${seed}
bash run_SARQ_step_2.sh $1 $2 large ${seed}
done
