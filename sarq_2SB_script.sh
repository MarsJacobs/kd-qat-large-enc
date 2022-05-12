#----------------------------# (map | cc | co) | (Attn Score | Attnmap | Context | Output)
# map : Teacher Map Intervention
# cc : Teacher Context intervention (w/ Context Loss)
# co : Teacher Context intervention (w/ Attention Output Loss)
# S : Attention Score | M : Attention Map | C : Attention Context | O : Attention Output

# STEP 1 PREPARATION
# bash run_SARQ_step_1_ci_o.sh $1 $2

# STEP 2 TRAINING
seed=42
                            # M O LI
# bash run_SARQ_step_2.sh $1 $2 1 0 1
for seed in 42 52 62
do
bash run_SARQ_step_2.sh $1 $2 1 0 0 ${seed}
# bash run_SARQ_step_2.sh $1 $2 0 1 0 ${seed}
done
# bash run_SARQ_step_2.sh $1 $2 1 1 0

# for seed in 42 52 62 72 82
# do
#     bash run_SARQ-1step.sh $1 $2 0 1 0 0 large ${seed} # Map
#     bash run_SARQ-1step.sh $1 $2 0 0 0 1 large ${seed} # Output
# done