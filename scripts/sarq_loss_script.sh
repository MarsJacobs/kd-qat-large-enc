#----------------------------# (map | cc | co) | (Attn Score | Attnmap | Context | Output)
# map : Teacher Map Intervention
# cc : Teacher Context intervention (w/ Context Loss)
# co : Teacher Context intervention (w/ Attention Output Loss)
# S : Attention Score | M : Attention Map | C : Attention Context | O : Attention Output

# STEP 1 PREPARATION
bash run_SARQ_step_1_ci_c.sh $1 $2
bash run_SARQ_step_1_ci_o.sh $1 $2

# STEP 2 TRAINING
                               # S M C O
bash run_SARQ_step_2.sh $1 $2 cc 0 0 1 0
bash run_SARQ_step_2.sh $1 $2 cc 0 0 0 1
bash run_SARQ_step_2.sh $1 $2 cc 0 1 1 0
bash run_SARQ_step_2.sh $1 $2 cc 0 1 0 1
                               # S M C O
bash run_SARQ_step_2.sh $1 $2 co 0 0 1 0
bash run_SARQ_step_2.sh $1 $2 co 0 0 0 1
bash run_SARQ_step_2.sh $1 $2 co 0 1 1 0
bash run_SARQ_step_2.sh $1 $2 co 0 1 0 1

bash run_SARQ_step_1.sh $1 $2
                                # S M C O
bash run_SARQ_step_2.sh $1 $2 map 1 1 0 0
bash run_SARQ_step_2.sh $1 $2 map 1 1 1 0
bash run_SARQ_step_2.sh $1 $2 map 1 1 0 1


