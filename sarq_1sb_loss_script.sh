#----------------------------# (map | cc | co) | (Attn Score | Attnmap | Context | Output)
# map : Teacher Map Intervention
# cc : Teacher Context intervention (w/ Context Loss)
# co : Teacher Context intervention (w/ Attention Output Loss)
# S : Attention Score | M : Attention Map | C : Attention Context | O : Attention Output


# TRAINING (Pred, Rep Fix)
#----------------------BERT BASE----------------------#
                           # S M C O
# bash run_SARQ-1step.sh $1 $2 1 0 0 0 base # Ternary
# bash run_SARQ-1step.sh $1 $2 1 1 0 0 base # SARQ

# bash run_SARQ-1step.sh $1 $2 0 1 0 0 base # Map

# # # Context | Output Effect Check
# bash run_SARQ-1step.sh $1 $2 0 0 1 0 base # Context
# bash run_SARQ-1step.sh $1 $2 0 0 0 1 base # Output

# # # AttnMap + Context/Output
# bash run_SARQ-1step.sh $1 $2 0 1 1 0 base # Context + Map
# bash run_SARQ-1step.sh $1 $2 0 1 0 1 base # Output + Map

#----------------------BERT LARGE----------------------#
                           
for seed in 42 52 62 
do                                             
    bash run_SARQ-1step.sh $1 $2 large ${seed} 
    # bash run_SARQ-1step.sh $1 $2 large ${seed} ${temp} 0 1 0 
    # bash run_SARQ-1step.sh $1 $2 large ${seed} ${temp} 0 0 1
done
