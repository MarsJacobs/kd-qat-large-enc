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
                           # S M C O
# bash run_SARQ-1step.sh $1 $2 1 0 0 0 large # Ternary
# bash run_SARQ-1step.sh $1 $2 1 1 0 0 large # SARQ

# bash run_SARQ-1step.sh $1 $2 0 1 0 0 large # Map

# Context | Output Effect Check
# bash run_SARQ-1step.sh $1 $2 0 0 1 0 large # Context
# bash run_SARQ-1step.sh $1 $2 0 0 0 1 large # Output

# AttnMap + Context/Output
# bash run_SARQ-1step.sh $1 $2 0 1 1 0 large
# bash run_SARQ-1step.sh $1 $2 0 1 0 1 large # Output + Map

for seed in 42 52 62 72 82
do
    bash run_SARQ-1step.sh $1 $2 large ${seed} # Output + Input Intervention
done

for seed in 42 52 62 72 82
do
    bash run_SARQ-1step.sh $1 $2 base ${seed} # Output + Input Intervention
done