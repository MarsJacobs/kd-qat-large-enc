##----------------------------# (map | cc | co) | (Attn Score | Attnmap | Context | Output)
# map : Teacher Map Intervention
# cc : Teacher Context intervention (w/ Context Loss)
# co : Teacher Context intervention (w/ Attention Output Loss)
# S : Attention Score | M : Attention Map | C : Attention Context | O : Attention Output
bash run_SARQ_step_1_TI.sh $1 $2 
    
for seed in 42 52 62 72 82
do                                     #S M C O 
    bash run_SARQ_step_2.sh $1 $2 $seed 1 1 0 0 map
done
