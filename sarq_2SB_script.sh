##----------------------------# (map | cc | co) | (Attn Score | Attnmap | Context | Output)
# map : Teacher Map Intervention
# cc : Teacher Context intervention (w/ Context Loss)
# co : Teacher Context intervention (w/ Attention Output Loss)
# S : Attention Score | M : Attention Map | C : Attention Context | O : Attention Output

for seed in 42 52 62 72 82
do
    bash run_SARQ_step_1.sh $1 $2 $seed 2E-5
    bash run_SARQ_step_2.sh $1 $2 $seed 0

done