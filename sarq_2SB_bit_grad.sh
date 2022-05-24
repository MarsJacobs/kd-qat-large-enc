##----------------------------# (map | cc | co) | (Attn Score | Attnmap | Context | Output)
# map : Teacher Map Intervention
# cc : Teacher Context intervention (w/ Context Loss)
# co : Teacher Context intervention (w/ Attention Output Loss)
# S : Attention Score | M : Attention Map | C : Attention Context | O : Attention Output

for seed in 42 52 62 72 82
do
    qk_FP=1
    stop_grad_step_1=1
    stop_grad_step_2=1

    bash run_SARQ_step_1.sh $1 $2 $qk_FP $stop_grad_step_1 $seed
    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    qk_FP=1
    stop_grad_step_1=1
    stop_grad_step_2=0

    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    # ==================================================================#

    qk_FP=1
    stop_grad_step_1=0
    stop_grad_step_2=1

    bash run_SARQ_step_1.sh $1 $2 $qk_FP $stop_grad_step_1 $seed
    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    qk_FP=1
    stop_grad_step_1=0
    stop_grad_step_2=0

    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    # ==================================================================#

    qk_FP=0
    stop_grad_step_1=1
    stop_grad_step_2=1

    bash run_SARQ_step_1.sh $1 $2 $qk_FP $stop_grad_step_1 $seed
    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    qk_FP=0
    stop_grad_step_1=1
    stop_grad_step_2=0

    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    # ==================================================================#

    qk_FP=0
    stop_grad_step_1=0
    stop_grad_step_2=1

    bash run_SARQ_step_1.sh $1 $2 $qk_FP $stop_grad_step_1 $seed
    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    qk_FP=0
    stop_grad_step_1=0
    stop_grad_step_2=0

    bash run_SARQ_step_2.sh $1 $2 $stop_grad_step_2 $seed

    # ==================================================================#
done