# Ternary
#bash run_glue.sh $1 $2 42 0 
# 1SB
#bash run_glue.sh $1 $2 42 1 
# 2SB
#bash run_shim_2.sh $1 $2 0 0 qat_step2

# 4bit
bash run_shim_2.sh $1 $2 1 1 qat_step2

# 2SB_Gradual
bash run_shim_2.sh $1 $2 0 0 gradual
