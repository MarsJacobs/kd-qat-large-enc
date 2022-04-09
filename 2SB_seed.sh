bash run_SARQ_step_1.sh $1 $2

for var in 42 52 62
do
	bash run_SARQ_step_2.sh $1 $2 $var
done
