for temp in 0.1 1 5 10 20
do
	for seed in 42 52 62
    do
        bash run_kl_scaling.sh $1 $2 $temp $seed base
    done
done

for temp in 0.1 1 5 10 20
do
	for seed in 42 52 62
    do
        bash run_kl_scaling.sh $1 $2 $temp $seed large
    done
done