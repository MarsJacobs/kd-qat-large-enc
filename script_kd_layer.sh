for num in 0 1 2 3 4 5 6
do                             
    bash run_kd_layer.sh $1 $2 $num base
done

for num in 0 1 2 3 4 5 6 
do                             
    bash run_kd_layer.sh $1 $2 $num large
done