for num in 7 9 10 11
do                             
    bash run_kd_single_layer.sh $1 $2 $num base
done

for num in 15 18 20 21 22 23 
do                             
    bash run_kd_single_layer.sh $1 $2 $num large
done