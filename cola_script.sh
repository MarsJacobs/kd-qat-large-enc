
for seed in {40..45}
do
   bash run_cola_2.sh 3 cola $seed
done

for seed in {40..45}
do
   bash run_ternary.sh 3 cola $seed
done

