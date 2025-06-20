BASE_DATADIR=".."

cd ..
cd ..


python grad_attack.py \
    --attack ggi \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --device cuda \
    --save_results
