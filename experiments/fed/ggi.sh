BASE_DATADIR=".."

cd ..
cd ..

python server_attack.py \
    --attack ggi \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar100 \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 1000 \
    --num_rounds 5 \
    --local_epochs 5 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 4000 \
    --rec_batch_size 1 \
    --rec_lr 0.1 \
    --tv 1e-6 \
    --device cuda \
    --save_results