BASE_DATADIR=".."

cd ..
cd ..

# run ggl
# non-iid
python server_attack.py \
    --attack ggl \
    --model mlp \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 1000 \
    --num_rounds 10 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 25000 \
    --device cuda \
    --save_results