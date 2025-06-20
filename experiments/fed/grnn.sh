BASE_DATADIR=".."

cd ..
cd ..

# run grnn
# non-iid
python server_attack.py \
    --attack grnn \
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
    --rec_epochs 100 \
    --rec_batch_size 4 \
    --rec_lr 0.0001 \
    --tv 1e-3 \
    --device cuda \
    --save_results