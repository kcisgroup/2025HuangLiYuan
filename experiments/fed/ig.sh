BASE_DATADIR=".."

cd ..
cd ..

# run ig_single
# non-iid
python server_attack.py \
    --attack ig_single \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 1000 \
    --num_rounds 5 \
    --local_epochs 10 \
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

# run weight updates
# non-iid
python server_attack.py \
    --attack ig_multi \
    --model cnn \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 100 \
    --num_rounds 5 \
    --local_epochs 10 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 8000 \
    --rec_batch_size 1 \
    --rec_lr 0.1 \
    --tv 1e-6 \
    --device cuda \
    --save_results

# run multi updates
# non-iid
python server_attack.py \
    --attack ig_multi \
    --model mlp \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 100 \
    --num_rounds 1 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --rec_epochs 24000 \
    --rec_batch_size 8 \
    --rec_lr 1 \
    --tv 1e-6 \
    --device cuda \
    --save_results

