BASE_DATADIR="C:/Users/merlin/data/"

cd ..
cd ..

# run robbing the fed
# non-iid
python server_attack.py \
    --attack rtf \
    --model mlp \
    --imprint \
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
    --device cuda \
    --save_results

# run robbing the fed
# iid
python server_attack.py \
    --attack rtf \
    --model cnn \
    --imprint \
    --iid \
    --base_data_dir $BASE_DATADIR \
    --dataset cifar10 \
    --normalize \
    --total_clients 100 \
    --num_rounds 1 \
    --local_epochs 1 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0 \
    --device cuda \
    --save_results