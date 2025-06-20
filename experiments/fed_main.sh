MODEL="cnn"
BASE_DATADIR=".."
DATASET="cifar10"

cd ..
python fed_main.py \
    --strategy fedavg \
    --model $MODEL \
    --base_data_dir $BASE_DATADIR \
    --dataset $DATASET \
    --normalize \
    --data_augment \
    --p_type dirichlet \
    --beta 0.5 \
    --total_clients 100 \
    --num_rounds 100 \
    --local_epochs 2 \
    --batch_size 50 \
    --lr 0.1 \
    --lr_decay 0.95 \
    --client_momentum 0.5 \
    --save_results

