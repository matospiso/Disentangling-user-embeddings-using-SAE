#!/bin/bash
echo "Training ELSA models..."
for embedding_dim in 512 1024 2048; do
    python train_elsa.py --dataset "$DATASET" --embedding_dim "$embedding_dim" --lr 3e-4 --epochs 25 --early_stopping 10 &
done
wait

echo "Training MultVAE models..."
for embedding_dim in 256 512 1024; do
    python train_multvae.py --dataset "$DATASET" --embedding_dim "$embedding_dim" --hidden_dims "$(($embedding_dim * 3))" --lr 1e-3 --epochs 25 &
done
wait