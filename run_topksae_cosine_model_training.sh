#!/bin/bash
echo "Training SAE models..."
for cf_model in ELSA MultVAE; do
    for checkpoint_file in checkpoints/"$DATASET"/"$cf_model"*; do
        checkpoint=$(basename "$checkpoint_file")
        for scaling_factor in 2 4 8; do
            cf_embedding_dim=$(echo "$checkpoint" | cut -d'-' -f2)
            sae_embedding_dim=$((cf_embedding_dim * scaling_factor))
            for k in 8 16 32 64; do
                SAVE_CKPT=False python train_sae.py --dataset "$DATASET" --pretrained_model_checkpoint "$checkpoint" --model_class TopKSAE --embedding_dim "$sae_embedding_dim" --reconstruction_loss Cosine --lr 3e-4 --l1_coef 0.0003 --k "$k" --epochs 250 --early_stopping 50 &
            done
            wait
        done
    done
done