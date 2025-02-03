#!/bin/bash
echo "Training SAE models..."
for checkpoint_file in checkpoints/"$DATASET"/"$CF_MODEL"*; do
    checkpoint=$(basename "$checkpoint_file")
    for sae_class in BasicSAE TopKSAE; do
        for scaling_factor in 2 4 8; do
            cf_embedding_dim=$(echo "$checkpoint" | cut -d'-' -f2)
            sae_embedding_dim=$((cf_embedding_dim * scaling_factor))
            if [[ "$sae_class" == "BasicSAE" ]]; then
                for l1_coef in 0.01 0.003 0.001 0.0003; do
                    SAVE_CKPT=False python train_sae.py --dataset "$DATASET" --pretrained_model_checkpoint "$checkpoint" --model_class "$sae_class" --embedding_dim "$sae_embedding_dim" --lr 3e-4 --l1_coef "$l1_coef" --epochs 250 --early_stopping 50 &
                done
                wait
            elif [[ "$sae_class" == "TopKSAE" ]]; then
                for k in 8 16 32 64; do
                    SAVE_CKPT=False python train_sae.py --dataset "$DATASET" --pretrained_model_checkpoint "$checkpoint" --model_class "$sae_class" --embedding_dim "$sae_embedding_dim" --lr 3e-4 --l1_coef 0.0003 --k "$k" --epochs 250 --early_stopping 50 &
                done
                wait
            fi
        done
    done
done