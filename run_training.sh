mkdir -p logs
log_file=logs/training_run_$(date +"%Y-%m-%d_%H-%M-%S").log

echo "Training ELSA models..." | tee -a "$log_file"
for embedding_dim in 512 1024 2048; do
    python train_elsa.py --dataset ml-25m --embedding_dim "$embedding_dim" --lr 3e-4 --epochs 50 2> >(cat >&2) | tee -a "$log_file"
done

echo "Training SAE models for individual ELSA checkpoints..." | tee -a "$log_file"
for checkpoint_file in checkpoints/ml-25m/*; do
    checkpoint=$(basename "$checkpoint_file")
    for sae_class in BasicSAE TopKSAE; do
        for scaling_factor in 2 4 8; do
            elsa_embedding_dim=$(echo "$checkpoint" | cut -d'-' -f2)
            sae_embedding_dim=$((elsa_embedding_dim * scaling_factor))
            if [[ "$sae_class" == "BasicSAE" ]]; then
                for l1_coef in 0.01 0.005 0.002 0.001; do
                    python train_sae.py --dataset ml-25m --pretrained_model_checkpoint "$checkpoint" --model_class "$sae_class" --embedding_dim "$sae_embedding_dim" --lr 3e-4 --l1_coef "$l1_coef" --epochs 500 --early_stopping 50 2> >(cat >&2) | tee -a "$log_file"
                done
            elif [[ "$sae_class" == "TopKSAE" ]]; then
                for k in 8 16 32 64; do
                    python train_sae.py --dataset ml-25m --pretrained_model_checkpoint "$checkpoint" --model_class "$sae_class" --embedding_dim "$sae_embedding_dim" --lr 3e-4 --l1_coef 0.001 --k "$k" --epochs 500 --early_stopping 50 2> >(cat >&2) | tee -a "$log_file"
                done
            fi
        done
    done
done