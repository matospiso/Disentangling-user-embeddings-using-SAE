for embedding_dim in 512 1024 2048; do
    python train_elsa.py --dataset ml-25m --embedding_dim "$embedding_dim" --epochs 1 2> >(cat >&2) | tee logs/elsa.log
done