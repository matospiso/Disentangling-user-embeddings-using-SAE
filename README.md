# Disentangling Interaction-Based User Embeddings Using Sparse Autoencoders

### Environment setup
```bash
conda create -y -n sae python==3.11
conda activate sae

pip install -r requirements.txt
```

### Download [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) dataset
```bash
bash download_ml-25m_dataset.sh
```

### Train ELSA model
```bash
python train_elsa.py --dataset DATASET_NAME --embedding_dim EMBEDDING_DIM
```
See `train_elsa.py` for additional arguments.