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

### Checkpoints
Training jobs store checkpoints after surpassing previous best evaluation result. Each checkpoint file contains
- the training job config,  
- the model and optimizer states, and 
- the number of completed epochs.

The checkpoint files use the following naming structure:
```
MODEL_CLASS-EMBEDDING_DIM-HASH.ckpt  # e.g., ELSA-512-92a7c516.ckpt
```
where HASH is the first 8 characters of the SHA256 fingerprint of the serialized job config. Only the `model_class` and `embedding_dim` are apparent from the naming. To inspect the full config stored in the checkpoint, run
```bash
python -c "from torch import load; checkpoint_path='checkpoints/DATASET_NAME/CHECKPOINT_FILE'; print(load(checkpoint_path, weights_only=False)['job_cfg'])"
```
