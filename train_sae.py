import argparse
import importlib
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets import Dataloader, prepare_interaction_data
from util import CHECKPOINT_FOLDER, load_config_from_checkpoint, run_training_loop, load_checkpoint


SAE_EXTRA_PARAMS = ["l1_coef", "k"]


def train_sae(cfg: dict, device: torch.device):
    print(f"Training sparse autoencoder using config {cfg}")

    pretrained_model_checkpoint = f"{CHECKPOINT_FOLDER}/{cfg['dataset']}/{cfg['pretrained_model_checkpoint']}"
    pretrained_model_cfg = load_config_from_checkpoint(pretrained_model_checkpoint)
    print(f"Source model config: {pretrained_model_cfg}")

    _, train_csr, val_csr, _, _, _, _, _ = prepare_interaction_data(pretrained_model_cfg)
    train_interaction_dataloader = Dataloader(train_csr, pretrained_model_cfg["batch_size"], device, pretrained_model_cfg["seed"])
    val_interaction_dataloader = Dataloader(val_csr, pretrained_model_cfg["batch_size"], device)

    pretrained_model_class = getattr(importlib.import_module(pretrained_model_cfg["model_module"]), pretrained_model_cfg["model_class"])
    pretrained_model = pretrained_model_class(train_csr.shape[1], pretrained_model_cfg["embedding_dim"], pretrained_model_cfg["seed"]).to(device)
    load_checkpoint(pretrained_model, None, pretrained_model_checkpoint, device)
    train_user_embeddings = np.vstack(
        [
            pretrained_model.encode(batch).detach().cpu().numpy()
            for batch in tqdm(train_interaction_dataloader, desc="Computing user embeddings from train interactions")
        ]
    )
    val_user_embeddings = np.vstack(
        [
            pretrained_model.encode(batch).detach().cpu().numpy()
            for batch in tqdm(val_interaction_dataloader, desc="Computing user embeddings from val interactions")
        ]
    )
    print(f"Train user embeddings shape={train_user_embeddings.shape}, val user embeddings shape={val_user_embeddings.shape}")

    train_dataloader = Dataloader(train_user_embeddings, cfg["batch_size"], device, cfg["seed"])
    val_dataloader = Dataloader(val_user_embeddings, cfg["batch_size"], device)
    sae_model_class = getattr(importlib.import_module(cfg["model_module"]), cfg["model_class"])
    extra_params = {k: cfg[k] for k in cfg.keys() if k in SAE_EXTRA_PARAMS}
    sae_model = sae_model_class(train_user_embeddings.shape[1], cfg["embedding_dim"], cfg["seed"], **extra_params).to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    run_training_loop(sae_model, optimizer, train_dataloader, val_dataloader, cfg, device)

    # TODO load last checkpoint and compute metrics on validation split (1. reconstruction quality (normalized MSE) 2. sparsity (L0), 3. relative recommendation quality degradation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for SAE training script.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--pretrained_model_checkpoint", type=str, required=True, help="Filename of checkpoint containing pre-trained model")
    parser.add_argument("--model_module", type=str, default="sae", help="Module containing SAE model")
    parser.add_argument("--model_class", type=str, default="BasicSAE", help="Model class name")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension of SAE model")
    parser.add_argument("--l1_coef", type=float, default=0.01, help="L1 loss coefficient (BasicSAE)")
    parser.add_argument("--k", type=int, default=32, help="Top K parameter (TopKSAE, BatchTopKSAE)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta_1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta_2 coefficient")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
    cfg = vars(parser.parse_args())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    train_sae(cfg, device)
