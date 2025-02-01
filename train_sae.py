import argparse
from copy import deepcopy
import importlib
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets import Dataloader, prepare_interaction_data, split_input_target_interactions
from util import (
    CHECKPOINT_FOLDER,
    evaluate_cosine_similarity,
    evaluate_recall_at_k,
    get_checkpoint_filepath,
    get_checkpoint_name,
    load_config_from_checkpoint,
    run_training_loop,
    load_checkpoint,
    set_seed,
)


def train_sae(cfg: dict, device: torch.device):
    print(f"Training sparse autoencoder using config {cfg}")

    pretrained_model_checkpoint = f"{CHECKPOINT_FOLDER}/{cfg['dataset']}/{cfg['pretrained_model_checkpoint']}"
    pretrained_model_cfg = load_config_from_checkpoint(pretrained_model_checkpoint)
    print(f"Source model config: {pretrained_model_cfg}")

    _, train_csr, val_csr, _, _, _, _, _ = prepare_interaction_data(pretrained_model_cfg)
    train_interaction_dataloader = Dataloader(train_csr, pretrained_model_cfg["batch_size"], device)
    val_interaction_dataloader = Dataloader(val_csr, pretrained_model_cfg["batch_size"], device)

    pretrained_model_class = getattr(importlib.import_module(pretrained_model_cfg["model_module"]), pretrained_model_cfg["model_class"])
    pretrained_model = pretrained_model_class(train_csr.shape[1], pretrained_model_cfg["embedding_dim"]).to(device)
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

    train_dataloader = Dataloader(train_user_embeddings, cfg["batch_size"], device, shuffle=True)
    val_dataloader = Dataloader(val_user_embeddings, cfg["batch_size"], device)
    sae_model_class = getattr(importlib.import_module(cfg["model_module"]), cfg["model_class"])
    sae_extra_params = {k: cfg[k] for k in cfg.keys() if k in ["l1_coef", "k"]}
    sae_model = sae_model_class(train_user_embeddings.shape[1], cfg["embedding_dim"], **sae_extra_params).to(device)
    optimizer = optim.Adam(sae_model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    run_training_loop(sae_model, optimizer, train_dataloader, val_dataloader, cfg, device)

    load_checkpoint(sae_model, None, get_checkpoint_filepath(cfg), device, cfg)

    # Evaluation 1: cosine similarity between SAE inputs and SAE outputs
    sae_model.eval()
    cosine_similarity_results = evaluate_cosine_similarity(sae_model, val_dataloader)
    print(
        f"Model = {get_checkpoint_name(cfg)} | Cosine similarity = {np.mean(cosine_similarity_results):.6f} +- {np.std(cosine_similarity_results) / np.sqrt(len(cosine_similarity_results)):.6f}"
    )

    # Evaluation 2: degradation in Recall @ k ( disentangled model (=with SAE inserted) vs unmodified pretrained model )
    val_inputs, val_targets = split_input_target_interactions(val_csr, pretrained_model_cfg["target_interaction_ratio"])
    pretrained_model.eval()
    pretrained_model_recall_results = evaluate_recall_at_k(
        pretrained_model,
        Dataloader(val_inputs, pretrained_model_cfg["batch_size"], device),
        Dataloader(val_targets, pretrained_model_cfg["batch_size"], device),
        pretrained_model_cfg["eval_topk"],
    )

    def forward_with_sae(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(sae_model(self.encode(x))[0]) - x

    disentangled_model = deepcopy(pretrained_model)
    disentangled_model.forward = forward_with_sae.__get__(disentangled_model, disentangled_model.__class__)
    disentangled_model.eval()
    disentangled_model_recall_results = evaluate_recall_at_k(
        disentangled_model,
        Dataloader(val_inputs, pretrained_model_cfg["batch_size"], device),
        Dataloader(val_targets, pretrained_model_cfg["batch_size"], device),
        pretrained_model_cfg["eval_topk"],
    )
    recall_degradations = disentangled_model_recall_results - pretrained_model_recall_results
    print(
        f"Model = {get_checkpoint_name(cfg)} | Recall @ {pretrained_model_cfg['eval_topk']} degradation = {np.mean(recall_degradations):.6f} +- {np.std(recall_degradations) / np.sqrt(len(recall_degradations)):.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for SAE training script.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--pretrained_model_checkpoint", type=str, required=True, help="Filename of checkpoint containing pre-trained model")
    parser.add_argument("--model_module", type=str, default="sae", help="Module containing SAE model")
    parser.add_argument("--model_class", type=str, default="BasicSAE", help="Model class name")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension of SAE model")
    parser.add_argument("--reconstruction_loss", type=str, default="L2", help="Reconstruction loss (L2 or Cosine)")
    parser.add_argument("--l1_coef", type=float, default=0.01, help="L1 loss coefficient (BasicSAE, TopKSAE)")
    parser.add_argument("--k", type=int, default=32, help="Top K parameter (TopKSAE)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta_1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta_2 coefficient")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
    cfg = vars(parser.parse_args())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    set_seed(cfg["seed"])
    train_sae(cfg, device)
