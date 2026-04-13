import argparse
import ast
import importlib
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim

from datasets import Dataloader, prepare_interaction_data, split_input_target_interactions
from teaser import load_item_tag_matrix
from util import (
    evaluate_ndcg_at_k,
    evaluate_recall_at_k,
    get_checkpoint_name,
    get_results_filepath,
    run_training_loop,
    save_results,
    set_seed,
)


def evaluate_on_split(model, split: sp.csr_matrix, cfg: dict, device: torch.device) -> dict:
    inputs, targets = split_input_target_interactions(split, cfg["target_interaction_ratio"])
    inputs, targets = Dataloader(inputs, cfg["batch_size"], device), Dataloader(targets, cfg["batch_size"], device)
    model.eval()
    recalls = evaluate_recall_at_k(model, inputs, targets, cfg["eval_topk"])
    ndcgs = evaluate_ndcg_at_k(model, inputs, targets, cfg["eval_topk"])
    return {
        "recall": {"mean": float(np.mean(recalls)), "se": float(np.std(recalls) / np.sqrt(len(recalls)))},
        "ndcg": {"mean": float(np.mean(ndcgs)), "se": float(np.std(ndcgs) / np.sqrt(len(ndcgs)))},
    }


def train_teaser(cfg: dict, device: torch.device):
    print(f"Training TEASER model using config {cfg}")

    _, train_csr, val_csr, test_csr, _, _, _, items = prepare_interaction_data(cfg)
    train_dataloader = Dataloader(train_csr, cfg["batch_size"], device, shuffle=True)
    val_dataloader = Dataloader(val_csr, cfg["batch_size"], device)

    item_tag_matrix, tags = load_item_tag_matrix(cfg["dataset"], items, cfg["tag_min_count"], weighted=ast.literal_eval(cfg["weighted_tags"]))
    cfg["embedding_dim"] = int(item_tag_matrix.shape[1])
    print(f"Loaded metadata matrix with shape={tuple(item_tag_matrix.shape)} and {len(tags)} tags")

    model_class = getattr(importlib.import_module(cfg["model_module"]), cfg["model_class"])
    model = model_class(
        item_tag_matrix.to(device),
        reconstruction_loss=cfg["reconstruction_loss"],
        lambda1=cfg["lambda1"],
        lambda2=cfg["lambda2"],
        init_std=cfg["init_std"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    run_training_loop(model, optimizer, train_dataloader, val_dataloader, cfg, device, save_ckpt=ast.literal_eval(os.environ.get("SAVE_CKPT", "True")))

    results = {"val": evaluate_on_split(model, val_csr, cfg, device), "test": evaluate_on_split(model, test_csr, cfg, device)}
    for split, split_res in results.items():
        for m in split_res.keys():
            print(f"model = {get_checkpoint_name(cfg)} | split = {split} | {m} @ {cfg['eval_topk']} = {split_res[m]['mean']:.6f} +- {split_res[m]['se']:.6f}")
    save_results(results, cfg, get_results_filepath(cfg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for TEASER training script.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--val_user_ratio", type=float, default=0.1, help="Ratio of validation users")
    parser.add_argument("--test_user_ratio", type=float, default=0.1, help="Ratio of test users")
    parser.add_argument("--target_interaction_ratio", type=float, default=0.2, help="Ratio of interactions used as target")
    parser.add_argument("--model_module", type=str, default="teaser", help="Module containing TEASER model")
    parser.add_argument("--model_class", type=str, default="TEASERGD", help="Model class name")
    parser.add_argument("--tag_min_count", type=int, default=100, help="Minimum metadata tag frequency")
    parser.add_argument("--weighted_tags", type=str, default="False", help="Whether to use weighted metadata assignments when available")
    parser.add_argument("--reconstruction_loss", type=str, default="MSE", help="Reconstruction loss (MSE or Cosine)")
    parser.add_argument("--lambda1", type=float, default=1e-3, help="Regularization coefficient on the metadata-mediated item-item matrix")
    parser.add_argument("--lambda2", type=float, default=1e-4, help="Regularization coefficient on the encoder")
    parser.add_argument("--init_std", type=float, default=0.01, help="Encoder weight initialization standard deviation")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta_1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta_2 coefficient")
    parser.add_argument("--eval_topk", type=int, default=20, help="Evalutation top k")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
    cfg = vars(parser.parse_args())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    set_seed(cfg["seed"])
    train_teaser(cfg, device)
