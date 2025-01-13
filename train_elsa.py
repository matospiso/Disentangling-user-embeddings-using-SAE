import argparse
import importlib
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets import Dataloader, convert_to_csr, load_interactions, split_input_target_interactions, split_train_val_test_users
from util import get_checkpoint_filepath, save_checkpoint, load_checkpoint


def evaluate_recall_at_k(model, inputs: Dataloader, targets: Dataloader, k: int) -> np.ndarray:
    recall = []
    for input_batch, target_batch in zip(inputs, targets):
        scores = model(input_batch)
        scores = torch.where(input_batch != 0, 0, scores)  # mask input interactions
        topk_scores, topk_indices = torch.topk(scores, k)
        target_batch = target_batch.bool()
        predicted_batch = torch.zeros_like(target_batch).scatter_(1, topk_indices, torch.ones_like(topk_indices, dtype=bool))
        r = (predicted_batch & target_batch).sum(axis=1) / target_batch.sum(axis=1)
        recall.append(r)
    return torch.cat(recall).detach().cpu().numpy()


def train_elsa(cfg: dict, device: torch.device):
    print(f"Training ELSA model using config {cfg}")

    interactions_df = load_interactions(cfg["dataset"])
    interactions_csr = convert_to_csr(interactions_df)
    train_csr, val_csr, _ = split_train_val_test_users(interactions_csr, cfg["val_user_ratio"], cfg["test_user_ratio"], cfg["seed"])
    dataloader = Dataloader(train_csr, cfg["batch_size"], device, cfg["seed"])

    model_class = getattr(importlib.import_module(cfg["model_module"]), cfg["model_class"])
    model = model_class(train_csr.shape[1], cfg["embedding_dim"], cfg["seed"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    checkpoint_path = get_checkpoint_filepath(cfg)
    try:
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device, cfg)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0

    best_result = 0.0
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            loss = model.train_step(optimizer, batch)
            pbar.set_description(f"Epoch {epoch + 1}/{cfg['epochs']}", refresh=False)
            pbar.set_postfix({"Loss": loss})
            if i == len(pbar) - 1:
                model.eval()
                val_inputs, val_targets = split_input_target_interactions(val_csr, cfg["target_interaction_ratio"], cfg["seed"])
                eval_results = evaluate_recall_at_k(
                    model, Dataloader(val_inputs, cfg["batch_size"], device), Dataloader(val_targets, cfg["batch_size"], device), cfg["eval_topk"]
                )
                pbar.set_postfix_str(
                    pbar.postfix + f", Recall@{cfg['eval_topk']}={np.mean(eval_results):.4f}+-{np.std(eval_results) / np.sqrt(len(eval_results)):.4f}"
                )
        if best_result < np.mean(eval_results):
            best_result = np.mean(eval_results)
            save_checkpoint(model, optimizer, epoch + 1, cfg, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for ELSA training script.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--val_user_ratio", type=float, default=0.1, help="Ratio of validation users")
    parser.add_argument("--test_user_ratio", type=float, default=0.1, help="Ratio of test users")
    parser.add_argument("--target_interaction_ratio", type=float, default=0.2, help="Ratio of interactions used as target")
    parser.add_argument("--model_module", type=str, default="elsa", help="Module containing ELSA model")
    parser.add_argument("--model_class", type=str, default="ELSA", help="Model class name")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension of ELSA model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval_topk", type=int, default=20, help="Evalutation top k")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
    cfg = vars(parser.parse_args())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    train_elsa(cfg, device)
