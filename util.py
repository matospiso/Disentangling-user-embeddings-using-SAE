from hashlib import sha256
import json
import random
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets import Dataloader

CHECKPOINT_FOLDER = "checkpoints"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)  # CPU seed
    torch.mps.manual_seed(seed)  # Metal seed
    torch.cuda.manual_seed(seed)  # GPU seed
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # NumPy seed
    random.seed(seed)  # Python seed


def hash_dict(d: dict, length: int = 8) -> str:
    serialized = json.dumps(d, sort_keys=True).encode()
    return sha256(serialized).hexdigest()[:length]


def get_checkpoint_name(cfg: dict) -> str:
    return f"{cfg['model_class']}-{cfg['embedding_dim']}-{hash_dict(cfg)}"


def get_checkpoint_filepath(cfg: dict) -> str:
    return f"{CHECKPOINT_FOLDER}/{cfg['dataset']}/{get_checkpoint_name(cfg)}.ckpt"


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, job_cfg: dict, filepath: str) -> None:
    checkpoint = {
        "epoch": epoch,
        "job_cfg": job_cfg,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_config_from_checkpoint(filepath: str) -> dict:
    return torch.load(filepath, weights_only=False, map_location=torch.device("cpu"))["job_cfg"]


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer | None, filepath: str, device: torch.device, job_cfg: dict | None = None
) -> tuple[int, dict | None]:
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    cfg = checkpoint["job_cfg"]
    if job_cfg is not None and job_cfg != cfg:
        print(f"Loaded checkpoint from {filepath} does not match current job config\nCheckpoint cfg: {checkpoint['job_cfg']}\nCurrent cfg: {job_cfg}")
        print("Starting from scratch.")
        return 0, None
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from {filepath} (after {epoch} epochs)")
    return epoch, cfg


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


def run_training_loop(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: Dataloader,
    val_dataloader: Dataloader,
    cfg: dict,
    device: torch.device,
) -> None:
    checkpoint_path = get_checkpoint_filepath(cfg)
    try:
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device, cfg)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0
    if start_epoch == cfg["epochs"]:
        print(f"Checkpoint already trained for {cfg['epochs']} of {cfg['epochs']} epochs, training is complete.")
        return None
    best_loss, epochs_without_improvement = float("inf"), 0
    start_time = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):
            loss_dict = model.train_step(optimizer, batch)
            pbar.set_description(f"Epoch {epoch + 1}/{cfg['epochs']}", refresh=False)
            pbar.set_postfix({k: v.item() for k, v in loss_dict.items()})
            if i == len(pbar) - 1:
                model.eval()
                val_losses = {k: 0.0 for k in loss_dict.keys()}
                for val_batch in val_dataloader:
                    _losses = {k: v.item() for k, v in model.compute_loss_dict(val_batch).items()}
                    for k in val_losses.keys():
                        val_losses[k] += _losses[k] * val_batch.shape[0] / val_dataloader.dataset_size
                pbar.set_postfix_str(pbar.postfix + " | Val: " + ", ".join([f"{k}={v:.3f}" for k, v in val_losses.items()]))
        if val_losses["Loss"] < best_loss:
            best_loss, epochs_without_improvement = val_losses["Loss"], 0
        else:
            epochs_without_improvement += 1
        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, epoch + 1, cfg, checkpoint_path)
        if epochs_without_improvement >= cfg["early_stopping"]:
            print("Reached early stopping condition, terminating training.")
            break
    save_checkpoint(model, optimizer, epoch + 1, cfg, checkpoint_path)
    print(f"Training loop for {get_checkpoint_name(cfg)} took {time.perf_counter() - start_time:.4f} seconds.")


def evaluate_recall_at_k(model, inputs: Dataloader, targets: Dataloader, k: int) -> np.ndarray:
    recall = []
    for input_batch, target_batch in zip(inputs, targets):
        topk_scores, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
        topk_indices = torch.tensor(topk_indices, device=target_batch.device)
        target_batch = target_batch.bool()
        predicted_batch = torch.zeros_like(target_batch).scatter_(1, topk_indices, torch.ones_like(topk_indices, dtype=bool))
        # recall formula from https://arxiv.org/pdf/1802.05814
        r = (predicted_batch & target_batch).sum(axis=1) / torch.minimum(target_batch.sum(axis=1), torch.ones_like(target_batch.sum(axis=1)) * k)
        recall.append(r)
    return torch.cat(recall).detach().cpu().numpy()


def evaluate_ndcg_at_k(model, inputs: Dataloader, targets: Dataloader, k: int) -> np.ndarray:
    ndcg = []
    for input_batch, target_batch in zip(inputs, targets):
        topk_scores, topk_indices = model.recommend(input_batch, k, mask_interactions=True)
        topk_indices = torch.tensor(topk_indices, device=target_batch.device)
        target_batch = target_batch.bool()
        relevance = target_batch.gather(1, topk_indices).float()
        # DCG@k
        gains = 2**relevance - 1
        discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        dcg = (gains / discounts).sum(dim=1)
        # IDCG@k (ideal DCG)
        sorted_relevance, _ = torch.sort(target_batch.float(), dim=1, descending=True)
        ideal_gains = 2 ** sorted_relevance[:, :k] - 1
        ideal_discounts = torch.log2(torch.arange(2, k + 2, device=relevance.device, dtype=torch.float))
        idcg = (ideal_gains / ideal_discounts).sum(dim=1)
        idcg[idcg == 0] = 1
        # nDCG@k
        ndcg.append(dcg / idcg)
    return torch.cat(ndcg).detach().cpu().numpy()


def evaluate_cosine_similarity(model, inputs: Dataloader) -> np.ndarray:
    cosine = []
    for input_batch in inputs:
        output_batch = model(input_batch)[0]
        cosine.append(nn.functional.cosine_similarity(input_batch, output_batch, 1))
    return torch.cat(cosine).detach().cpu().numpy()
