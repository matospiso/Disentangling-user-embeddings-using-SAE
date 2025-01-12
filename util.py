from hashlib import sha256
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

CHECKPOINT_FOLDER = "checkpoints"


def hash_dict(d: dict, length: int = 8) -> str:
    serialized = json.dumps(d, sort_keys=True).encode()
    return sha256(serialized).hexdigest()[:length]


def get_checkpoint_filepath(cfg: dict) -> str:
    return f"{CHECKPOINT_FOLDER}/{cfg['dataset']}/{cfg['model_class']}-{cfg['embedding_dim']}-{hash_dict(cfg)}.ckpt"


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
    return torch.load(filepath, weights_only=False)["job_cfg"]


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
