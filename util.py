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


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, job_cfg: dict, filepath: str):
    checkpoint = {
        "epoch": epoch,
        "job_cfg": job_cfg,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, job_cfg: dict, filepath: str, device: torch.device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    if not checkpoint["job_cfg"] == job_cfg:
        print(f"Loaded checkpoint from {filepath} does not match current job config\nCheckpoint cfg: {checkpoint['job_cfg']}\nCurrent cfg: {job_cfg}")
        print("Starting from scratch.")
        return 0
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from {filepath} (after {epoch} epochs)")
    return epoch
