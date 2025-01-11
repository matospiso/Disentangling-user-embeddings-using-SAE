import os
import torch
import torch.nn as nn
import torch.optim as optim

CHECKPOINT_FOLDER = "checkpoints"


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, filepath: str):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    os.makedirs("/".join(filepath.split("/")[:-1]), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filepath: str, device: torch.device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from {filepath} (after {epoch} epochs)")
    return epoch
