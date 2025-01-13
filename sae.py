from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util import l2_normalize


class SAE(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, seed: int):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.encoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([input_dim, embedding_dim]), generator=rng))
        self.encoder_b = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_w = nn.Parameter(nn.init.kaiming_uniform_(torch.empty([embedding_dim, input_dim]), generator=rng))
        self.decoder_b = nn.Parameter(torch.zeros(input_dim))
        self.normalize_decoder()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder_w.data = l2_normalize(self.decoder_w.data)
        if self.decoder_w.grad is not None:
            self.decoder_w.grad -= (self.decoder_w.grad * self.decoder_w.data).sum(-1, keepdim=True) * self.decoder_w.data

    def standardize_input(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=-1, keepdim=True)
        x -= x_mean
        x_std = x.std(dim=-1, keepdim=True)
        x /= x_std + 1e-7
        return x, x_mean, x_std

    def destandardize_output(self, x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        return x_mean + x * x_std

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> float:
        _, losses = self(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        self.normalize_decoder()
        optimizer.step()
        return {k: v.item() for k, v in losses.items()}


class BasicSAE(SAE):
    def __init__(self, input_dim: int, embedding_dim: int, seed: int, **extra_params: dict):
        super().__init__(input_dim, embedding_dim, seed)
        self.l1_coef = extra_params["l1_coef"]

    def compute_losses(self, x: torch.Tensor, e: torch.Tensor, x_out: torch.Tensor) -> dict:
        l2_loss = (x_out - x).pow(2).sum(-1).sqrt().div(x.norm(dim=-1)).mean()
        l1_loss = e.abs().sum(-1).mean()
        l0_loss = (e > 0).float().sum(-1).mean()
        loss = l2_loss + self.l1_coef * l1_loss
        return {"Loss": loss, "L2": l2_loss, "L1": l1_loss, "L0": l0_loss}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x, x_mean, x_std = self.standardize_input(x)
        e = F.relu((x - self.decoder_b) @ self.encoder_w + self.encoder_b)
        x_out = e @ self.decoder_w + self.decoder_b
        if not self.training:
            return self.destandardize_output(x_out, x_mean, x_std)
        return self.destandardize_output(x_out, x_mean, x_std), self.compute_losses(x, e, x_out)


class TopKSAE(SAE): ...


class BatchTopKSAE(SAE): ...
