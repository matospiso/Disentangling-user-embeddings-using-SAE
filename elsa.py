import torch
import torch.nn as nn
import torch.optim as optim


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True)


def normalized_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((l2_normalize(y_pred) - l2_normalize(y_true)) ** 2).sum(-1).mean()


class ELSA(nn.Module):
    """Scalable Linear Shallow Autoencoder
    Paper: https://dl.acm.org/doi/abs/10.1145/3523227.3551482"""

    def __init__(self, input_dim: int, embedding_dim: int, seed: int):
        super().__init__()
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.encoder = nn.Parameter(nn.init.xavier_uniform_(torch.empty([input_dim, embedding_dim]), generator=rng))
        self.normalize_encoder()

    def normalize_encoder(self) -> None:
        self.encoder.data = l2_normalize(self.encoder.data)
        if self.encoder.grad is not None:
            self.encoder.grad -= (self.encoder.grad * self.encoder.data).sum(-1, keepdim=True) * self.encoder.data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.encoder

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return e @ self.encoder.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x)) - x

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> float:
        loss = normalized_mse_loss(self(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        self.normalize_encoder()
        optimizer.step()
        return loss.item()
