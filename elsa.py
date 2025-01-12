import torch
import torch.nn as nn


def l2_normalize(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    return x.div(x.pow(2).sum(axis, True).sqrt())


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true) ** 2).sum(-1).mean()


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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.encoder

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return e @ self.encoder.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x)) - x
