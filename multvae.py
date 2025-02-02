import torch
import torch.nn as nn
import torch.optim as optim


class Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class MultVAE(nn.Module):
    """Multinomial Variational Autoencoder for collaborative filtering
    Paper: https://arxiv.org/pdf/1802.05814"""

    def __init__(self, input_dim: int, hidden_dims: list[int], embedding_dim: int, annealing_beta: float, annealing_steps: int):
        super().__init__()

        dims = [input_dim] + hidden_dims + [embedding_dim]
        self.encoder = nn.Sequential()
        for i in range(len(dims) - 2):
            linear = Linear(nn.init.xavier_uniform_(torch.empty([dims[i + 1], dims[i]])), torch.zeros(dims[i + 1]))
            self.encoder.add_module("fc{}".format(i), linear)
            self.encoder.add_module("act{}".format(i), nn.Tanh())
        self.encoder_mu = Linear(nn.init.xavier_uniform_(torch.empty([dims[-1], dims[-2]])), torch.zeros(dims[-1]))
        self.encoder_logvar = Linear(nn.init.xavier_uniform_(torch.empty([dims[-1], dims[-2]])), torch.zeros(dims[-1]))

        dims = dims[::-1]
        self.decoder = nn.Sequential()
        for i in range(len(dims) - 1):
            w = nn.init.xavier_uniform_(torch.empty([dims[i + 1], dims[i]]))
            b = torch.zeros(dims[i + 1])
            self.decoder.add_module("fc{}".format(i), Linear(w, b))
            if i != len(dims) - 2:
                self.decoder.add_module("act{}".format(i), nn.Tanh())

        self.annealing_beta = annealing_beta
        self.annealing_steps = annealing_steps
        self.beta = 0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.encoder_mu(h), self.encoder_logvar(h)

    def sample_from_prior(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * logvar.exp().sqrt()

    def decode(self, e: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.decoder(e), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        if self.training:
            return self.decode(self.sample_from_prior(mu, logvar))
        return self.decode(mu)

    def compute_loss_dict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(nn.functional.dropout(batch, 0.5))
        out = self.decode(self.sample_from_prior(mu, logvar))
        nll = -torch.mean(batch * torch.log(torch.clamp(out, min=1e-7)), dim=-1)  # NLL / number of items
        d_kl = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar)), dim=-1)  # D_KL / embedding_dim
        return {"NLL": torch.mean(nll), "D_KL": torch.mean(d_kl), "Loss": torch.mean(nll + self.beta * d_kl)}

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        optimizer.step()
        self.beta = min(self.beta + 1 / self.annealing_steps, self.annealing_beta)
        return losses

    @torch.no_grad()
    def recommend(self, interaction_batch: torch.Tensor, k: int, mask_interactions: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self(interaction_batch)
        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)  # mask input interactions
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
