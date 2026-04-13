import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim


def load_item_tag_matrix(
    dataset_name: str,
    items: np.ndarray,
    min_tag_count: int = 100,
    weighted: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    items_to_idxs = {str(item_id): idx for idx, item_id in enumerate(items)}

    if dataset_name == "ML-25M":
        tag_df = (
            pl.scan_csv(f"data/{dataset_name}/tags.csv")
            .rename({"userId": "user_id", "movieId": "item_id"})
            .cast({"user_id": pl.String, "item_id": pl.String})
            .with_columns(pl.col("tag").str.to_lowercase().str.strip_chars().alias("tag"))
            .filter(pl.col("tag").count().over("tag") >= min_tag_count)
            .filter(pl.col("item_id").is_in(items))
            .with_columns(pl.col("item_id").replace_strict(items_to_idxs).alias("item_idx"))
            .cast({"tag": pl.Categorical})
            .collect()
        )
        values = np.ones(len(tag_df), dtype=np.float32)
    elif dataset_name == "MSD":
        tag_path = "data/MSD/song_tag_assignment.csv"
        tag_df = (
            pl.scan_csv(tag_path)
            .rename({"song_id": "item_id"})
            .cast({"item_id": pl.String})
            .with_columns(pl.col("tag").str.to_lowercase().str.strip_chars().alias("tag"))
            .filter(pl.col("tag").count().over("tag") >= min_tag_count)
            .filter(pl.col("item_id").is_in(items))
            .with_columns(pl.col("item_id").replace_strict(items_to_idxs).alias("item_idx"))
            .cast({"tag": pl.Categorical})
            .collect()
        )
        if weighted and "weight" in tag_df.columns:
            values = tag_df["weight"].cast(pl.Float32).to_numpy()
        else:
            values = np.ones(len(tag_df), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported dataset for TEASER metadata: {dataset_name}")

    tags = tag_df["tag"].cat.get_categories().to_numpy()
    item_tag = sp.csr_matrix(
        (
            values,
            (
                tag_df["item_idx"].to_numpy(),
                tag_df["tag"].to_physical().to_numpy(),
            ),
        ),
        shape=(len(items), len(tags)),
        dtype=np.float32,
    )
    item_tag.data[:] = 1.0 if not weighted else item_tag.data
    item_tag.eliminate_zeros()
    return torch.tensor(item_tag.toarray(), dtype=torch.float32), tags


class TEASERGD(nn.Module):
    def __init__(
        self,
        item_tag_matrix: torch.Tensor,
        reconstruction_loss: str = "MSE",
        lambda1: float = 1e-3,
        lambda2: float = 1e-4,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.reconstruction_loss = reconstruction_loss
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_items, self.num_tags = item_tag_matrix.shape
        self.encoder = nn.Parameter(torch.randn(self.num_items, self.num_tags) * init_std)
        self.register_buffer("item_tag_matrix", item_tag_matrix)

    def decoder(self) -> torch.Tensor:
        return self.item_tag_matrix

    def user_profiles(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.encoder

    def item_diagonal(self, decoder: torch.Tensor | None = None) -> torch.Tensor:
        decoder = self.decoder() if decoder is None else decoder
        return (self.encoder * decoder).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder()
        profiles = self.user_profiles(x)
        scores = profiles @ decoder.T
        return scores - x * self.item_diagonal(decoder).unsqueeze(0)

    def steer_scores(self, x: torch.Tensor, feedback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decoder = self.decoder()
        profiles = self.user_profiles(x)
        steered_profiles = profiles + feedback
        scores = steered_profiles @ decoder.T
        return scores, steered_profiles

    def compute_loss_dict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        decoder = self.decoder()
        out = self(batch)
        recon_losses = {
            "MSE": (out - batch).pow(2).mean(),
            "Cosine": (1 - nn.functional.cosine_similarity(out, batch, dim=1, eps=1e-8)).mean(),
        }
        recon = recon_losses[self.reconstruction_loss]

        gram = decoder.T @ decoder
        item_item_norm_sq = ((self.encoder @ gram) * self.encoder).sum()
        diagonal_sq = self.item_diagonal(decoder).pow(2).sum()
        offdiag_norm_sq = item_item_norm_sq - diagonal_sq

        losses = {
            "Recon": recon,
            "MSE": recon_losses["MSE"],
            "Cosine": recon_losses["Cosine"],
            "B Fro": offdiag_norm_sq / (self.num_items**2),
            "E Fro": self.encoder.pow(2).mean(),
        }
        losses["Loss"] = losses["Recon"] + self.lambda1 * losses["B Fro"] + self.lambda2 * losses["E Fro"]
        return losses

    def train_step(self, optimizer: optim.Optimizer, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.compute_loss_dict(batch)
        optimizer.zero_grad()
        losses["Loss"].backward()
        optimizer.step()
        return losses

    @torch.no_grad()
    def recommend(self, interaction_batch: torch.Tensor, k: int, mask_interactions: bool = True) -> tuple[np.ndarray, np.ndarray]:
        scores = self(interaction_batch)
        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)
        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()
