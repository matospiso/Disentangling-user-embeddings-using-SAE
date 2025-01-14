import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

DATA_FOLDER = "data"
DATASET_RATING_FILE = {
    "ml-25m": "ratings.csv",
}


def load_interactions_dataframe(dataset_name: str) -> pl.DataFrame:
    interactions_df = pl.scan_csv(f"{DATA_FOLDER}/{dataset_name}/{DATASET_RATING_FILE[dataset_name]}")
    if dataset_name == "ml-25m":
        interactions_df = interactions_df.rename({"userId": "user_id", "movieId": "item_id", "rating": "value"})
    interactions_df = (
        interactions_df.select(["user_id", "item_id", "value"])
        .filter(pl.col("value") >= 4.0)
        .cast({"user_id": pl.String, "item_id": pl.String, "value": pl.Float32})
        .cast({"user_id": pl.Categorical, "item_id": pl.Categorical})
        .collect()
    )
    print(f"Dataset info: users={interactions_df['user_id'].n_unique()}, items={interactions_df['item_id'].n_unique()}, interactions={len(interactions_df)}")
    return interactions_df


def convert_to_csr(interactions_df: pl.DataFrame) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    return (
        sp.csr_matrix(
            (
                np.ones(len(interactions_df), dtype=np.float32),
                (interactions_df["user_id"].to_physical().to_numpy(), interactions_df["item_id"].to_physical().to_numpy()),
            ),
            shape=(interactions_df["user_id"].n_unique(), interactions_df["item_id"].n_unique()),
        ),
        interactions_df["user_id"].cat.get_categories().to_numpy(),
        interactions_df["item_id"].cat.get_categories().to_numpy(),
    )


def split_train_val_test_users(
    users, user_item_csr: sp.csr_matrix, val_ratio: float, test_ratio: float, seed: int
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.permutation(user_item_csr.shape[0])
    train_user_idxs = p[: int(-(val_ratio + test_ratio) * len(p))]
    val_user_idxs = p[int(-(val_ratio + test_ratio) * len(p)) : int(-test_ratio * len(p))]
    test_user_idxs = p[int(-test_ratio * len(p)) :]
    train_csr, val_csr, test_csr = user_item_csr[train_user_idxs], user_item_csr[val_user_idxs], user_item_csr[test_user_idxs]
    train_users, val_users, test_users = users[train_user_idxs], users[val_user_idxs], users[test_user_idxs]
    print(f"Train split info: users={train_csr.shape[0]}, items={train_csr.shape[1]}, interactions={train_csr.nnz}")
    print(f"Val split info: users={val_csr.shape[0]}, items={val_csr.shape[1]}, interactions={val_csr.nnz}")
    print(f"Test split info: users={test_csr.shape[0]}, items={test_csr.shape[1]}, interactions={test_csr.nnz}")
    return train_csr, val_csr, test_csr, train_users, val_users, test_users


def prepare_interaction_data(cfg: dict) -> tuple[pl.DataFrame, sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = cfg["dataset"]
    interactions_df = load_interactions_dataframe(dataset)
    interactions_csr, users, items = convert_to_csr(interactions_df)
    train_csr, val_csr, test_csr, train_users, val_users, test_users = split_train_val_test_users(
        users,
        interactions_csr,
        cfg["val_user_ratio"],
        cfg["test_user_ratio"],
        cfg["seed"],
    )
    return interactions_df, train_csr, val_csr, test_csr, train_users, val_users, test_users, items


def split_input_target_interactions(user_item_csr: sp.csr_matrix, target_ratio: float, seed: int) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    rng = np.random.default_rng(seed)
    target_mask = np.concatenate(
        [
            rng.permuted(np.array([True] * int(np.ceil(row_nnz * target_ratio)) + [False] * int((row_nnz - np.ceil(row_nnz * target_ratio)))))
            for row_nnz in np.diff(user_item_csr.indptr)
        ]
    )
    inputs, targets = user_item_csr.copy(), user_item_csr.copy()
    inputs.data *= ~target_mask
    targets.data *= target_mask
    inputs.eliminate_zeros()
    targets.eliminate_zeros()
    return inputs, targets


class Dataloader:
    def __init__(self, data: sp.csr_matrix | np.ndarray | torch.Tensor, batch_size: int, device: torch.device, seed: int | None = None):
        self.data = data
        self.dataset_size = self.data.shape[0]
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed) if seed is not None else None  # if seed = None, loading is deterministic

    def __len__(self) -> int:
        return -(-self.dataset_size // self.batch_size)

    def __iter__(self):
        self.permutation = self.rng.permutation(self.dataset_size) if self.rng is not None else np.arange(self.dataset_size)
        self.i = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self.i >= self.dataset_size:
            raise StopIteration
        next_i = min(self.i + self.batch_size, self.dataset_size)
        batch = self.data[self.permutation[self.i : next_i]]
        self.i = next_i
        return torch.tensor(batch.toarray() if isinstance(batch, sp.csr_matrix) else batch, device=self.device)
