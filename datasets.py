import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

DATA_FOLDER = "data"


def load_interactions_dataframe(dataset_name: str) -> pl.DataFrame:
    if dataset_name == "ML-25M":
        interactions_df = (
            pl.scan_csv(f"{DATA_FOLDER}/{dataset_name}/ratings.csv")
            .rename({"userId": "user_id", "movieId": "item_id", "rating": "value"})
            .select(["user_id", "item_id", "value"])
            .filter(pl.col("value") >= 4.0)
            .collect()
        )
        print("Removing users with < 5 interactions...")
        interactions_df = interactions_df.filter(
            interactions_df["user_id"].is_in(interactions_df["user_id"].value_counts().filter(pl.col("count") >= 5)["user_id"])
        )
    elif dataset_name == "MSD":
        # follows MultVAE paper (https://arxiv.org/pdf/1802.05814)
        interactions_df = (
            pl.scan_csv(f"{DATA_FOLDER}/{dataset_name}.txt", has_header=False, separator="\t")
            .rename({"column_1": "user_id", "column_2": "item_id", "column_3": "value"})
            .with_columns(pl.col("value").pow(0).alias("value"))
            .collect()
        )
        print("Removing items with < 200 interactions...")
        interactions_df = interactions_df.filter(
            interactions_df["item_id"].is_in(interactions_df["item_id"].value_counts().filter(pl.col("count") >= 200)["item_id"])
        )
        print("Removing users with < 20 interactions...")
        interactions_df = interactions_df.filter(
            interactions_df["user_id"].is_in(interactions_df["user_id"].value_counts().filter(pl.col("count") >= 20)["user_id"])
        )
    elif dataset_name == "Electronics":
        interactions_df = (
            pl.scan_ndjson(f"{DATA_FOLDER}/{dataset_name}.json")
            .rename({"reviewerID": "user_id", "asin": "item_id", "overall": "value"})
            .select(["user_id", "item_id", "value"])
            .filter(pl.col("value") >= 4.0)
            .collect()
        )
        print("Removing items with < 25 interactions...")
        interactions_df = interactions_df.filter(
            interactions_df["item_id"].is_in(interactions_df["item_id"].value_counts().filter(pl.col("count") >= 25)["item_id"])
        )
        print("Removing users with < 5 interactions...")
        interactions_df = interactions_df.filter(
            interactions_df["user_id"].is_in(interactions_df["user_id"].value_counts().filter(pl.col("count") >= 5)["user_id"])
        )
    interactions_df = interactions_df.cast({"user_id": pl.String, "item_id": pl.String, "value": pl.Float32}).cast(
        {"user_id": pl.Categorical, "item_id": pl.Categorical}
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
    users, user_item_csr: sp.csr_matrix, val_ratio: float, test_ratio: float
) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    p = np.random.permutation(user_item_csr.shape[0])
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
    )
    return interactions_df, train_csr, val_csr, test_csr, train_users, val_users, test_users, items


def split_input_target_interactions(user_item_csr: sp.csr_matrix, target_ratio: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    target_mask = np.concatenate(
        [
            np.random.permutation(np.array([True] * int(np.ceil(row_nnz * target_ratio)) + [False] * int((row_nnz - np.ceil(row_nnz * target_ratio)))))
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
    def __init__(self, data: sp.csr_matrix | np.ndarray | torch.Tensor, batch_size: int, device: torch.device, shuffle: bool = False):
        self.data = data
        self.dataset_size = self.data.shape[0]
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def __len__(self) -> int:
        return -(-self.dataset_size // self.batch_size)

    def __iter__(self):
        self.permutation = np.random.permutation(self.dataset_size) if self.shuffle else np.arange(self.dataset_size)
        self.i = 0
        return self

    def __next__(self) -> torch.Tensor:
        if self.i >= self.dataset_size:
            raise StopIteration
        next_i = min(self.i + self.batch_size, self.dataset_size)
        batch = self.data[self.permutation[self.i : next_i]]
        self.i = next_i
        return torch.tensor(batch.toarray() if isinstance(batch, sp.csr_matrix) else batch, device=self.device)
