import importlib
import io
from pathlib import Path
import zipfile
import numpy as np
import polars as pl
import requests
import scipy.sparse as sp
import torch
import torch.nn as nn

from datasets import DATA_FOLDER, prepare_interaction_data, Dataloader
from util import CHECKPOINT_FOLDER, get_checkpoint_filepath, load_checkpoint, load_config_from_checkpoint
from elsa import ELSA
from sae import SAE

device = torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.mps.is_available() else torch.device("cpu"))

DATASET = "ML-25M"
SAE_CHECKPOINT_PATH = f"{CHECKPOINT_FOLDER}/{DATASET}/TopKSAE-8192-d6337b64.ckpt"

# If DATA_FOLDER / CHECKPOINT_FOLDER are strings in your code, you can wrap them:
DATA_FOLDER_PATH = Path(DATA_FOLDER)
CHECKPOINT_FOLDER_PATH = Path(CHECKPOINT_FOLDER)
ML25M_DIR = DATA_FOLDER_PATH / DATASET
ML25M_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

DEFAULT_K = 50


def ensure_ml25m_data():
    """
    Ensure data/ML-25M exists.
    If not, download and unzip Movielens 25M into DATA_FOLDER/ML-25M.
    """
    if ML25M_DIR.exists():
        return

    DATA_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    # Download zip into memory (simplest; for very big data you might stream to disk instead)
    resp = requests.get(ML25M_ZIP_URL)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(DATA_FOLDER_PATH)

    # The zip unpacks into "ml-25m" – rename to "ML-25M"
    unpacked = DATA_FOLDER_PATH / "ml-25m"
    if unpacked.exists():
        unpacked.rename(ML25M_DIR)


def compute_tfidf(X: np.ndarray) -> np.ndarray:
    tf = X / np.sum(X, axis=0, keepdims=True)
    tf[np.isnan(tf)] = 0
    df = np.count_nonzero(X, axis=1)
    num_documents = X.shape[1]
    idf = np.log((num_documents + 1) / (df + 1)) + 1
    return tf * idf[:, np.newaxis]


def get_tag_item_counts(tag_df: pl.DataFrame, items: np.ndarray, user_subset=None):
    tags = tag_df["tag"].unique(maintain_order=True).to_numpy()
    num_tags = tag_df["tag"].n_unique()
    num_items = len(items)
    if user_subset is not None:
        tag_df = tag_df.filter(pl.col("user_id").is_in(user_subset))
    tag_item_counts = sp.csr_matrix(
        (
            np.ones(len(tag_df), dtype=np.float32),
            (tag_df["tag"].to_physical().to_numpy(), tag_df["item_idx"].to_numpy()),
        ),
        shape=(num_tags, num_items),
    )
    return tags, tag_item_counts


class SAESteeredModel:
    """
    Steering concept IDs correspond to tag indices.
    """

    def __init__(self, base_model, sae: SAE, concept_neuron_mapping: torch.Tensor, alpha: float):
        self.base_model = base_model
        self.sae = sae
        self.concept_neuron_mapping = concept_neuron_mapping.to(device)
        self.alpha = alpha

    def eval(self):
        self.base_model.eval()

    @torch.no_grad()
    def encode(self, interaction_batch: torch.Tensor):
        user_embeddings = self.base_model.encode(interaction_batch)
        if self.base_model.__class__ != ELSA:
            user_embeddings = user_embeddings[0]
        sae_embeddings, _, input_mean, input_std = self.sae.encode(user_embeddings)
        return sae_embeddings, input_mean, input_std

    @torch.no_grad()
    def decode(self, sae_embeddings: torch.Tensor, input_mean: torch.Tensor, input_std: torch.Tensor):
        return self.base_model.decode(self.sae.decode(sae_embeddings, input_mean, input_std))

    @torch.no_grad()
    def recommend(
        self,
        interaction_batch: torch.Tensor,
        steering_concept_ids: torch.Tensor | None,
        k: int = 20,
        mask_interactions: bool = True,
    ):
        sae_embeddings, input_mean, input_std = self.encode(interaction_batch)

        if steering_concept_ids is not None:
            neuron_batch = self.concept_neuron_mapping[steering_concept_ids]
            s = sae_embeddings.sum(-1, keepdim=True)
            sae_embeddings *= (1 - self.alpha) / s
            sae_embeddings[torch.arange(neuron_batch.shape[0]), neuron_batch] += self.alpha
            sae_embeddings *= s

        if self.base_model.__class__ != ELSA:
            scores = self.decode(sae_embeddings, input_mean, input_std)
        else:
            scores = nn.ReLU()(self.decode(sae_embeddings, input_mean, input_std) - interaction_batch)

        if mask_interactions:
            scores = torch.where(interaction_batch != 0, 0, scores)

        topk_scores, topk_indices = torch.topk(scores, k)
        return topk_scores.cpu().numpy(), topk_indices.cpu().numpy()


def load_artifacts(alpha: float = 0.0) -> dict:
    """
    Do all the heavy lifting once:
    - load ELSA + SAE
    - prepare interactions
    - build tag↔neuron mapping
    - build steered model wrapper
    """
    ensure_ml25m_data()

    # --- SAE & ELSA configs / weights ---
    sae_cfg = load_config_from_checkpoint(SAE_CHECKPOINT_PATH)
    elsa_checkpoint_path = f"{CHECKPOINT_FOLDER}/{DATASET}/{sae_cfg['pretrained_model_checkpoint']}"
    elsa_cfg = load_config_from_checkpoint(elsa_checkpoint_path)

    _, train_csr, val_csr, test_csr, train_users, val_users, test_users, items = prepare_interaction_data(elsa_cfg)

    train_users_to_idxs = {uid: uidx for uidx, uid in enumerate(train_users)}
    val_users_to_idxs = {uid: uidx for uidx, uid in enumerate(val_users)}
    test_users_to_idxs = {uid: uidx for uidx, uid in enumerate(test_users)}
    items_to_idxs = {iid: iidx for iidx, iid in enumerate(items)}

    # Polars DataFrame with: item_id, title, genres, image_url
    items_df = (
        # movies.csv is original ML-25M file with (movieId, title, genres)
        pl.scan_csv(ML25M_DIR / "movies.csv")
        .rename({"movieId": "item_id"})
        .cast({"item_id": pl.String})
        # images.csv: your custom file with (item_id, image_url)
        .join(pl.scan_csv(ML25M_DIR / "images.csv").rename({"movieId": "item_id"}).cast({"item_id": pl.String}), on="item_id", how="left")
        .collect()
    )

    # base ELSA
    elsa_model_class = getattr(importlib.import_module(elsa_cfg["model_module"]), elsa_cfg["model_class"])
    elsa = elsa_model_class(train_csr.shape[1], elsa_cfg["embedding_dim"]).to(device)
    _ = load_checkpoint(elsa, None, get_checkpoint_filepath(elsa_cfg), device, None)
    elsa.eval()

    # SAE
    sae_model_class = getattr(importlib.import_module(sae_cfg["model_module"]), sae_cfg["model_class"])
    sae_extra_params = {k: sae_cfg[k] for k in sae_cfg.keys() if k in ["l1_coef", "k"]}
    sae = sae_model_class(
        elsa_cfg["embedding_dim"],
        sae_cfg["embedding_dim"],
        sae_cfg["reconstruction_loss"],
        **sae_extra_params,
    ).to(device)
    _ = load_checkpoint(sae, None, get_checkpoint_filepath(sae_cfg), device, None)
    sae.eval()

    # --- tags / tag_df ---
    tag_df = (
        pl.scan_csv(f"{DATA_FOLDER}/{DATASET}/tags.csv")
        .rename({"userId": "user_id", "movieId": "item_id"})
        .cast({"user_id": pl.String, "item_id": pl.String})
        .with_columns(pl.col("tag").str.to_lowercase().str.strip_chars().alias("tag"))
        .filter(pl.col("tag").count().over("tag") >= 100)
        .filter(pl.col("item_id").is_in(items))
        .with_columns(pl.col("item_id").replace_strict(items_to_idxs).alias("item_idx"))
        .cast({"tag": pl.Categorical})
        .collect()
    )

    tags, train_tag_item_counts = get_tag_item_counts(tag_df, items, user_subset=train_users)

    # --- sparse item embeddings ---
    sparse_item_embeddings_list = []

    onehot_items_dataloader = Dataloader(
        sp.eye(len(items), dtype=np.float32, format="csr"),
        batch_size=1024,
        device=device,
    )

    with torch.no_grad():
        for onehot_batch in onehot_items_dataloader:
            user_embedding = elsa.encode(onehot_batch)
            sae_embedding, _, input_mean, input_std = sae.encode(user_embedding)
            sparse_item_embeddings_list.append(sp.csr_matrix(sae_embedding.cpu().numpy()))

    sparse_item_embeddings = sp.vstack(sparse_item_embeddings_list)

    pti = train_tag_item_counts.copy()
    pti.data /= pti.data.sum()

    tag_neuron_activity = pti @ sparse_item_embeddings  # tag x neuron

    # tag -> neuron whose firing is most unique to that tag
    top_neuron_per_tag = compute_tfidf(tag_neuron_activity.toarray()).argmax(axis=1)

    # concept_neuron_mapping: concept = tag index
    concept_neuron_mapping = torch.tensor(top_neuron_per_tag, dtype=torch.long)

    # Build the steered model wrapper
    steered_model = SAESteeredModel(elsa, sae, concept_neuron_mapping, alpha=alpha)
    steered_model.eval()

    artifacts = {
        "device": device,
        "elsa": elsa,
        "steered_model": steered_model,
        "train_csr": train_csr,
        "val_csr": val_csr,
        "test_csr": test_csr,
        "train_users": train_users,
        "val_users": val_users,
        "test_users": test_users,
        "train_users_to_idxs": train_users_to_idxs,
        "val_users_to_idxs": val_users_to_idxs,
        "test_users_to_idxs": test_users_to_idxs,
        "items": items,
        "items_df": items_df,
        "tags": tags,
        "tag_df": tag_df,
        "train_tag_to_idx": {t: i for i, t in enumerate(tags)},
    }
    return artifacts


def all_user_ids(artifacts) -> np.ndarray:
    return np.concatenate(
        [
            artifacts["train_users"],
            artifacts["val_users"],
            artifacts["test_users"],
        ]
    ).astype(str)


def all_neuron_labels(artifacts) -> np.ndarray:
    # use tag strings as neuron labels
    return artifacts["tags"].astype(str)


def _get_user_row(artifacts, user_id: str):
    """Return (csr_matrix_row, csr_full) or (None, None) if user not found."""
    if user_id in artifacts["train_users_to_idxs"]:
        idx = artifacts["train_users_to_idxs"][user_id]
        csr = artifacts["train_csr"]
    elif user_id in artifacts["val_users_to_idxs"]:
        idx = artifacts["val_users_to_idxs"][user_id]
        csr = artifacts["val_csr"]
    elif user_id in artifacts["test_users_to_idxs"]:
        idx = artifacts["test_users_to_idxs"][user_id]
        csr = artifacts["test_csr"]
    else:
        return None, None
    return csr[idx], csr


def user_interactions_df(artifacts, user_id: str, max_items: int = DEFAULT_K) -> pl.DataFrame:
    row, _ = _get_user_row(artifacts, user_id)

    if row is None:
        return pl.DataFrame()

    item_idxs = row.indices
    if len(item_idxs) == 0:
        return pl.DataFrame()

    items_df: pl.DataFrame = artifacts["items_df"]
    items = artifacts["items"]
    # map item_idx -> item_id as string
    item_ids = [str(items[i]) for i in item_idxs]

    # Polars-native filter
    df = items_df.filter(pl.col("item_id").is_in(item_ids))
    # Optionally limit
    return df.head(max_items)


def _recommend_common(
    artifacts,
    user_id: str,
    steering_concept_ids: torch.Tensor | None,
    alpha: float | None,
    k: int = 30,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    row, _ = _get_user_row(artifacts, user_id)
    if row is None:
        return None, None

    interaction_batch = torch.tensor(row.toarray(), dtype=torch.float32, device=artifacts["device"])
    steered_model: SAESteeredModel = artifacts["steered_model"]
    if alpha is not None:
        steered_model.alpha = alpha
    scores, indices = steered_model.recommend(interaction_batch, steering_concept_ids, k=k)
    return scores[0], indices[0]


def recommend_unsteered(artifacts, user_id: str, k: int = DEFAULT_K) -> pl.DataFrame:
    scores, indices = _recommend_common(artifacts, user_id, steering_concept_ids=None, alpha=None, k=k)
    if scores is None:
        return pl.DataFrame()

    items_df: pl.DataFrame = artifacts["items_df"]
    items = artifacts["items"]
    item_ids = [str(items[i]) for i in indices]

    # Build ranking dataframe to preserve order
    ranking_df = pl.DataFrame(
        {
            "item_id": item_ids,
            "score": scores,
        }
    )

    # Join with metadata using Polars
    df = ranking_df.join(items_df, on="item_id", how="left")
    return df


def recommend_steered(artifacts, user_id: str, neuron_label: str, alpha: float, k: int = DEFAULT_K) -> pl.DataFrame:
    tag_to_idx = artifacts["train_tag_to_idx"]
    if neuron_label not in tag_to_idx:
        return pl.DataFrame()

    tag_idx = tag_to_idx[neuron_label]
    steering_concept_ids = torch.tensor([tag_idx], dtype=torch.long, device=artifacts["device"])

    scores, indices = _recommend_common(artifacts, user_id, steering_concept_ids, alpha, k=k)
    if scores is None:
        return pl.DataFrame()

    items_df: pl.DataFrame = artifacts["items_df"]
    items = artifacts["items"]
    item_ids = [str(items[i]) for i in indices]

    ranking_df = pl.DataFrame(
        {
            "item_id": item_ids,
            "score": scores,
        }
    )

    df = ranking_df.join(items_df, on="item_id", how="left")
    return df
