import argparse
import importlib
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from datasets import Dataloader, convert_to_csr, load_interactions, split_input_target_interactions, split_train_val_test_users
from util import CHECKPOINT_FOLDER, get_checkpoint_filepath, load_config_from_checkpoint, save_checkpoint, load_checkpoint


SAE_EXTRA_PARAMS = ["l1_coef", "k"]


def train_sae(cfg: dict, device: torch.device):
    print(f"Training sparse autoencoder using config {cfg}")

    pretrained_model_checkpoint = f"{CHECKPOINT_FOLDER}/{cfg['dataset']}/{cfg['pretrained_model_checkpoint']}"
    pretrained_model_cfg = load_config_from_checkpoint(pretrained_model_checkpoint)
    print(f"Source model config: {pretrained_model_cfg}")

    dataset = cfg["dataset"]
    interactions_df = load_interactions(dataset)
    interactions_csr = convert_to_csr(interactions_df)
    train_csr, val_csr, _ = split_train_val_test_users(
        interactions_csr, pretrained_model_cfg["val_user_ratio"], pretrained_model_cfg["test_user_ratio"], pretrained_model_cfg["seed"]
    )
    interaction_dataloader = Dataloader(train_csr, pretrained_model_cfg["batch_size"], device, pretrained_model_cfg["seed"])

    pretrained_model_class = getattr(importlib.import_module(pretrained_model_cfg["model_module"]), pretrained_model_cfg["model_class"])
    pretrained_model = pretrained_model_class(train_csr.shape[1], pretrained_model_cfg["embedding_dim"], pretrained_model_cfg["seed"]).to(device)
    load_checkpoint(pretrained_model, None, pretrained_model_checkpoint, device)
    user_embeddings = np.vstack(
        [pretrained_model.encode(batch).detach().cpu().numpy() for batch in tqdm(interaction_dataloader, desc="Computing user embeddings from interactions")]
    )
    print(f"User embeddings shape={user_embeddings.shape}")

    dataloader = Dataloader(user_embeddings, cfg["batch_size"], device, cfg["seed"])
    model_class = getattr(importlib.import_module(cfg["model_module"]), cfg["model_class"])
    extra_params = {k: cfg[k] for k in cfg.keys() if k in SAE_EXTRA_PARAMS}
    model = model_class(user_embeddings.shape[1], cfg["embedding_dim"], cfg["seed"], **extra_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))

    checkpoint_path = get_checkpoint_filepath(cfg)
    try:
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device, cfg)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0

    best_result = 0.0
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            losses = model.train_step(optimizer, batch)
            pbar.set_description(f"Epoch {epoch + 1}/{cfg['epochs']}", refresh=False)
            pbar.set_postfix(losses)
            # if i == len(pbar) - 1:
            #     model.eval()
            #     val_inputs, val_targets = split_input_target_interactions(val_csr, cfg["target_interaction_ratio"], cfg["seed"])
            #     eval_results = evaluate_recall_at_k(model, val_inputs, val_targets, cfg["eval_topk"], cfg["batch_size"], device)
            #     pbar.set_postfix_str(
            #         pbar.postfix + f", Recall@{cfg['eval_topk']}={np.mean(eval_results):.4f}+-{np.std(eval_results) / np.sqrt(len(eval_results)):.4f}"
            #     )
        # if best_result < np.mean(eval_results):
        #     best_result = np.mean(eval_results)
        #     save_checkpoint(model, optimizer, epoch + 1, cfg, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for SAE training script.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--pretrained_model_checkpoint", type=str, required=True, help="Filename of checkpoint containing pre-trained model")
    parser.add_argument("--model_module", type=str, default="sae", help="Module containing SAE model")
    parser.add_argument("--model_class", type=str, default="BasicSAE", help="Model class name")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension of SAE model")
    parser.add_argument("--l1_coef", type=float, default=0.01, help="L1 loss coefficient (BasicSAE)")
    parser.add_argument("--k", type=int, default=32, help="Top K parameter (TopKSAE, BatchTopKSAE)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta_1 coefficient")
    parser.add_argument("--beta2", type=float, default=0.99, help="Adam beta_2 coefficient")
    parser.add_argument("--seed", type=float, default=42, help="Random seed")
    cfg = vars(parser.parse_args())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
    train_sae(cfg, device)
