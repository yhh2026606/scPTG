from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .community import cluster_centers, multiscale_communities
from .graph import Graph, build_graph
from .model import AnchoredGNN, DenoisingAutoencoder, PrototypeHead, guarded_readout, prototype_separation


@dataclass
class Result:
    predictions: np.ndarray
    embedding: np.ndarray
    probabilities: np.ndarray
    community_labels: np.ndarray
    history: list[dict[str, float | int | str]]
    graph: Graph
    trajectory_trust: float
    trajectory_agreement: float


def train(x: np.ndarray, pca: np.ndarray, n_clusters: int, config: dict, device: torch.device, seed: int, logger: logging.Logger) -> Result:
    model_cfg, train_cfg, graph_cfg = config["model"], config["train"], config["graph"]
    autoencoder = DenoisingAutoencoder(
        x.shape[1], list(model_cfg["hidden_dims"]), int(model_cfg["latent_dim"]), float(model_cfg["dropout"])
    ).to(device)
    history = _pretrain(autoencoder, x, config, device, seed, logger)
    latent = _encode(autoencoder, x, device, int(train_cfg["encode_batch_size"]))
    fused = _fuse_numpy(latent, pca, int(model_cfg["pca_residual_dim"]), float(model_cfg["pca_residual_weight"]))

    graph = build_graph(pca, int(graph_cfg["k"]), float(graph_cfg["rank_sigma"]), float(graph_cfg["self_loop"])).to(device)
    community = multiscale_communities(
        pca, graph.edges, graph.weights, n_clusters, list(model_cfg["community_resolutions"]), seed
    )
    logger.info("Graph: %s", graph.metadata)
    logger.info("Community initialization: %d clusters", np.unique(community).size)

    gnn = AnchoredGNN(
        fused.shape[1], int(model_cfg["gnn_num_layers"]),
        float(model_cfg["gnn_init_alpha"]), float(model_cfg["gnn_init_beta"]),
    ).to(device)
    head = PrototypeHead(n_clusters, fused.shape[1], float(model_cfg["temperature"])).to(device)
    head.initialize(torch.from_numpy(cluster_centers(fused, community, n_clusters)).to(device))

    encoder_parameters = list(autoencoder.encoder.parameters())
    cluster_parameters = list(gnn.parameters()) + list(head.parameters())
    joint_lr = float(train_cfg["joint_lr"])
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": joint_lr * float(train_cfg["encoder_lr_scale"])},
        {"params": cluster_parameters, "lr": joint_lr},
    ], weight_decay=float(train_cfg["weight_decay"]))
    x_tensor = torch.from_numpy(x).to(device)
    pca_tensor = torch.from_numpy(pca).to(device)
    anchor_target = F.one_hot(torch.from_numpy(community).to(device), n_clusters).float()
    joint_epochs = int(train_cfg["joint_epochs"])

    for epoch in range(joint_epochs):
        frozen = epoch < int(train_cfg["freeze_encoder_epochs"])
        for parameter in autoencoder.encoder.parameters():
            parameter.requires_grad_(not frozen)
        autoencoder.train(); gnn.train(); head.train()
        optimizer.zero_grad(set_to_none=True)
        anchor = _fuse_torch(autoencoder.encode(x_tensor), pca_tensor, int(model_cfg["pca_residual_dim"]), float(model_cfg["pca_residual_weight"]))
        graph_warmup = min(1.0, (epoch + 1) / max(1, int(train_cfg["graph_warmup_epochs"])))
        states = gnn(anchor, graph.propagation, graph_warmup)
        assignments = [head(state) for state in states]
        q = assignments[-1]
        kl = (anchor_target * (torch.log(anchor_target.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(dim=1).mean()
        separation = prototype_separation(head.prototypes, float(train_cfg["sep_margin"]))
        cluster_warmup = min(1.0, (epoch + 1) / max(1, int(train_cfg["cluster_warmup_epochs"])))
        loss = cluster_warmup * (float(train_cfg["lambda_final"]) * kl + float(train_cfg["lambda_sep"]) * separation)
        loss.backward()
        nn.utils.clip_grad_norm_(encoder_parameters + cluster_parameters, float(train_cfg["grad_clip"]))
        optimizer.step()
        record = {"stage": "joint", "epoch": epoch + 1, "loss": float(loss.detach()), "loss_kl": float(kl.detach()), "loss_sep": float(separation.detach())}
        history.append(record)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == joint_epochs:
            logger.info("Joint epoch %d/%d | loss %.6f | KL %.6f", epoch + 1, joint_epochs, record["loss"], record["loss_kl"])

    for parameter in autoencoder.encoder.parameters():
        parameter.requires_grad_(True)
    autoencoder.eval(); gnn.eval(); head.eval()
    with torch.no_grad():
        anchor = _fuse_torch(autoencoder.encode(x_tensor), pca_tensor, int(model_cfg["pca_residual_dim"]), float(model_cfg["pca_residual_weight"]))
        states = gnn(anchor, graph.propagation, 1.0)
        assignments = [head(state) for state in states]
        final_q, trust, agreement = guarded_readout(
            assignments, anchor_target, float(train_cfg["trajectory_trust_center"]), float(train_cfg["trajectory_trust_temperature"])
        )
    return Result(
        final_q.argmax(dim=1).cpu().numpy(), states[-1].cpu().numpy(), final_q.cpu().numpy(), community,
        history, graph, float(trust.cpu()), float(agreement.cpu()),
    )


def _pretrain(model, x, config, device, seed, logger):
    train_cfg, model_cfg = config["train"], config["model"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=int(train_cfg["batch_size"]), shuffle=True,
                        generator=generator, num_workers=int(config["runtime"]["num_workers"]), pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["pretrain_lr"]), weight_decay=float(train_cfg["weight_decay"]))
    best_loss, best_state, patience, history = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    epochs = int(train_cfg["pretrain_epochs"])
    for epoch in range(epochs):
        total = 0.0
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device, non_blocking=True)
            mask = torch.rand(batch.shape, device=device) < float(model_cfg["mask_rate"])
            corrupted = batch.masked_fill(mask, 0.0)
            reconstruction, _ = model(corrupted)
            element = F.huber_loss(reconstruction, batch, delta=float(model_cfg["huber_delta"]), reduction="none")
            masked = (element * mask).sum() / mask.sum().clamp_min(1)
            unmasked = (element * ~mask).sum() / (~mask).sum().clamp_min(1)
            fraction = float(model_cfg["masked_loss_fraction"])
            loss = fraction * masked + (1.0 - fraction) * unmasked
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"])); optimizer.step()
            total += float(loss.detach()) * len(batch)
        epoch_loss = total / len(x)
        history.append({"stage": "pretrain", "epoch": epoch + 1, "loss": epoch_loss})
        if epoch_loss < best_loss - 1e-6:
            best_loss, best_state, patience = epoch_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            logger.info("AE epoch %d/%d | loss %.6f", epoch + 1, epochs, epoch_loss)
        if patience >= int(train_cfg["early_stopping_patience"]):
            logger.info("AE early stopping at epoch %d", epoch + 1)
            break
    model.load_state_dict(best_state)
    return history


def _encode(model, x, device, batch_size):
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            output.append(model.encode(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def _fuse_numpy(latent, pca, pca_dim, pca_weight):
    latent = (latent - latent.mean(axis=0, keepdims=True)) / np.maximum(latent.std(axis=0, keepdims=True), 1e-6)
    return np.ascontiguousarray(np.concatenate([latent, pca_weight * pca[:, :min(pca_dim, pca.shape[1])]], axis=1), dtype=np.float32)


def _fuse_torch(latent, pca, pca_dim, pca_weight):
    latent = (latent - latent.mean(dim=0, keepdim=True)) / latent.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return torch.cat([latent, pca_weight * pca[:, :min(pca_dim, pca.shape[1])]], dim=1)

