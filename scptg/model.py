from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout)
        self.decoder = _mlp([latent_dim, *reversed(hidden_dims), input_dim], dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


class AnchoredGNN(nn.Module):
    def __init__(self, dim: int, layers: int, init_alpha: float, init_beta: float) -> None:
        super().__init__()
        gamma = 1.0 - init_alpha - init_beta
        if gamma <= 0:
            raise ValueError("init_alpha + init_beta must be below 1")
        self.transforms = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(layers)])
        for transform in self.transforms:
            nn.init.eye_(transform.weight)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.raw_alpha = nn.Parameter(torch.full((layers,), math.log(init_alpha / gamma)))
        self.raw_beta = nn.Parameter(torch.full((layers,), math.log(init_beta / gamma)))

    def forward(self, anchor: torch.Tensor, propagation: torch.Tensor, warmup: float = 1.0) -> list[torch.Tensor]:
        states = [anchor]
        current = anchor
        for i, (transform, norm) in enumerate(zip(self.transforms, self.norms)):
            logits = torch.stack([self.raw_alpha[i], self.raw_beta[i], anchor.new_zeros(())])
            weights = F.softmax(logits, dim=0)
            alpha = float(max(0.0, min(1.0, warmup))) * weights[0]
            beta = weights[1]
            gamma = 1.0 - alpha - beta
            message = transform(torch.sparse.mm(propagation, current))
            current = norm(gamma * anchor + alpha * message + beta * current)
            states.append(current)
        return states


class PrototypeHead(nn.Module):
    def __init__(self, n_clusters: int, dim: int, temperature: float) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(n_clusters, dim))
        self.temperature = float(temperature)
        nn.init.normal_(self.prototypes, std=0.02)

    def initialize(self, centers: torch.Tensor) -> None:
        with torch.no_grad():
            self.prototypes.copy_(centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = F.normalize(z, dim=1) @ F.normalize(self.prototypes, dim=1).T
        return F.softmax(logits / self.temperature, dim=1)


def guarded_readout(assignments: list[torch.Tensor], anchor: torch.Tensor, center: float, temperature: float):
    anchor_labels = anchor.argmax(dim=1)
    layer_labels = torch.stack([q.argmax(dim=1) for q in assignments], dim=1)
    agreement = (layer_labels == anchor_labels.unsqueeze(1)).float().mean()
    trust = torch.sigmoid((agreement - center) / temperature)
    output = trust * assignments[-1] + (1.0 - trust) * anchor
    return output / output.sum(dim=1, keepdim=True).clamp_min(1e-8), trust, agreement


def prototype_separation(prototypes: torch.Tensor, margin: float) -> torch.Tensor:
    normalized = F.normalize(prototypes, dim=1)
    similarity = normalized @ normalized.T
    mask = ~torch.eye(len(prototypes), dtype=torch.bool, device=prototypes.device)
    return F.relu(similarity[mask] - margin).square().mean()


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

