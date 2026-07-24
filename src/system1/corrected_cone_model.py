"""Text-and-graph encoder used by the reconstructed corrected-cone experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectedConeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        epsilon: float = 0.1,
        max_radius: float = 0.95,
    ):
        super().__init__()
        if not 0.0 < epsilon < max_radius < 1.0:
            raise ValueError("require 0 < epsilon < max_radius < 1")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.max_radius = max_radius
        self.semantic_proj = nn.Linear(input_dim, output_dim)
        self.structure_proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(output_dim * 2, 1)
        self.radial_head = nn.Linear(output_dim * 2, 1)

    @staticmethod
    def _aggregate(
        node_values: torch.Tensor,
        message_edges: torch.Tensor,
    ) -> torch.Tensor:
        if message_edges.numel() == 0:
            return torch.zeros_like(node_values)
        source, target = message_edges
        output = torch.zeros_like(node_values)
        degree = torch.zeros(
            node_values.shape[0],
            1,
            dtype=node_values.dtype,
            device=node_values.device,
        )
        output.index_add_(0, target, node_values[source])
        degree.index_add_(
            0,
            target,
            torch.ones(
                target.shape[0],
                1,
                dtype=node_values.dtype,
                device=node_values.device,
            ),
        )
        return output / degree.clamp_min(1.0)

    def _map_to_ball(
        self,
        semantic: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([semantic, structural], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        direction_raw = gate * semantic + (1.0 - gate) * structural
        direction = F.normalize(direction_raw, p=2, dim=-1, eps=1e-12)
        radius_fraction = torch.sigmoid(self.radial_head(combined))
        radius = self.epsilon + (
            self.max_radius - self.epsilon
        ) * radius_fraction
        return direction * radius

    def forward(
        self,
        features: torch.Tensor,
        message_edges: torch.Tensor,
    ) -> torch.Tensor:
        semantic = self.semantic_proj(features)
        structural_seed = F.relu(self.structure_proj(features))
        structural = self._aggregate(structural_seed, message_edges)
        return self._map_to_ball(semantic, structural)

    def encode_queries(self, features: torch.Tensor) -> torch.Tensor:
        """Encode proof-state text without graph neighbors."""
        semantic = self.semantic_proj(features)
        structural = torch.zeros_like(semantic)
        return self._map_to_ball(semantic, structural)


class CorrectedConeGoalEncoder:
    """Inference wrapper that maps raw proof-state text into the trained ball."""

    def __init__(
        self,
        checkpoint_path: str,
        sentence_model_path: str,
        device: str = "cuda",
    ):
        from sentence_transformers import SentenceTransformer

        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        self.device = device
        self.c = 1.0
        self.output_dim = int(config["output_dim"])
        self.use_hgcn = True
        self.projector = None
        self.sentence_encoder = SentenceTransformer(
            sentence_model_path, device=device
        )
        self.sentence_encoder.eval()
        self.model = CorrectedConeEncoder(
            input_dim=int(config["input_dim"]),
            output_dim=self.output_dim,
            epsilon=float(config["epsilon"]),
            max_radius=float(config["max_radius"]),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def encode(self, text: str, mode: str = "hyperbolic") -> torch.Tensor:
        del mode
        if not text:
            return torch.zeros(1, self.output_dim, device=self.device)
        with torch.no_grad():
            features = self.sentence_encoder.encode(
                [text],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return self.model.encode_queries(features)
