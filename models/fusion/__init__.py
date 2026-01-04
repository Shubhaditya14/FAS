"""Feature fusion modules for FAS system."""

from typing import List

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Feature fusion module for combining multi-modal features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 2,
        fusion_type: str = "concat",
    ):
        """Initialize feature fusion module.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout probability
            fusion_type: Type of fusion (concat, attention, adaptive)
        """
        super().__init__()

        self.fusion_type = fusion_type

        # Build fusion layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.fusion_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            Class logits
        """
        return self.fusion_net(x)


class AttentionFusion(nn.Module):
    """Attention-based feature fusion."""

    def __init__(self, feature_dim: int, num_modalities: int):
        """Initialize attention fusion.

        Args:
            feature_dim: Dimension of each feature modality
            num_modalities: Number of modalities to fuse
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            features: List of feature tensors from different modalities

        Returns:
            Fused features
        """
        # Stack features
        stacked = torch.stack(features, dim=1)  # [B, num_modalities, D]

        # Calculate attention weights
        attn_weights = self.attention(stacked)  # [B, num_modalities, 1]
        attn_weights = self.softmax(attn_weights)  # [B, num_modalities, 1]

        # Apply attention
        fused = (stacked * attn_weights).sum(dim=1)  # [B, D]

        return fused


__all__ = ["FeatureFusion", "AttentionFusion"]
__all__ = ['FeatureFusion', 'AttentionFusion']
