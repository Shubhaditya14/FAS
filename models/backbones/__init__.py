"""Backbone architectures for FAS system."""

from typing import Optional

import timm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def get_backbone(
    name: str = "efficientnet-b0",
    pretrained: bool = True,
    freeze_layers: int = 0,
) -> nn.Module:
    """Get backbone model.

    Args:
        name: Backbone architecture name
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        freeze_layers: Number of initial layers to freeze

    Returns:
        Backbone model
    """
    if "efficientnet" in name.lower():
        if pretrained:
            model = EfficientNet.from_pretrained(name, num_classes=num_classes)
        else:
            model = EfficientNet.from_name(name, num_classes=num_classes)
    else:
        # Use timm for other architectures
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    # Freeze layers if specified
    if freeze_layers > 0:
        count = 0
        for param in model.parameters():
            if count < freeze_layers:
                param.requires_grad = False
                count += 1
            else:
                break

    return model


__all__ = ["get_backbone"]
__all__ = ['get_backbone']
