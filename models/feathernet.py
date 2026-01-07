"""FeatherNet architecture for face anti-spoofing.

This implementation matches the pretrained checkpoint structure exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Module):
    """Convolution + BatchNorm + PReLU block."""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class DepthwiseConvBN(nn.Module):
    """Depthwise convolution + BatchNorm (no activation) - for conv_6_dw."""

    def __init__(self, channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            padding,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ProjectConv(nn.Module):
    """Project convolution with correct naming for checkpoint."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class SEModule(nn.Module):
    """SE module matching checkpoint naming: se_module.fc1, se_module.bn1, etc."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = channels // reduction
        self.fc1 = nn.Conv2d(channels, reduced, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.fc2 = nn.Conv2d(reduced, channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.prelu(self.bn1(self.fc1(se)), torch.ones(1, device=se.device) * 0.25)
        se = torch.sigmoid(self.bn2(self.fc2(se)))
        return x * se


class InvertedResidualBlock(nn.Module):
    """Inverted residual block matching checkpoint structure.

    Structure: conv (1x1 expand) -> conv_dw (3x3 depthwise) -> project (1x1 reduce)
    Optional SE module after project.
    """

    def __init__(
        self, in_channels, out_channels, expand_channels, stride=1, use_se=False
    ):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.use_se = use_se

        # Expansion 1x1 conv
        self.conv = ConvBNPReLU(in_channels, expand_channels, kernel_size=1, padding=0)

        # Depthwise 3x3 conv
        self.conv_dw = ConvBNPReLU(
            expand_channels,
            expand_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=expand_channels,
        )

        # Projection 1x1 conv (no activation)
        self.project = ProjectConv(expand_channels, out_channels)

        # SE module with correct naming
        if use_se:
            self.se_module = SEModule(out_channels, reduction=4)

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.project(out)

        if self.use_se:
            out = self.se_module(out)

        if self.use_residual:
            out = out + identity

        return out


class TransitionBlock(nn.Module):
    """Transition block between stages.

    Structure: conv (1x1 expand) -> conv_dw (3x3 depthwise stride 2) -> project (1x1 reduce)
    """

    def __init__(self, in_channels, out_channels, expand_channels, stride=2):
        super().__init__()
        # Expansion 1x1 conv
        self.conv = ConvBNPReLU(in_channels, expand_channels, kernel_size=1, padding=0)

        # Depthwise 3x3 conv with stride
        self.conv_dw = ConvBNPReLU(
            expand_channels,
            expand_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=expand_channels,
        )

        # Projection 1x1 conv
        self.project = ProjectConv(expand_channels, out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv_dw(out)
        out = self.project(out)
        return out


class Stage(nn.Module):
    """A stage containing multiple inverted residual blocks."""

    def __init__(self, blocks):
        super().__init__()
        self.model = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.model:
            x = block(x)
        return x


class FTGenerator(nn.Module):
    """Feature Transform Generator - convolutional version matching checkpoint."""

    def __init__(self):
        super().__init__()
        # Conv layers: 128->128->64->1 (with BatchNorm, no explicit activation in checkpoint)
        self.ft = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        return self.ft(x)


class FeatherNetBackbone(nn.Module):
    """FeatherNet backbone matching the pretrained checkpoint exactly."""

    def __init__(self, num_classes=2):
        super().__init__()

        # Initial conv: 3 -> 32, stride 2
        self.conv1 = ConvBNPReLU(3, 32, kernel_size=3, stride=2, padding=1)

        # Depthwise conv: 32 -> 32
        self.conv2_dw = ConvBNPReLU(
            32, 32, kernel_size=3, stride=1, padding=1, groups=32
        )

        # Transition 2->3: 32 -> 64, stride 2
        self.conv_23 = TransitionBlock(32, 64, expand_channels=103, stride=2)

        # Stage 3: 4 blocks, last with SE (64 -> 64)
        self.conv_3 = Stage(
            [
                InvertedResidualBlock(
                    64, 64, expand_channels=13, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    64, 64, expand_channels=13, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    64, 64, expand_channels=13, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    64, 64, expand_channels=13, stride=1, use_se=True
                ),
            ]
        )

        # Transition 3->4: 64 -> 128, stride 2
        self.conv_34 = TransitionBlock(64, 128, expand_channels=231, stride=2)

        # Stage 4: 6 blocks, last with SE (128 -> 128)
        self.conv_4 = Stage(
            [
                InvertedResidualBlock(
                    128, 128, expand_channels=231, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    128, 128, expand_channels=52, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    128, 128, expand_channels=26, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    128, 128, expand_channels=77, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    128, 128, expand_channels=26, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    128, 128, expand_channels=26, stride=1, use_se=True
                ),
            ]
        )

        # Transition 4->5: 128 -> 128, stride 2
        self.conv_45 = TransitionBlock(128, 128, expand_channels=308, stride=2)

        # Stage 5: 2 blocks, last with SE (128 -> 128)
        self.conv_5 = Stage(
            [
                InvertedResidualBlock(
                    128, 128, expand_channels=26, stride=1, use_se=False
                ),
                InvertedResidualBlock(
                    128, 128, expand_channels=26, stride=1, use_se=True
                ),
            ]
        )

        # Final layers
        self.conv_6_sep = ConvBNPReLU(128, 512, kernel_size=1, padding=0)
        # conv_6_dw has NO PReLU in checkpoint - just conv + bn
        self.conv_6_dw = DepthwiseConvBN(512, kernel_size=8, stride=1, padding=0)

        # Fully connected layers
        self.linear = nn.Linear(512, 128, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.prob = nn.Linear(128, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = self.linear(x)
        x = self.bn(x)
        x = self.prob(x)

        return x


class FeatherNetB(nn.Module):
    """FeatherNet-B for face anti-spoofing with pretrained weight loading."""

    def __init__(self, num_classes=2, input_size=128):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Main model backbone
        self.model = FeatherNetBackbone(num_classes=num_classes)

        # FT Generator for auxiliary supervision (optional during inference)
        self.FTGenerator = FTGenerator()

    def forward(self, x, return_ft=False):
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)
            return_ft: If True, also return FT map

        Returns:
            If return_ft: (logits, ft_map)
            Else: spoof probability (after sigmoid on spoof class)
        """
        # Get intermediate features for FT generator (before final pooling)
        feat = self.model.conv1(x)
        feat = self.model.conv2_dw(feat)
        feat = self.model.conv_23(feat)
        feat = self.model.conv_3(feat)
        feat = self.model.conv_34(feat)
        feat = self.model.conv_4(feat)
        feat = self.model.conv_45(feat)
        feat = self.model.conv_5(feat)

        # FT generator operates on conv_5 output
        if return_ft:
            ft_map = self.FTGenerator(feat)

        # Continue through final layers
        x = self.model.conv_6_sep(feat)
        x = self.model.conv_6_dw(x)
        x = x.view(x.size(0), -1)
        x = self.model.linear(x)
        x = self.model.bn(x)
        logits = self.model.prob(x)

        if return_ft:
            return logits, ft_map

        # Return spoof probability using softmax over 2 classes
        # logits: [class_0_logit, class_1_logit]
        # class_0 = Spoof, class_1 = Real
        probs = torch.softmax(logits, dim=1)
        spoof_prob = probs[:, 0]  # Probability of class 0 (spoof)
        return spoof_prob.unsqueeze(1)  # Return (B, 1) tensor

    def predict(self, x):
        """Get spoof probability for input.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Spoof probability (B, 1)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def extract_features(self, x):
        """Extract feature embeddings.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Feature embeddings (B, 128)
        """
        self.eval()
        with torch.no_grad():
            # Extract features through backbone
            feat = self.model.conv1(x)
            feat = self.model.conv2_dw(feat)
            feat = self.model.conv_23(feat)
            feat = self.model.conv_3(feat)
            feat = self.model.conv_34(feat)
            feat = self.model.conv_4(feat)
            feat = self.model.conv_45(feat)
            feat = self.model.conv_5(feat)
            feat = self.model.conv_6_sep(feat)
            feat = self.model.conv_6_dw(feat)
            feat = feat.view(feat.size(0), -1)
            feat = self.model.linear(feat)
            feat = self.model.bn(feat)
            return feat

    def load_pretrained(self, checkpoint_path, device="cpu"):
        """Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to .pth file
            device: Device to load on
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix from DataParallel (only from START, preserve 'se_module')
        new_state_dict = {}
        for k, v in state_dict.items():
            # Use slicing to only remove prefix, not replace all occurrences
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        # Load weights
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        if len(missing) > 0:
            print(f"Warning: {len(missing)} missing keys")
            if len(missing) <= 10:
                for key in missing:
                    print(f"  - {key}")

        if len(unexpected) > 0:
            print(f"Warning: {len(unexpected)} unexpected keys")
            if len(unexpected) <= 10:
                for key in unexpected:
                    print(f"  - {key}")

        if len(missing) == 0 and len(unexpected) == 0:
            print(f"Successfully loaded all weights from {checkpoint_path}")
        else:
            print(
                f"Loaded pretrained weights from {checkpoint_path} (with some mismatches)"
            )


def create_feathernet(
    num_classes=2, input_size=128, pretrained_path=None, device="cpu"
):
    """Factory function to create FeatherNet model.

    Args:
        num_classes: Number of output classes
        input_size: Input image size
        pretrained_path: Path to pretrained weights
        device: Device to load on

    Returns:
        FeatherNet model
    """
    model = FeatherNetB(num_classes=num_classes, input_size=input_size)

    if pretrained_path:
        model.load_pretrained(pretrained_path, device=device)

    return model.to(device)
