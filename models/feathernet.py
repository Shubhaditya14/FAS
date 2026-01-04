"""FeatherNet architecture for face anti-spoofing."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolution block with Conv + BN + PReLU."""

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


class DepthwiseConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.prelu_dw = nn.PReLU(in_channels)
        self.prelu_pw = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.prelu_dw(self.bn_dw(self.depthwise(x)))
        x = self.prelu_pw(self.bn_pw(self.pointwise(x)))
        return x


class SEModule(nn.Module):
    """Squeeze-and-Excitation module for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        # Scale
        return x * y


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2-style bottleneck)."""

    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se=False):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(ConvBlock(in_channels, hidden_dim, kernel_size=1, padding=0))

        # Depthwise
        layers.append(
            ConvBlock(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
            )
        )

        # SE module
        if use_se:
            layers.append(SEModule(hidden_dim))

        # Projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FTGenerator(nn.Module):
    """Feature Transform Generator network."""

    def __init__(self, input_dim=512, hidden_dims=[512, 128, 512]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim

        self.generator = nn.Sequential(*layers)

    def forward(self, x):
        return self.generator(x)


class FeatherNetB(nn.Module):
    """FeatherNet-B architecture for face anti-spoofing."""

    def __init__(self, num_classes=2, input_size=128, embedding_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Initial convolution
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=2, padding=1)

        # Inverted residual blocks
        # Stage 1 (conv_23)
        self.conv_23 = nn.Sequential(
            InvertedResidual(32, 32, stride=1, expand_ratio=1, use_se=False),
            InvertedResidual(32, 32, stride=2, expand_ratio=2, use_se=False),
        )

        # Stage 2 (conv_3)
        self.conv_3 = nn.Sequential(
            InvertedResidual(32, 48, stride=1, expand_ratio=2, use_se=False),
            InvertedResidual(48, 48, stride=2, expand_ratio=2, use_se=False),
        )

        # Stage 3 (conv_4)
        self.conv_4 = nn.Sequential(
            InvertedResidual(48, 96, stride=1, expand_ratio=2, use_se=True),
            InvertedResidual(96, 96, stride=1, expand_ratio=2, use_se=True),
            InvertedResidual(96, 96, stride=2, expand_ratio=2, use_se=True),
        )

        # Stage 4 (conv_5)
        self.conv_5 = nn.Sequential(
            InvertedResidual(96, 128, stride=1, expand_ratio=2, use_se=True),
            InvertedResidual(128, 128, stride=1, expand_ratio=2, use_se=True),
        )

        # Final depthwise and pooling (conv_6_dw)
        self.conv_6_dw = nn.Sequential(
            ConvBlock(128, 512, kernel_size=1, padding=0), nn.AdaptiveAvgPool2d(1)
        )

        # Feature transform generator (optional)
        self.ft_generator = FTGenerator(input_dim=512)

        # Classification head
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.PReLU()
        )

        self.prob = nn.Linear(embedding_dim, 1)  # Binary classification

    def forward(self, x, return_features=False):
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)
            return_features: If True, return features before classification

        Returns:
            Output tensor or (output, features) tuple
        """
        # Feature extraction
        x = self.conv1(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6_dw(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Feature transform (optional)
        ft_features = self.ft_generator(x)

        # Embedding
        features = self.embedding(ft_features)

        # Classification
        output = self.prob(features)
        output = torch.sigmoid(output)

        if return_features:
            return output, features
        return output

    def extract_features(self, x):
        """Extract feature embeddings.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            Feature embeddings (B, embedding_dim)
        """
        _, features = self.forward(x, return_features=True)
        return features

    def load_pretrained(self, checkpoint_path, device="cpu"):
        """Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to .pth file
            device: Device to load on
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

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

        # Remove 'module.' prefix from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        # Load weights
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights from {checkpoint_path}")


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
