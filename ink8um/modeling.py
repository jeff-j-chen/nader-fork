"""ResNet3D-50 encoder + 2D U-Net decoder for ink detection on 8 µm surface volumes.

Standalone re-implementation of the network used in training (``RegressionPLModel``
with ``model_impl="resnet3d_hybrid"``, 2D decoder, input InstanceNorm). Module and
parameter names are identical to the training checkpoints, so the weights load
with ``strict=True``.

Forward pass for a batch of ``(B, 24, 64, 64)`` tiles (depth, height, width):

1. ``InstanceNorm3d(1, affine=True)`` on the single-channel volume.
2. Trilinear upsample to ``(96, 256, 256)``.
3. ResNet3D-50 encoder (Kinetics-style, 1 input channel) -> 4 feature levels.
4. Max-pool each level over depth -> 2D feature maps.
5. 2D U-Net decoder -> ``(B, 1, 64, 64)`` ink logits (back on the tile grid).
"""
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


# ---------------------------------------------------------------------------
# ResNet3D encoder (after kenshohara/3D-ResNets-PyTorch, MIT License)
# ---------------------------------------------------------------------------

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet3DEncoder(nn.Module):
    """ResNet3D that returns the outputs of layer1..layer4."""

    def __init__(self, layers=(3, 4, 6, 3), block_inplanes=(64, 128, 256, 512), n_input_channels=1,
                 conv1_t_size=7, conv1_t_stride=1):
        super().__init__()
        block = Bottleneck
        self.in_planes = block_inplanes[0]
        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0])
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], stride=2)
        self.out_channels = [p * block.expansion for p in block_inplanes]

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        layers += [block(self.in_planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


# ---------------------------------------------------------------------------
# 2D decoder
# ---------------------------------------------------------------------------

class Decoder2D(nn.Module):
    def __init__(self, encoder_dims):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True),
            ) for i in range(1, len(encoder_dims))])
        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)

    def forward(self, feature_maps):
        feature_maps = list(feature_maps)
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            feature_maps[i - 1] = self.convs[i - 1](f)
        return self.logit(feature_maps[0])


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class InkDetector(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pytorch",
    tags=["ink-detection", "vesuvius-challenge", "herculaneum", "computed-tomography", "resnet3d"],
    license="mit",
):
    """Tile-level ink detector.

    Input: ``(B, D, H, W)`` or ``(B, 1, D, H, W)`` float tensor, uint8 intensities clipped to
    [0, 200] and scaled by 1/255. Output: ``(B, 1, H, W)`` ink logits for ``H = W = tile_size``.

    The remaining constructor arguments are inference defaults stored in ``config.json``
    (tile geometry, intensity clipping, and whether the depth axis must be reversed).
    """

    def __init__(
        self,
        in_depth: int = 24,
        tile_size: int = 64,
        input_upsample_hw: int = 256,
        input_upsample_depth: int = 96,
        encoder_depth: int = 50,
        stride: int = 21,
        clip_max: int = 200,
        reverse_layers: bool = False,
    ):
        super().__init__()
        if encoder_depth != 50:
            raise ValueError("Only the ResNet3D-50 encoder is supported")
        self.in_depth = int(in_depth)
        self.tile_size = int(tile_size)
        self.input_upsample_hw = int(input_upsample_hw)
        self.input_upsample_depth = int(input_upsample_depth)
        self.stride = int(stride)
        self.clip_max = int(clip_max)
        self.reverse_layers = bool(reverse_layers)

        self.normalization = nn.InstanceNorm3d(num_features=1, affine=True)
        self.backbone = ResNet3DEncoder()
        self.decoder = Decoder2D(encoder_dims=self.backbone.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, None]
        x = self.normalization(x)
        target = (self.input_upsample_depth or int(x.shape[2]), self.input_upsample_hw, self.input_upsample_hw)
        if self.input_upsample_hw and tuple(x.shape[-3:]) != target:
            x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
        features = self.backbone(x)
        pooled = [torch.max(f, dim=2)[0] for f in features]
        return self.decoder(pooled)
