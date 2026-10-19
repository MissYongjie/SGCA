"""SGCA network: dual-branch encoder-decoder with scribble-guided attention.

The layer layout and tensor operations are kept identical to the implementation
used to produce the results reported in the paper (187,655 trainable parameters
for 3-channel inputs, see Table 6 of the paper).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGaussianSmoothing(nn.Module):
    """Learnable isotropic Gaussian smoothing that turns sparse scribbles into a soft prior.

    The standard deviation is parameterised in log-space and optimised jointly
    with the network, so the spatial extent of the scribble prior adapts to the scene.
    """

    def __init__(self, kernel_size: int = 11, sigma_init: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def forward(self, scribble: torch.Tensor) -> torch.Tensor:
        coord = torch.arange(self.kernel_size, dtype=scribble.dtype, device=scribble.device)
        coord = coord - self.kernel_size // 2
        y, x = torch.meshgrid(coord, coord, indexing="ij")
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(scribble, kernel, padding=self.kernel_size // 2)


class ScribbleGuidedAttention(nn.Module):
    """Attention on a pooled feature grid whose weights are modulated by the scribble prior.

    Features are average-pooled by ``downsample_ratio`` to keep the token count
    tractable on full scenes, attention weights are re-weighted by the
    (resized) scribble weight map, and the output is upsampled back.
    """

    def __init__(self, dim: int, heads: int = 4, downsample_ratio: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.pool = nn.AvgPool2d(downsample_ratio, downsample_ratio)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_down = self.pool(x)
        h_d, w_d = x_down.shape[-2:]
        n = h_d * w_d

        prior = F.interpolate(weight_map, size=(h_d, w_d), mode="bilinear", align_corners=False)

        # NOTE: the plain reshapes below (no permute) follow the original implementation
        # exactly; changing them alters the token layout and the reported numbers.
        q, k, v = [t.reshape(b, self.heads, n, c // self.heads) for t in self.to_qkv(x_down).chunk(3, dim=1)]
        attn = (torch.einsum("bhtc,bhsc->bhts", q, k) * self.scale).softmax(dim=-1)
        attn = attn * prior.view(b, 1, 1, n)

        out = torch.einsum("bhts,bhsc->bhtc", attn, v)
        out = out.reshape(b, h_d, w_d, c).permute(0, 3, 1, 2)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return self.to_out(out)


class SGCABranch(nn.Module):
    """Modality-specific encoder -> scribble-guided attention -> decoder."""

    def __init__(self, in_channels: int, feature_dim: int = 64, heads: int = 4, downsample_ratio: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.ReLU())
        self.attention = ScribbleGuidedAttention(feature_dim, heads, downsample_ratio)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1), nn.ReLU(),
            nn.Conv2d(feature_dim, in_channels, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, weight_map: torch.Tensor):
        feat = self.encoder(x)
        recon = self.decoder(self.attention(feat, weight_map))
        return feat, recon


class SGCA(nn.Module):
    """Scribble-Guided Cross-Attention framework (one branch per acquisition / modality)."""

    def __init__(self, in_channels: int, feature_dim: int = 64, heads: int = 4, downsample_ratio: int = 8,
                 gauss_kernel: int = 11, gauss_sigma: float = 2.0):
        super().__init__()
        self.branch_t1 = SGCABranch(in_channels, feature_dim, heads, downsample_ratio)
        self.branch_t2 = SGCABranch(in_channels, feature_dim, heads, downsample_ratio)
        self.smoothing = AdaptiveGaussianSmoothing(gauss_kernel, gauss_sigma)

    def forward(self, x1, x2, scribble, weight_map):
        f1, recon1 = self.branch_t1(x1, weight_map)
        f2, recon2 = self.branch_t2(x2, weight_map)
        prior = self.smoothing(scribble)
        return f1, f2, recon1, recon2, prior

    @staticmethod
    def distance(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Pixel-wise L2 feature distance D(p) = ||F1(p) - F2(p)||_2, shape [B, H, W]."""
        return torch.norm(f1 - f2, p=2, dim=1)
