"""Composite objective of SGCA: consistency + separation + SSIM reconstruction."""
import torch
from pytorch_msssim import ssim


class SGCALoss(torch.nn.Module):
    """L = w_con * L_con + w_sep * L_sep + L_rec.

    * ``L_con``: prior-weighted L2 feature distance on (likely) unchanged pixels, weighted by (1 - M).
    * ``L_sep``: squared hinge ``max(0, m - D)^2`` on (likely) changed pixels, weighted by M.
    * ``L_rec``: ``w_rec * (1 - SSIM(x_hat_t, x_t))`` self-reconstruction for each branch plus
      ``w_cross * SSIM(x_hat_t, x_s)`` between each reconstruction and the *other* acquisition,
      which keeps every decoder modality-specific.
    """

    def __init__(self, margin: float = 1.0, w_con: float = 1.0, w_sep: float = 1.0,
                 w_rec: float = 1.0, w_cross: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.margin, self.eps = margin, eps
        self.w_con, self.w_sep, self.w_rec, self.w_cross = w_con, w_sep, w_rec, w_cross

    def forward(self, f1, f2, recon1, recon2, x1, x2, prior):
        dist = torch.norm(f1 - f2, p=2, dim=1)
        m = prior.squeeze(1)

        w_unchg = 1 - m
        l_con = (w_unchg * dist).sum() / (w_unchg.sum() + self.eps)
        l_sep = (m * torch.clamp(self.margin - dist, min=0) ** 2).sum() / (m.sum() + self.eps)

        l_self = self.w_rec * ((1 - ssim(recon1, x1, data_range=1)) + (1 - ssim(recon2, x2, data_range=1)))
        l_cross = self.w_cross * (ssim(recon1, x2, data_range=1) + ssim(recon2, x1, data_range=1))
        l_rec = l_self + l_cross

        total = self.w_con * l_con + self.w_sep * l_sep + l_rec
        return total, {"con": l_con.item(), "sep": l_sep.item(), "rec": l_rec.item(), "total": total.item()}
