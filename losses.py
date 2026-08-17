import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._make_window(window_size, sigma))

    @staticmethod
    def _make_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        window_2d = g.t() @ g
        return window_2d.unsqueeze(0).unsqueeze(0)  # (1,1,size,size)

    def forward(self, pred, target):
        pad = self.window_size // 2
        window = self.window.to(dtype=pred.dtype)

        mu_p = F.conv2d(pred, window, padding=pad)
        mu_t = F.conv2d(target, window, padding=pad)
        mu_p2, mu_t2, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        # E[X^2] - E[X]^2 can round to a tiny negative number even though true
        # variance can't be negative -- clamp defensively so it can't poison the
        # division below (this is what actually produces NaN when it happens).
        sigma_p2 = (F.conv2d(pred * pred, window, padding=pad) - mu_p2).clamp(min=0)
        sigma_t2 = (F.conv2d(target * target, window, padding=pad) - mu_t2).clamp(min=0)
        sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / \
                   ((mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2))
        return 1.0 - ssim_map.mean()


class SobelGradientLoss(nn.Module):

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, pred, target):
        kx = self.kx.to(dtype=pred.dtype)
        ky = self.ky.to(dtype=pred.dtype)

        pred_gx, pred_gy = F.conv2d(pred, kx, padding=1), F.conv2d(pred, ky, padding=1)
        target_gx = F.conv2d(target, kx, padding=1)
        target_gy = F.conv2d(target, ky, padding=1)

        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)
        return F.l1_loss(pred_grad, target_grad)


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_fp32 = pred.float()
        target_fp32 = target.float()
        pred_fft = torch.fft.rfft2(pred_fp32, norm="forward")
        target_fft = torch.fft.rfft2(target_fp32, norm="forward")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


class CompositeLoss(nn.Module):
    def __init__(self, w_charbonnier=1.0, w_ssim=0.2, w_gradient=0.05, w_fft=0.05):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.gradient = SobelGradientLoss()
        self.fft = FFTLoss()
        self.w = dict(charbonnier=w_charbonnier, ssim=w_ssim, gradient=w_gradient,
                      fft=w_fft)

    def forward(self, pred, target):
        l_char = self.charbonnier(pred, target)
        l_ssim = self.ssim(pred, target)
        l_grad = self.gradient(pred, target)
        l_fft = self.fft(pred, target)

        total = (self.w["charbonnier"] * l_char + self.w["ssim"] * l_ssim +
                 self.w["gradient"] * l_grad + self.w["fft"] * l_fft)

        parts = {
            "total": total.item(),
            "charbonnier": l_char.item(),
            "ssim": l_ssim.item(),
            "gradient": l_grad.item(),
            "fft": l_fft.item(),
        }
        return total, parts