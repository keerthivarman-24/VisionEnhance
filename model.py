import torch
import torch.nn as nn
import torch.nn.functional as F


def icnr_init(conv_weight, upscale_factor, initializer=nn.init.kaiming_normal_):
    out_channels, in_channels, kh, kw = conv_weight.shape
    r2 = upscale_factor ** 2
    sub_out = out_channels // r2
    sub_kernel = torch.zeros(sub_out, in_channels, kh, kw)
    initializer(sub_kernel)
    kernel = sub_kernel.repeat_interleave(r2, dim=0)
    conv_weight.data.copy_(kernel)


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):

    def __init__(self, channels, expand_ratio=2, ffn_expand_ratio=2):
        super().__init__()
        dw_channels = channels * expand_ratio

        # spatial mixing
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1,
                                 groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # channel mixing (FFN)
        self.norm2 = LayerNorm2d(channels)
        ffn_channels = channels * ffn_expand_ratio
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.gamma1

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * self.gamma2

        return x


class RestorationNet(nn.Module):

    def __init__(self, in_channels=1, width=32, num_blocks=14, refine_blocks=2,
                 upscale=2):
        super().__init__()
        self.upscale = upscale

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])

        self.pre_upsample = nn.Conv2d(width, width * (upscale ** 2), kernel_size=3,
                                       padding=1)
        icnr_init(self.pre_upsample.weight, upscale)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.refine = nn.Sequential(*[NAFBlock(width) for _ in range(refine_blocks)])
        self.out_conv = nn.Conv2d(width, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale, mode="bicubic",
                              align_corners=False)

        feat = self.intro(x)
        feat = self.body(feat)
        feat = self.pre_upsample(feat)
        feat = self.pixel_shuffle(feat)
        feat = self.refine(feat)
        out = self.out_conv(feat)

        return out + base


if __name__ == "__main__":
    model = RestorationNet(width=32, num_blocks=14)
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params / 1e6:.2f}M")