import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.out_layers(self.in_layers(x)) + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(batch, 3, channels, height * width).transpose(2, 3)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q @ k.transpose(-2, -1)) * (channels**-0.5)
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).view(batch, channels, height, width)
        return x + self.proj(h)


class RefineUNet(nn.Module):
    """Latent-guided image refinement U-Net."""

    def __init__(self, in_channels=35, out_channels=3, base_channels=128):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.tanh = nn.Tanh()

        self.down = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "block": nn.Sequential(
                            ResBlock(base_channels, base_channels),
                            ResBlock(base_channels, base_channels),
                        ),
                        "downsample": nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1),
                    }
                ),
                nn.ModuleDict(
                    {
                        "block": nn.Sequential(
                            ResBlock(base_channels, base_channels * 2),
                            ResBlock(base_channels * 2, base_channels * 2),
                        ),
                        "downsample": nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),
                    }
                ),
                nn.ModuleDict(
                    {
                        "block": nn.Sequential(
                            ResBlock(base_channels * 2, base_channels * 4),
                            ResBlock(base_channels * 4, base_channels * 4),
                            AttentionBlock(base_channels * 4),
                        ),
                        "downsample": nn.Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),
                    }
                ),
            ]
        )

        self.mid = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 8),
            AttentionBlock(base_channels * 8),
            ResBlock(base_channels * 8, base_channels * 4),
        )

        self.up = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "upsample": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        "block": nn.Sequential(
                            ResBlock(base_channels * 8, base_channels * 4),
                            AttentionBlock(base_channels * 4),
                        ),
                    }
                ),
                nn.ModuleDict(
                    {
                        "upsample": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        "block": nn.Sequential(
                            ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2),
                            ResBlock(base_channels * 2, base_channels * 2),
                        ),
                    }
                ),
                nn.ModuleDict(
                    {
                        "upsample": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        "block": nn.Sequential(
                            ResBlock(base_channels * 2 + base_channels, base_channels),
                            ResBlock(base_channels, base_channels),
                        ),
                    }
                ),
            ]
        )

        self.final_res = ResBlock(base_channels + base_channels, base_channels)
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, z):
        z_up = F.interpolate(z, size=x.shape[2:], mode="bilinear", align_corners=False)
        h = self.conv_in(torch.cat([x, z_up], dim=1))

        skips = [h]
        for layer in self.down:
            h = layer["block"](h)
            skips.append(h)
            h = layer["downsample"](h)

        h = self.mid(h)

        for layer in self.up:
            h = layer["upsample"](h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = layer["block"](h)

        h = torch.cat([h, skips.pop()], dim=1)
        h = self.final_res(h)
        return self.tanh(self.conv_out(F.silu(self.norm_out(h))))


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for local texture realism."""

    def __init__(self, in_channels=3, base_channels=128, n_layers=3):
        super().__init__()
        kernel_size = 4
        padding = 1

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for layer_idx in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**layer_idx, 8)
            layers += [
                nn.Conv2d(
                    base_channels * nf_mult_prev,
                    base_channels * nf_mult,
                    kernel_size,
                    stride=2,
                    padding=padding,
                ),
                nn.BatchNorm2d(base_channels * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(
                base_channels * nf_mult_prev,
                base_channels * nf_mult,
                kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.BatchNorm2d(base_channels * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_channels * nf_mult, 1, kernel_size, stride=1, padding=padding),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    unet = RefineUNet()
    disc = PatchDiscriminator()
    print(f"RefineUNet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f}M")
    print(f"PatchDiscriminator params: {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M")
