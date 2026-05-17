import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


def nonlinearity(x):
    return x * torch.sigmoid(x)


def normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            if with_conv
            else None
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
                self.nin_shortcut = None
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                self.conv_shortcut = None
        else:
            self.nin_shortcut = None
            self.conv_shortcut = None

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            if self.conv_shortcut is not None:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, height, width = q.shape
        q = q.reshape(b, c, height * width).permute(0, 2, 1)
        k = k.reshape(b, c, height * width)
        attn = torch.bmm(q, k) * (int(c) ** -0.5)
        attn = F.softmax(attn, dim=2)
        v = v.reshape(b, c, height * width)
        h = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, height, width)
        return x + self.proj_out(h)


def make_attn(in_channels, attn_type="vanilla"):
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    if attn_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported attention type: {attn_type}")


class DinoV3RePackEncoder(nn.Module):
    """Frozen DINOv3-B/16 feature extractor plus trainable 1x1 projection."""

    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        **unused,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("DINOv3 RePack encoder expects RGB inputs.")

        self.resolution = resolution
        self.z_channels = z_channels
        self.double_z = double_z
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.dino = AutoModel.from_pretrained(model_name)
        self.dino.eval()
        self.dino.requires_grad_(False)

        self.hidden_size = self.dino.config.hidden_size
        self.num_register_tokens = getattr(self.dino.config, "num_register_tokens", 0)
        patch_size = self.dino.config.patch_size
        self.patch_size = patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size

        out_channels = z_channels * (2 if double_z else 1)
        self.conv_out = nn.Conv2d(self.hidden_size, out_channels, kernel_size=1, bias=False)

        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

    @torch.no_grad()
    def encode_features(self, x):
        _, _, height, width = x.shape
        x = ((x + 1.0) / 2.0 - self.img_mean) / self.img_std
        outputs = self.dino(pixel_values=x)
        tokens = outputs.last_hidden_state[:, 1 + self.num_register_tokens :, :]

        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        expected = h_patches * w_patches
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"DINOv3 returned {tokens.shape[1]} patch tokens, expected {expected}."
            )

        return tokens.reshape(-1, h_patches, w_patches, self.hidden_size).permute(0, 3, 1, 2)

    def forward(self, x):
        with torch.no_grad():
            features = self.encode_features(x)
        return self.conv_out(features)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        attn_type="vanilla",
        **unused,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(f"Decoder latent shape: {self.z_shape}, {np.prod(self.z_shape)} values")

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for level in reversed(range(self.num_resolutions)):
            for block_id, block in enumerate(self.up[level].block):
                h = block(h)
                if block_id < len(self.up[level].attn):
                    h = self.up[level].attn[block_id](h)
            if level != 0:
                h = self.up[level].upsample(h)

        if self.give_pre_end:
            return h
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return torch.tanh(h) if self.tanh_out else h
