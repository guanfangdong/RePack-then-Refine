import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        alpha=1.0,
        patch_factor=1,
        ave_spectrum=False,
        log_matrix=False,
        batch_matrix=False,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        _, _, height, width = x.shape
        if height % self.patch_factor != 0 or width % self.patch_factor != 0:
            raise ValueError("patch_factor must divide image height and width.")

        patch_h = height // self.patch_factor
        patch_w = width // self.patch_factor
        patches = []
        for i in range(self.patch_factor):
            for j in range(self.patch_factor):
                patches.append(x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w])

        freq = torch.fft.fft2(torch.stack(patches, 1), norm="ortho")
        return torch.stack([freq.real, freq.imag], dim=-1)

    def loss_formulation(self, pred_freq, target_freq, matrix=None):
        if matrix is None:
            matrix = (pred_freq - target_freq).pow(2)
            matrix = torch.sqrt(matrix[..., 0] + matrix[..., 1]).pow(self.alpha)
            if self.log_matrix:
                matrix = torch.log(matrix + 1.0)
            if self.batch_matrix:
                matrix = matrix / matrix.max().clamp_min(1e-8)
            else:
                denom = matrix.max(-1).values.max(-1).values[:, :, :, None, None].clamp_min(1e-8)
                matrix = matrix / denom
            matrix = torch.clamp(torch.nan_to_num(matrix), min=0.0, max=1.0)

        distance = (pred_freq - target_freq).pow(2)
        distance = distance[..., 0] + distance[..., 1]
        return torch.mean(matrix.detach() * distance)

    def forward(self, pred, target, *args, **kwargs):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq) * self.loss_weight, {}

