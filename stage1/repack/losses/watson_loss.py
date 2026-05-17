import sys
from pathlib import Path

import torch
import torch.nn as nn

_WATSON_SRC = Path(__file__).with_name("watson_src")
_ROBUST_LOSS_SRC = _WATSON_SRC / "robust_loss_pytorch"
for path in (_ROBUST_LOSS_SRC, _WATSON_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loss_provider import LossProvider  # noqa: E402


class WatsonLoss(nn.Module):
    def __init__(
        self,
        type="Watson-DFT",
        loss_weight=1.0,
        deterministic=True,
        pretrained=False,
    ):
        super().__init__()
        if loss_weight < 0:
            raise ValueError("loss_weight must be non-negative.")

        self.loss_weight = loss_weight
        provider = LossProvider()
        self.watson_loss_fn = provider.get_loss_function(
            type,
            colorspace="RGB",
            pretrained=pretrained,
            reduction="sum",
            deterministic=deterministic,
        )
        self._loss_device = None

    def forward(self, pred, target, *args, **kwargs):
        if self.loss_weight == 0:
            return torch.zeros((), device=pred.device), {}
        if self._loss_device != pred.device:
            self.watson_loss_fn = self.watson_loss_fn.to(pred.device)
            self._loss_device = pred.device

        pred_01 = (pred + 1.0) / 2.0
        target_01 = (target + 1.0) / 2.0
        loss = self.watson_loss_fn(pred_01, target_01) / pred.numel()
        return loss * self.loss_weight, {}
