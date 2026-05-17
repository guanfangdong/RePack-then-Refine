import torch
from torch import nn

from repack.util import instantiate_from_config


class WeightedLossCollection(nn.Module):
    def __init__(self, **loss_configs):
        super().__init__()
        if not loss_configs:
            raise ValueError("At least one loss component is required.")
        self.losses = nn.ModuleDict(
            {name: instantiate_from_config(cfg) for name, cfg in loss_configs.items()}
        )
        print(f"Stage-1 losses: {list(self.losses.keys())}")

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step, **kwargs):
        split = kwargs.get("split", "train")
        total_loss = None
        logs = {}

        for name, loss_module in self.losses.items():
            if optimizer_idx != 0 and name != "lpips_disc_loss":
                continue
            loss_value, loss_logs = loss_module(
                inputs,
                reconstructions,
                posteriors,
                optimizer_idx,
                global_step,
                **kwargs,
            )
            total_loss = loss_value if total_loss is None else total_loss + loss_value
            logs[f"{split}/{name}"] = loss_value.detach().mean()
            for key, value in loss_logs.items():
                logs[key] = value

        if total_loss is None:
            total_loss = torch.zeros((), device=reconstructions.device)
        return total_loss, logs

