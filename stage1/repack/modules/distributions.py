import numpy as np
import torch


class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self, other=None, no_sum=False):
        if self.deterministic:
            return torch.zeros((), device=self.parameters.device)
        if other is None:
            kl = self.mean.pow(2) + self.var - 1.0 - self.logvar
        else:
            kl = (
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar
            )
        kl = 0.5 * kl
        return kl if no_sum else torch.sum(kl, dim=[1, 2, 3])

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.zeros((), device=self.parameters.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + (sample - self.mean).pow(2) / self.var,
            dim=dims,
        )


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = next(x for x in (mean1, logvar1, mean2, logvar2) if isinstance(x, torch.Tensor))
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x, device=tensor.device)
        for x in (logvar1, logvar2)
    ]
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + (mean1 - mean2).pow(2) * torch.exp(-logvar2)
    )

