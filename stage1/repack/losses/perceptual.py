import torch
import torch.nn as nn

try:
    from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
    from taming.modules.losses.lpips import LPIPS
    from taming.modules.losses.vqperceptual import adopt_weight, hinge_d_loss, vanilla_d_loss

    _TAMING_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the training environment
    NLayerDiscriminator = None
    LPIPS = None
    _TAMING_IMPORT_ERROR = exc

    def adopt_weight(weight, global_step, threshold=0, value=0.0):
        return weight if global_step >= threshold else value

    def hinge_d_loss(logits_real, logits_fake):
        loss_real = torch.mean(torch.relu(1.0 - logits_real))
        loss_fake = torch.mean(torch.relu(1.0 + logits_fake))
        return 0.5 * (loss_real + loss_fake)

    def vanilla_d_loss(logits_real, logits_fake):
        return 0.5 * (
            torch.mean(torch.nn.functional.softplus(-logits_real))
            + torch.mean(torch.nn.functional.softplus(logits_fake))
        )


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):
        super().__init__()
        if _TAMING_IMPORT_ERROR is not None:
            raise ImportError(
                "LPIPSWithDiscriminator requires taming-transformers. "
                "Install it before training stage1."
            ) from _TAMING_IMPORT_ERROR
        if disc_loss not in ["hinge", "vanilla"]:
            raise ValueError("disc_loss must be 'hinge' or 'vanilla'.")

        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(()) * logvar_init)

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
        **kwargs,
    ):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss if weights is None else weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = torch.sum(posteriors.kl()) / inputs.shape[0]

        if optimizer_idx == 0:
            if cond is None:
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0 and last_layer is not None:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer)
                except RuntimeError:
                    d_weight = torch.zeros((), device=inputs.device)
            else:
                d_weight = torch.zeros((), device=inputs.device)

            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
            )
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            return loss, {
                f"{split}/total_loss": loss.detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.as_tensor(disc_factor, device=inputs.device),
                f"{split}/g_loss": g_loss.detach().mean(),
            }

        if cond is None:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        else:
            logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
            logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

        disc_factor = adopt_weight(
            self.disc_factor,
            global_step,
            threshold=self.discriminator_iter_start,
        )
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
        return d_loss, {
            f"{split}/disc_loss": d_loss.detach().mean(),
            f"{split}/logits_real": logits_real.detach().mean(),
            f"{split}/logits_fake": logits_fake.detach().mean(),
        }

