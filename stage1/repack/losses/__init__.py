from repack.losses.frequency import FocalFrequencyLoss
from repack.losses.perceptual import LPIPSWithDiscriminator
from repack.losses.watson_loss import WatsonLoss
from repack.losses.weighted import WeightedLossCollection

__all__ = [
    "FocalFrequencyLoss",
    "LPIPSWithDiscriminator",
    "WatsonLoss",
    "WeightedLossCollection",
]

