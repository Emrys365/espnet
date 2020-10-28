from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP


class MaskEstimator(torch.nn.Module):
    def __init__(
        self,
        idim,
        type_="blstmp",
        layers=3,
        units=600,
        projs=600,
        dropout=0.0,
        nmask=6
    ):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int)

        typ = type_.lstrip("vgg").rstrip("p")
        if type_[-1] == "p":
            self.brnn = RNNP(idim, layers, units, projs, subsample, dropout, typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type_
        self.nmask = nmask
        # 6 masks:
        # (1) WPE mask for speaker 1
        # (2) WPE mask for speaker 2
        # (3) Beamforming speech mask for speaker 1
        # (4) Beamforming distortion mask for speaker 1
        # (5) Beamforming speech mask for speaker 2
        # (6) Beamforming distortion mask for speaker 2

        # self.linears = torch.nn.Linear(projs, 1 * nmask)
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(projs, 1) for _ in range(nmask)]
        )

        self.nonlinear = torch.nn.Sigmoid()

    def forward(
        self, xs: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """The forward function

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        _, Fdim, C, input_length = xs.size()
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.permute(0, 2, 3, 1)

        # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        # xs: (B, C, T, F) -> xs: (B * C, T, F)
        xs = xs.contiguous().view(-1, xs.size(-2), xs.size(-1))
        # ilens: (B,) -> ilens_: (B * C)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)

        # xs: (B * C, T, F) -> xs: (B * C, T, D)
        xs, _, _ = self.brnn(xs, ilens_)
        # xs: (B * C, T, D) -> xs: (B, C, T, D)
        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))

        masks = []
        for linear in self.linears:
            # xs: (B, C, T, D) -> mask:(B, C, T, 1)
            mask = linear(xs)

            mask = self.nonlinear(mask)
            # Zero padding
            mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

            # (B, C, T, 1) -> (B, F, C, T)
            #mask = mask.permute(0, 3, 1, 2).expand(-1, Fdim,-1, -1)
            mask = mask.permute(0, 3, 1, 2).repeat(1, Fdim,1, 1)

            # Take care of multi gpu cases: If input_length > max(ilens)
            if mask.size(-1) < input_length:
                mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens
