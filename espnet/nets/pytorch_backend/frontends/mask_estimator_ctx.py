from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP


class MaskEstimator(torch.nn.Module):
    def __init__(self, type, idim, layers, units, projs, dropout, nmask=1, ctx_dim=None, num_ctx=2):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int)

        projs2 = projs + ctx_dim if ctx_dim is not None else projs
        self.use_ctx = True if ctx_dim is not None else False
        if self.use_ctx:
            print('Using context embedding based mask estimator!', flush=True)
        self.num_ctx = num_ctx

        typ = type.lstrip("vgg").rstrip("p")
        if type[-1] == "p":
            self.brnn = RNNP(idim, layers, units, projs, subsample, dropout, typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type
        self.nmask = nmask
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(projs2, idim) if self.use_ctx and i < num_ctx
                else torch.nn.Linear(projs, idim)
                for i in range(nmask)
            ]
        )

    def forward(
        self, xs: ComplexTensor, ilens: torch.LongTensor, ctxs: torch.FloatTensor = None
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """The forward function

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
            ctxs: (num_spkr, B, T, ctx_dim)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        _, _, C, input_length = xs.size()
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

        # append contextual embeddings to the encoded representations
        if self.use_ctx and ctxs is not None:
            assert len(ctxs) == self.num_ctx, (len(ctxs), self.num_ctx)
            # (num_spkr, B, T', ctx_dim) -> (num_spkr, B, C, T, ctx_dim)
            T = xs.shape[-2]
            ctx_dim = ctxs.shape[-1]
            ctxs = torch.nn.functional.interpolate(ctxs, (T, ctx_dim))
            ctxs = ctxs.unsqueeze(-3).repeat(1, 1, C, 1, 1)
            xs2 = [
                # -> (B, C, T, D + ctx_dim)
                torch.cat((xs, ctx), dim=-1)
                for ctx in ctxs
            ]

        masks = []
        for i, linear in enumerate(self.linears):
            # xs: (B, C, T, D) -> mask:(B, C, T, F)
            if self.use_ctx and ctxs is not None and i < self.num_ctx:
                mask = linear(xs2[i])
            else:
                mask = linear(xs)

            mask = torch.sigmoid(mask)
            # Zero padding
            mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

            # (B, C, T, F) -> (B, F, C, T)
            mask = mask.permute(0, 3, 1, 2)

            # Take cares of multi gpu cases: If input_length > max(ilens)
            if mask.size(-1) < input_length:
                mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens
