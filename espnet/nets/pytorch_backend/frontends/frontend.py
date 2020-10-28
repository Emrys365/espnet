from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import logging
import numpy
import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

#from espnet.nets.pytorch_backend.frontends.dnn_beamformer import DNN_Beamformer
#from espnet.nets.pytorch_backend.frontends.dnn_wpe import DNN_WPE


class Frontend(nn.Module):
    def __init__(
        self,
        idim: int,
        use_vad_mask: bool = False,
        # WPE options
        use_wpe: bool = False,
        wtype: str = "blstmp",
        wlayers: int = 3,
        wunits: int = 300,
        wprojs: int = 320,
        wdropout_rate: float = 0.0,
        taps: int = 5,
        delay: int = 3,
        use_dnn_mask_for_wpe: bool = True,
        wpe_tag: str = 'default',
        # Beamformer options
        use_beamformer: bool = False,
        btype: str = "blstmp",
        blayers: int = 3,
        bunits: int = 300,
        bprojs: int = 320,
        bnmask: int = 2,
        badim: int = 320,
        ref_channel: int = -1,
        beamformer_type: str = "mvdr",
        bdropout_rate: float = 0.0,
        beamforming_tag: str = 'default',
    ):
        super().__init__()

        self.use_beamformer = use_beamformer
        self.use_wpe = use_wpe
        self.use_dnn_mask_for_wpe = use_dnn_mask_for_wpe
        # use frontend for all the data,
        # e.g. in the case of multi-speaker speech separation
        self.use_frontend_for_all = bnmask > 2
        self.use_vad_mask = use_vad_mask

        self.wpe_tag = wpe_tag
        self.beamforming_tag = beamforming_tag
        if self.wpe_tag == "default":
            logging.warning('Using default DNN_WPE module')
            from espnet.nets.pytorch_backend.frontends.dnn_wpe import DNN_WPE
        elif self.wpe_tag == "double_precison":
            logging.warning('Using double-precision DNN_WPE module')
            from espnet.nets.pytorch_backend.frontends.dnn_wpe_double_pt import DNN_WPE
        elif self.wpe_tag == "diagonal_loading":
            logging.warning('Using DNN_WPE module with diagonal loadings')
            from espnet.nets.pytorch_backend.frontends.dnn_wpe_diag_loading import DNN_WPE
        elif self.wpe_tag == "mask_flooring":
            logging.warning('Using DNN_WPE module with mask flooring')
            from espnet.nets.pytorch_backend.frontends.dnn_wpe_flooring import DNN_WPE
        elif self.wpe_tag == "new_inverse_impl":
            logging.warning('Using DNN_WPE module with new implementation of complex inverse/solve')
            from espnet.nets.pytorch_backend.frontends.dnn_wpe_new_inverse import DNN_WPE
        else:
            raise NotImplementedError('Unknown wpe_tag: {}'.format(self.wpe_tag))

        if self.beamforming_tag == "default":
            logging.warning('Using default DNN_Beamformer module')
            from espnet.nets.pytorch_backend.frontends.dnn_beamformer import DNN_Beamformer
        elif self.beamforming_tag == "double_precison":
            logging.warning('Using double-precision DNN_Beamformer module')
            from espnet.nets.pytorch_backend.frontends.dnn_beamformer_double_pt import DNN_Beamformer
        elif self.beamforming_tag == "diagonal_loading":
            logging.warning('Using DNN_Beamformer module with diagonal loadings')
            from espnet.nets.pytorch_backend.frontends.dnn_beamformer_diag_loading import DNN_Beamformer
        elif self.beamforming_tag == "mask_flooring":
            logging.warning('Using DNN_Beamformer module with mask flooring')
            from espnet.nets.pytorch_backend.frontends.dnn_beamformer_flooring import DNN_Beamformer
        elif self.beamforming_tag == "new_inverse_impl":
            logging.warning('Using DNN_Beamformer module with new implementation of complex inverse/solve')
            from espnet.nets.pytorch_backend.frontends.dnn_beamformer_new_inverse import DNN_Beamformer
        else:
            raise NotImplementedError('Unknown beamforming_tag: {}'.format(self.beamforming_tag))
            
        if self.use_wpe:
            if self.use_dnn_mask_for_wpe:
                # Use DNN for power estimation
                # (Not observed significant gains)
                iterations = 1
            else:
                # Performing as conventional WPE, without DNN Estimator
                iterations = 2

            self.wpe = DNN_WPE(
                wtype=wtype,
                widim=idim,
                wunits=wunits,
                wprojs=wprojs,
                wlayers=wlayers,
                taps=taps,
                delay=delay,
                dropout_rate=wdropout_rate,
                iterations=iterations,
                use_dnn_mask=use_dnn_mask_for_wpe,
                use_vad_mask=self.use_vad_mask,
            )
        else:
            self.wpe = None

        if self.use_beamformer:
            self.beamformer = DNN_Beamformer(
                btype=btype,
                bidim=idim,
                bunits=bunits,
                bprojs=bprojs,
                blayers=blayers,
                bnmask=bnmask,
                dropout_rate=bdropout_rate,
                badim=badim,
                ref_channel=ref_channel,
                beamformer_type=beamformer_type,
                use_vad_mask=self.use_vad_mask,
            )
        else:
            self.beamformer = None

    def forward(
        self, x: ComplexTensor, ilens: Union[torch.LongTensor, numpy.ndarray, List[int]]
    ) -> Tuple[ComplexTensor, torch.LongTensor, Optional[ComplexTensor]]:
        assert len(x) == len(ilens), (len(x), len(ilens))
        # (B, T, F) or (B, T, C, F)
        if x.dim() not in (3, 4):
            raise ValueError(f"Input dim must be 3 or 4: {x.dim()}")
        if not torch.is_tensor(ilens):
            ilens = torch.from_numpy(numpy.asarray(ilens)).to(x.device)

        mask = None
        h = x
        if h.dim() == 4:
            if self.training:
                choices = [(False, False)] if not self.use_frontend_for_all else []
                if self.use_wpe and self.use_beamformer:
                    choices.append((True, True))

#                if self.use_wpe:
#                    choices.append((True, False))

                if self.use_beamformer:
                    choices.append((False, True))

                use_wpe, use_beamformer = choices[numpy.random.randint(len(choices))]

            else:
                use_wpe = self.use_wpe and self.wpe.taps > 0
                use_beamformer = self.use_beamformer

            # 1. WPE
            if use_wpe:
                # h: (B, T, C, F) -> h: (B, T, C, F)
                h, ilens, mask = self.wpe(h, ilens)

            # 2. Beamformer
            if use_beamformer:
                # h: (B, T, C, F) -> h: (B, T, F)
                h, ilens, mask = self.beamformer(h, ilens)
                # mask post-processing
#                for spk in range(len(h)):
#                    h[spk] = h[spk] * mask[spk][..., self.beamformer.ref_channel, :]

        return h, ilens, mask


def frontend_for(args, idim):
    return Frontend(
        idim=idim,
        use_vad_mask=args.use_vad_mask,
        # WPE options
        use_wpe=args.use_wpe,
        wtype=args.wtype,
        wlayers=args.wlayers,
        wunits=args.wunits,
        wprojs=args.wprojs,
        wdropout_rate=args.wdropout_rate,
        taps=args.wpe_taps,
        delay=args.wpe_delay,
        use_dnn_mask_for_wpe=args.use_dnn_mask_for_wpe,
        wpe_tag=getattr(args, 'wpe_tag', 'default'),
        # Beamformer options
        use_beamformer=args.use_beamformer,
        btype=args.btype,
        blayers=args.blayers,
        bunits=args.bunits,
        bprojs=args.bprojs,
        bnmask=args.bnmask,
        badim=args.badim,
        ref_channel=args.ref_channel,
        bdropout_rate=args.bdropout_rate,
        beamforming_tag=getattr(args, 'beamforming_tag', 'default'),
    )
