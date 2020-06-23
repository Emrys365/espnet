#!/usr/bin/env python
# encoding: utf-8

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# https://github.com/nttcslab-sp/dnn_wpe
from pytorch_wpe import get_filter_matrix_conj
# from pytorch_wpe import perform_filter_operation_v2

import numpy as np
import torch
import torch.nn.functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


#######################################################
#                         NOTE                        #
#-----------------------------------------------------#
# This is a variant of NTT's WPD convolutional        #
# beamformer.                                         #
#                                                     #
# reference:                                          #
#  [1] Nakatani, T., & Kinoshita, K. (2019). Maximum  #
#      likelihood convolutional beamformer for        #
#      simultaneous denoising and dereverberation.    #
#      arXiv preprint arXiv:1908.02710.               #
#  [2] Nakatani, T., & Kinoshita, K. (2019).          #
#      Simultaneous denoising and dereverberation for #
#      low-latency applications using frame-by-frame  #
#      online unified convolutional beamformer. Proc. #
#      Interspeech 2019, 111-115.                     #
#######################################################
class MaxEigenVector(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        def get_one_hot(targets, nb_classes):
            res = np.eye(nb_classes, dtype=int)[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape)+[nb_classes])

        device = input.device
        dtype = input.dtype
        # input: (..., C, C)
        input_np = input.detach().cpu().numpy()
        # eigendecomposition of `input` and `input.conj()` has the same order
        eigenvals, eigenvecs = np.linalg.eig(input_np)
        eigenvals_conj, eigenvecs_conj = np.linalg.eig(input_np.conj().swapaxes(-1,-2))
        # max_eigen_value: (...,)
        max_eigen_value = np.max(eigenvals, axis=-1)
        # max_index: (...,)
        max_index = np.argmax(eigenvals, axis=-1)
        # max_index: (..., C)
        max_index2 = get_one_hot(max_index, eigenvals.shape[-1])
        # max_eigen_vector: (..., C, 1)
        max_eigen_vector = np.einsum('...cd,...de->...ce', eigenvecs, max_index2[..., None])
        max_eigen_vector_conj = np.einsum('...cd,...de->...ce', eigenvecs_conj, max_index2[..., None])
        # ctx.save_for_backward(input)
        ctx.intrm = input, max_eigen_value, max_eigen_vector, max_eigen_vector_conj, device

        # uncomment the following lines to validate the result
        # assert np.allclose(np.matmul(input_np, max_eigen_vector[..., None]).squeeze(), max_eigen_value[..., None] * max_eigen_vector)
        # assert np.all([np.allclose(max_eigen_vector[ix], eigenvecs[ix][:, max_index[ix]]) for ix in np.ndindex(max_index.shape)])
        # (..., C)
        real = torch.from_numpy(max_eigen_vector.squeeze().real)
        imag = torch.from_numpy(max_eigen_vector.squeeze().imag)
        real.requires_grad = True
        imag.requires_grad = True
        return real.to(device, dtype=dtype), imag.to(device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # reference: On Differentiating Eigenvalues and Eigenvectors
        # input, = ctx.saved_tensors
        input, max_eigen_value, max_eigen_vector, max_eigen_vector_conj, device = ctx.intrm
        output = grad_output.data.cpu().numpy()

        *B, Cdim = max_eigen_vector.shape[-1]
        ones = (1,) * len(B)
        # beye: (..., C, C)
        beye = np.tile(np.eye(Cdim).reshape(*ones, Cdim, Cdim), (*B, 1, 1))
        pinv = np.linalg.pinv(eigenvals[..., None] * beye - input)
        mid_term = beye - np.matmaul(max_eigen_vector, max_eigen_vector_conj.conj().swapaxes(-1, -2)) \
                   / np.matmul(max_eigen_vector_conj.conj().swapaxes(-1, -2), max_eigen_vector)[..., None, None]
        grad = np.matmul(np.matmul(pinv, midterm), np.matmul(output, max_eigen_vector))
        return ComplexTensor(grad.squeeze()).to(device)


def get_inverse(mat):
    '''Calculate the inverse of the input complex matrix with stabler performance

    Args:
        mat: ComplexTensor
    
    Returns:
        inv_mat: ComplexTensor
    '''
    try:
        inv_mat = mat.inverse()
    except:
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(mat.real),
                                            torch.rand_like(mat.real))*1e-4
            mat = mat / 10e+4 + reg_coeff_tensor
            inv_mat = mat.inverse()
        except:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(mat.real),
                                            torch.rand_like(mat.real))*1e-1
            mat = mat / 10e+10 + reg_coeff_tensor
            inv_mat = mat.inverse()
    return inv_mat

def signal_framing(signal: Union[torch.Tensor, ComplexTensor],
                   frame_length: int,
                   frame_step: int,
                   bdelay: int,
                   do_padding: bool = False,
                   pad_value: int = 0) -> Union[torch.Tensor, ComplexTensor]:
    """Expand `signal` into several frames of `frame_length`.

    Args:
        signal : (..., T)
        frame_length:   length of each segment
        frame_step:     step for selecting frames
        bdelay:         delay for WPD
        do_padding:     whether or not to pad the input signal at the beginning
                          of the time dimension
        pad_value:      value to fill in the padding

    Returns:
        torch.Tensor:
            if do_padding: (..., T, frame_length)
            else:          (..., T - bdelay - frame_length + 2, frame_length)
    """
    if isinstance(signal, ComplexTensor):
        real = signal_framing(signal.real, frame_length, frame_step, bdelay, do_padding, pad_value)
        imag = signal_framing(signal.imag, frame_length, frame_step, bdelay, do_padding, pad_value)
        return ComplexTensor(real, imag)
    else:
        frame_length2 = frame_length - 1
        # pad to the left at the last dimension of `signal` (time dimension)
        if do_padding:
            # (..., T) --> (..., T + bdelay + frame_length - 2)
            signal = F.pad(signal, (bdelay + frame_length2 - 1, 0), 'constant', pad_value)

        # indices:
        # [[ 0, 1, ..., frame_length2 - 1,              frame_length2 - 1 + bdelay ],
        #  [ 1, 2, ..., frame_length2,                  frame_length2 + bdelay     ],
        #  [ 2, 3, ..., frame_length2 + 1,              frame_length2 + 1 + bdelay ],
        #  ...
        #  [ T-bdelay-frame_length2, ..., T-1-bdelay,   T-1 ]
        indices = [[*range(i, i + frame_length2), i + frame_length2 + bdelay - 1]
                    for i in range(0, signal.shape[-1] - frame_length2 - bdelay + 1,
                                   frame_step)]

        # (..., T - bdelay - frame_length + 2, frame_length)
        signal = signal[..., indices]
        return signal

def get_power_spectral_density_matrix(xs: ComplexTensor, mask: torch.Tensor,
                                      normalization=True,
                                      eps: float = 1e-15
                                      ) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    (ported from ESPnet)

    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (ComplexTensor): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = FC.einsum('...ct,...et->...tce', [xs, xs.conj()])

    # Averaging mask along C: (..., C, T) -> (..., T)
    mask = mask.mean(dim=-2)

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    # psd: (..., T, C, C)
    psd = psd_Y * mask[..., None, None]
    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(dim=-3)

    return psd

def get_covariances(Y: ComplexTensor, inverse_power: torch.Tensor,
                    bdelay: int, btaps: int) -> ComplexTensor:
    """Calculates the power normalized spatio-temporal covariance matrix of the framed signal.

    Args:
        Y : Complext STFT signal with shape (B * F, C, T)
        inverse_power : Weighting factor with shape (B * F, T)

    Returns:
        Correlation matrix of shape (B * F, (btaps+1) * C, (btaps+1) * C)
        Correlation vector of shape (B * F, btaps + 1, C, C)
    """
    assert inverse_power.dim() == 2, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), \
        (inverse_power.size(0), Y.size(0))

    BF, C, T = Y.shape

    # (B * F, C, T - bdelay - btaps + 1, btaps + 1)
    Psi = signal_framing(
        Y, btaps + 1, 1, bdelay, do_padding=False)[..., :T - bdelay - btaps + 1, :]
    # Reverse along btaps-axis: [tau, tau-bdelay, tau-bdelay-1, ..., tau-bdelay-frame_length+1]
    Psi = FC.reverse(Psi, dim=-1)
    Psi_conj_norm = \
        Psi.conj() * inverse_power[..., None, bdelay + btaps - 1:, None]

    # let T' = T - bdelay - btaps + 1
    # (B * F, C, T', btaps + 1) x (B * F, C, T', btaps + 1) -> (B * F, btaps + 1, C, btaps + 1, C)
    covariance_matrix = FC.einsum('fdtk,fetl->fkdle', (Psi_conj_norm, Psi))

    btaps2 = Psi.shape[-1]    # = btaps + 1
    # (B * F, btaps + 1, C, btaps + 1, C) -> (B * F, (btaps + 1) * C, (btaps + 1) * C)
    covariance_matrix = covariance_matrix.view(BF, btaps2 * C, btaps2 * C)

    # (B * F, C, T', btaps + 1) x (B * F, C, T')
    #    --> (B * F, btaps +1, C, C)
    covariance_vector = FC.einsum(
        'fdtk,fet->fked', (Psi_conj_norm, Y[..., bdelay + btaps - 1:]))

    return covariance_matrix, covariance_vector

def perform_WPE_filtering(Y: ComplexTensor,
                          filter_matrix_conj: ComplexTensor,
                          btaps, bdelay) -> ComplexTensor:
    """perform_filter_operation_v2

    modified from https://github.com/nttcslab-sp/dnn_wpe/blob/master/pytorch_wpe.py#L172-L188

    Args:
        Y : Complex-valued STFT signal of shape (B * F, C, T)
        filter_matrix_conj: Filter matrix (B * F, btaps + 1, C, C)

    Returns:
        Y_enhanced: (B * F, C, T)
    """
    # (B * F, C, T) --> (B * F, C, T, btaps + 1)
    Y_tilde = signal_framing(
        Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Y_tilde = FC.reverse(Y_tilde, dim=-1)

    # (B * F, btaps + 1, C, C) x (B * F, C, T, btaps + 1)
    #   --> (B * F, C, T)
    reverb_tail = FC.einsum('fpde,fdtp->fet', (filter_matrix_conj, Y_tilde))
    return Y - reverb_tail

def get_RTF(psd_s, psd_n):
    """Estimate Relative Transfer Function

    Args:
        psd_s (ComplexTensor): (B, F, C, C)
        psd_n (ComplexTensor): (B, F, C, C)

    Returns:
        rtf (ComplexTensor): (B, F, C * (btaps + 1))
    """
    pass

def get_WPD_filter_conj(Rf: ComplexTensor,
                        ubar: torch.Tensor,
                        eps: float = 1e-15) -> ComplexTensor:
    """Return the WPD (Weighted Power minimization Distortionless response convolutional beamformer) vector:

        h = (Rf^-1 @ ubar) / (ubar^H @ (Rf^-1) @ ubar)

    Reference:
        Maximum likelihood convolutional beamformer for simultaneous denoising
        and dereverberation; Nakatani, T. and Kinoshita, K., 2019;
        https://arxiv.org/abs/1908.02710

    Args:
        Rf (ComplexTensor): (B * F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        ubar (torch.Tensor): (B * F, (btaps + 1) * C)
            is the reference_vector.
        eps (float):

    Returns:
        filter_matrix_conj (ComplexTensor): (B * F, (btaps + 1) * C)
    """
    inv_Rf = get_inverse(Rf)
    if ubar.dim() < Rf.dim():
        # (B * F, (btaps + 1) * C, 1)
        ubar = ubar.unsqueeze(-1)

    # convolutional beamformer coefficients (length = btaps + 1)
    # (B * F, (btaps + 1) * C, 1)
    stacked_filter = FC.matmul(inv_Rf, ubar)
    # ubar^H @ (Rf^-1) @ ubar: (B * F, 1, 1) scalar
    denominator = FC.matmul(ubar.conj().transpose(-1, -2), inv_Rf)
    denominator = FC.matmul(denominator, ubar)
    # denominator = FC.einsum('...ij,...jk,...kl->...il', [ubar.conj().transpose(-1, -2), inv_Rf, ubar])
    # (B * F, (btaps + 1) * C)
    filter_matrix = stacked_filter.squeeze() / denominator.squeeze(-1)
    # (B * F, (btaps + 1) * C)
    return filter_matrix.conj()

def perform_WPD_filtering(Y: ComplexTensor,
                          filter_matrix_conj: ComplexTensor,
                          bdelay: int, btaps: int) \
        -> ComplexTensor:
    """perform_filter_operation

    Args:
        Y : Complex STFT signal with shape (B * F, C, T)
        filter_matrix_conj: Filter matrix (B * F, (btaps + 1) * C)

    Returns:
        enhanced (ComplexTensor): (B * F, T)
    """
    # (B * F, C, T) --> (B * F, C, T, btaps + 1)
    Ytilde = signal_framing(
        Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Ytilde = FC.reverse(Ytilde, dim=-1)

    BF, C, T = Ytilde.shape[:3]
    # --> (B * F, T, btaps + 1, C) --> (B * F, T, (btaps + 1) * C)
    Ytilde = Ytilde.permute(0, 2, 3, 1).contiguous().view(BF, T, -1)
    # (B * F, T, 1)
    enhanced = FC.matmul(Ytilde, filter_matrix_conj.unsqueeze(-1))
    return enhanced.squeeze(-1)


if __name__ == '__main__':
    ############################################
    #                  Example                 #
    ############################################
    eps = 1e-10
    btaps = 5
    bdelay = 3
    # pretend to be some STFT: (B, F, C, T)
    Z = ComplexTensor(torch.rand(4, 256, 2, 518), torch.rand(4, 256, 2, 518))

    # Calculate power: (B, F, C, T)
    power = Z.real ** 2 + Z.imag ** 2
    # pretend to be some mask
    mask_speech = torch.ones_like(Z.real)
    # (..., C, T) * (..., C, T) -> (..., C, T)
    power = power * mask_speech
    # Averaging along the channel axis: (B, F, C, T) -> (B, F, T)
    power = power.mean(dim=-2)
    # (B, F, T) --> (B * F, T)
    power = power.view(-1, power.shape[-1])
    inverse_power = 1 / torch.clamp(power, min=eps)

    B = Z.shape[0]
    C, T = Z.shape[-2:]
    # (B, F, C, T) --> (B * F, C, T)
    Y = Z.view(-1, *Z.shape[-2:])

    # covariance matrix: (B * F, (btaps+1) * C, (btaps+1) * C)
    # covariance vector: (B * F, btaps + 1, C, C)
    covariance_matrix, covariance_vector = get_covariances(Y, inverse_power, bdelay, btaps)

    #====== perform MIMO Nara-WPE first
    # (B * F, btaps + 1, C, C)
    WPE_filter_conj = get_filter_matrix_conj(covariance_matrix, covariance_vector)
    # dereverberated Y: (B * F, C, T)
    Y_enhanced = perform_WPE_filtering(Y, WPE_filter_conj, btaps, bdelay)
    # Y_enhanced = perform_filter_operation_v2(Y, WPE_filter_conj, btaps, bdelay)
    # (B, F, C, T)
    Y_enhanced = Y_enhanced.view(B, -1, C, T)

    #====== `Y_enhanced` is used to estimate the steering vector `u`
    # pretend to be the steering vector estimated by an attention network
    # (B, F, C)
    u = ComplexTensor(torch.rand(Z.real.shape[0], Z.real.shape[1], C), torch.rand(Z.real.shape[0], Z.real.shape[1], C))
    # (B, F, (btaps + 1) * C): [u^T, 0, 0, ..., 0]^T
    ubar = FC.pad(u, pad=(0, C * btaps), mode='constant', value=0)
    # (B * F, (btaps + 1) * C)
    ubar = ubar.view(-1, ubar.shape[-1])

    # (B * F, (btaps + 1) * C)
    WPD_filter_conj = get_WPD_filter_conj(covariance_matrix, ubar)

    # (B * F, T)
    enhanced = perform_WPD_filtering(Y, WPD_filter_conj, bdelay, btaps)
    # (B, F, T)
    enhanced = enhanced.view(B, -1, T)
