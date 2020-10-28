#!/usr/bin/env python
# encoding: utf-8

from typing import Union

# from pytorch_wpe import perform_filter_operation_v2

import torch
import torch.nn.functional as F
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


def signal_framing(signal: Union[torch.Tensor, ComplexTensor],
                   frame_length: int,
                   frame_step: int,
                   bdelay: int,
                   do_padding: bool = True,
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
        # pad to the right at the last dimension of `signal` (time dimension)
        if do_padding:
            # (..., T) --> (..., T + bdelay + frame_length - 2)
            signal = FC.pad(signal, (bdelay + frame_length2 - 1, 0), 'constant', pad_value)

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
        # signal[..., :-1] = -signal[..., :-1]
        return signal


def get_power_spectral_density_matrix(xs: ComplexTensor, mask: torch.Tensor,
                                      normalization=True,
                                      eps: float = 1e-15
                                      ) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    (ported from ESPnet)

    Args:
        xs (ComplexTensor): (B, F, C, T)
        mask (torch.Tensor): (B, F, C, T)
        normalization (bool):
        eps (float):
    Returns:
        psd (ComplexTensor): (B, F, C, C)
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


def get_stacked_covariance(Y: ComplexTensor,
                           mask: torch.Tensor,
                           btaps: int,
                           normalization=True,
                           eps: float = 1e-15
                           ) -> ComplexTensor:
    """Return PSD matrix of the stacked speech signal x(t,f) = [Y^T(t,f), 0, ..., 0]^T

    Args:
        Y (ComplexTensor): (B, F, C, T)
        mask (torch.Tensor): (B, F, C, T)
        btaps (int):
        normalization (bool):
        eps (float):
    Returns:
        psd (ComplexTensor): (B, F, (btaps + 1) * C, (btaps + 1) * C)
    """
    C = Y.shape[-2]
    # --> (B, F, T, C) --> (B, F, T, (btaps + 1) * C)
    #  --> (B, F, (btaps + 1) * C, T)
    Y_tilde = FC.pad(Y.transpose(-1, -2), (0, btaps * C), 'constant', 0).transpose(-1, -2)
    mask_tilde = FC.pad(mask.transpose(-1, -2), (0, btaps * C), 'constant', 0).transpose(-1, -2)
    # (B, F, (btaps + 1) * C, (btaps + 1) * C)
    return get_power_spectral_density_matrix(Y_tilde, mask_tilde, normalization, eps)


def get_masked_covariance(Y: ComplexTensor,
                          inverse_power: torch.Tensor,
                          bdelay: int,
                          btaps: int,
                          mask: ComplexTensor,
                          normalization=True,
                          eps: float = 1e-15) -> ComplexTensor:
    """Calculates the power normalized spatio-temporal covariance matrix of the framed signal.

    Args:
        Y : Complext STFT signal with shape (B, F, C, T)
        inverse_power : Weighting factor with shape (B, F, T)
        mask : Complext STFT signal with shape (B, F, C, T)

    Returns:
        Correlation matrix of shape (B, F, (btaps+1) * C, (btaps+1) * C)
    """
    assert inverse_power.dim() == 3, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), \
        (inverse_power.size(0), Y.size(0))

    Bs, Fdim, C, T = Y.shape

    # (B, F, C, T - bdelay - btaps + 1, btaps + 1)
    Psi = signal_framing(
        Y, btaps + 1, 1, bdelay, do_padding=False)[..., :T - bdelay - btaps + 1, :]
    # Reverse along btaps-axis: [tau, tau-bdelay, tau-bdelay-1, ..., tau-bdelay-frame_length+1]
    Psi = FC.reverse(Psi, dim=-1)
    Psi_norm = \
        Psi * inverse_power[..., None, bdelay + btaps - 1:, None]

    # Averaging mask along C: (B, F, C, T) -> (B, F, T)
    mask = mask.mean(dim=-2)
    # Normalized mask along T: (B, F, T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    # (B, F, T - bdelay - btaps + 1)
    mask_bar = mask[..., bdelay + btaps - 1:]

    # let T' = T - bdelay - btaps + 1
    # (B, F, C, T', btaps + 1) x (B, F, C, T', btaps + 1) -> (B, F, T', btaps + 1, C, btaps + 1, C)
    covariance_matrix = FC.einsum('bfdtk,bfetl->bftkdle', (Psi, Psi_norm.conj()))

    covariance_matrix = covariance_matrix * mask_bar[..., None, None, None, None]
    # sum along T': (B, F, btaps + 1, C, btaps + 1, C)
    covariance_matrix = covariance_matrix.sum(dim=2)

    # (B, F, btaps + 1, C, btaps + 1, C) -> (B, F, (btaps + 1) * C, (btaps + 1) * C)
    covariance_matrix = covariance_matrix.view(Bs, Fdim, (btaps + 1) * C, (btaps + 1) * C)
    return covariance_matrix


def get_covariances(Y: ComplexTensor,
                    inverse_power: torch.Tensor,
                    bdelay: int,
                    btaps: int,
                    get_vector: bool = False) -> ComplexTensor:
    """Calculates the power normalized spatio-temporal covariance matrix of the framed signal.

    Args:
        Y : Complext STFT signal with shape (B, F, C, T)
        inverse_power : Weighting factor with shape (B, F, T)

    Returns:
        Correlation matrix of shape (B, F, (btaps+1) * C, (btaps+1) * C)
        Correlation vector of shape (B, F, btaps + 1, C, C)
    """
    assert inverse_power.dim() == 3, inverse_power.dim()
    assert inverse_power.size(0) == Y.size(0), \
        (inverse_power.size(0), Y.size(0))

    Bs, Fdim, C, T = Y.shape

    # (B, F, C, T - bdelay - btaps + 1, btaps + 1)
    Psi = signal_framing(
        Y, btaps + 1, 1, bdelay, do_padding=False)[..., :T - bdelay - btaps + 1, :]
    # Reverse along btaps-axis: [tau, tau-bdelay, tau-bdelay-1, ..., tau-bdelay-frame_length+1]
    Psi = FC.reverse(Psi, dim=-1)
    Psi_norm = \
        Psi * inverse_power[..., None, bdelay + btaps - 1:, None]

    # let T' = T - bdelay - btaps + 1
    # (B, F, C, T', btaps + 1) x (B, F, C, T', btaps + 1) -> (B, F, btaps + 1, C, btaps + 1, C)
    covariance_matrix = FC.einsum('bfdtk,bfetl->bfkdle', (Psi, Psi_norm.conj()))

    # (B, F, btaps + 1, C, btaps + 1, C) -> (B, F, (btaps + 1) * C, (btaps + 1) * C)
    covariance_matrix = covariance_matrix.view(Bs, Fdim, (btaps + 1) * C, (btaps + 1) * C)

    if get_vector:
        # (B, F, C, T', btaps + 1) x (B, F, C, T')
        #    --> (B, F, btaps +1, C, C)
        covariance_vector = FC.einsum(
            'bfdtk,bfet->bfked', (Psi_norm, Y[..., bdelay + btaps - 1:].conj()))
        return covariance_matrix, covariance_vector
    else:
        return covariance_matrix


def perform_WPE_filtering(Y: ComplexTensor,
                          filter_matrix_conj: ComplexTensor,
                          btaps, bdelay) -> ComplexTensor:
    """perform_filter_operation_v2

    modified from https://github.com/nttcslab-sp/dnn_wpe/blob/master/pytorch_wpe.py#L172-L188

    Args:
        Y : Complex-valued STFT signal of shape (B, F, C, T)
        filter_matrix_conj: Filter matrix (B, F, btaps + 1, C, C)

    Returns:
        Y_enhanced: (B, F, C, T)
    """
    # (B, F, C, T) --> (B, F, C, T, btaps + 1)
    Y_tilde = signal_framing(
        Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Y_tilde = FC.reverse(Y_tilde, dim=-1)

    # (B, F, btaps + 1, C, C) x (B, F, C, T, btaps + 1)
    #   --> (B, F, C, T)
    reverb_tail = FC.einsum('bfpde,bfdtp->bfet', (filter_matrix_conj, Y_tilde))
    return Y - reverb_tail


def get_WPD_filter_conj(Rf: ComplexTensor,
                        Phi: ComplexTensor,
                        reference_vector: torch.Tensor,
                        eps: float = 1e-15) -> ComplexTensor:
    """Return the WPD (Weighted Power minimization Distortionless response convolutional beamformer) vector:

        h = (Rf^-1 @ Phi_{xx}) / tr[(Rf^-1) @ Phi_{xx}] @ u

    Reference:
        Maximum likelihood convolutional beamformer for simultaneous denoising
        and dereverberation; Nakatani, T. and Kinoshita, K., 2019;
        https://arxiv.org/abs/1908.02710

    Args:
        Rf (ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the power normalized spatio-temporal covariance matrix.
        Phi (ComplexTensor): (B, F, (btaps+1) * C, (btaps+1) * C)
            is the covariance matrix of [x^T(t,f) 0 ... 0]^T.
        reference_vector (torch.Tensor): (B, (btaps+1) * C)
            is the reference_vector.
        eps (float):

    Returns:
        filter_matrix_conj (ComplexTensor): (B, F, (btaps + 1) * C)
    """
    try:
        inv_Rf = Rf.inverse()
    except:
        try:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(Rf.real),
                                             torch.rand_like(Rf.real)) * 1e-4
            Rf = Rf / 10e+4
            Phi = Phi / 10e+4
            Rf += reg_coeff_tensor
            inv_Rf = Rf.inverse()
        except:
            reg_coeff_tensor = ComplexTensor(torch.rand_like(Rf.real),
                                             torch.rand_like(Rf.real)) * 1e-1
            Rf = Rf / 10e+10
            Phi = Phi / 10e+10
            Rf += reg_coeff_tensor
            inv_Rf = Rf.inverse()

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum('...ec,...cd->...ed', [inv_Rf, Phi])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum('...fec,...c->...fe', [ws, reference_vector])
    # (B, F, (btaps + 1) * C)
    return beamform_vector.conj()


def get_WPD_filter_conj_with_atf(
    psd_observed_bar: ComplexTensor,
    psd_speech: ComplexTensor,
    psd_noise: ComplexTensor,
    iterations: int = 3,
    reference_vector=None,
    normalize_ref_channel=None,
    eps: float = 1e-15,
) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_observed_bar (ComplexTensor): (..., F, C, C)
        psd_speech (ComplexTensor): (..., F, C, C)
        psd_noise (ComplexTensor): (..., F, C, C)
        iterations (int)
        reference_vector (torch.Tensor): (..., C)
        normalize_ref_channel (int)
        eps (float):
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # Add eps
    B, F = psd_noise.shape[:2]
    C = psd_noise.size(-1)
    eye = torch.eye(C, dtype=psd_noise.dtype, device=psd_noise.device)
    shape = [1 for _ in range(psd_noise.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(B, F, 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(psd_noise).real.abs()[..., None, None] * eps
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps
    psd_noise = psd_noise + epsilon * eye

    # (B, F, C)
#     eigenvec = power_iteration(
#         FC.solve(psd_speech, psd_noise)[0],
#         reference_vector,
#         iterations=iterations
#     )
#     atf = FC.matmul(psd_noise, eigenvec.unsqueeze(-1))
    phi = FC.solve(psd_speech, psd_noise)[0]
    atf = phi[..., reference_vector, None] if isinstance(reference_vector, int) else FC.matmul(phi, reference_vector.unsqueeze(-1))
    for i in range(iterations - 2):
        atf = FC.matmul(phi, atf)
#         atf = atf / complex_norm(atf)
    atf = FC.matmul(psd_speech, atf)

    # (B, F, (K+1)*C, 1)
    atf = FC.pad(atf, (0, 0, 0, psd_observed_bar.shape[-1] - C), 'constant', 0)
    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    numerator = FC.solve(atf, psd_observed_bar)[0].squeeze(-1)
#     numerator = FC.einsum("...ec,...cd->...ed", [ComplexTensor(np.linalg.inv(psd_noise.numpy())), psd_speech])
    denominator = FC.einsum("...d,...d->...", [atf.squeeze(-1).conj(), numerator])
    if normalize_ref_channel is not None:
        scale = atf.squeeze(-1)[..., normalize_ref_channel, None].conj()
        beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    else:
        beamforming_vector = numerator / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector.conj()


def perform_WPD_filtering(Y: ComplexTensor,
                          filter_matrix_conj: ComplexTensor,
                          bdelay: int, btaps: int) \
        -> ComplexTensor:
    """perform_filter_operation

    Args:
        Y : Complex STFT signal with shape (B, F, C, T)
        filter_matrix_conj: Filter matrix (B, F, (btaps + 1) * C)

    Returns:
        enhanced (ComplexTensor): (B, F, T)
    """
    # (B, F, C, T) --> (B, F, C, T, btaps + 1)
    Ytilde = signal_framing(
        Y, btaps + 1, 1, bdelay, do_padding=True, pad_value=0)
    Ytilde = FC.reverse(Ytilde, dim=-1)

    Bs, Fdim, C, T = Y.shape
    # --> (B, F, T, btaps + 1, C) --> (B, F, T, (btaps + 1) * C)
    Ytilde = Ytilde.permute(0, 1, 3, 4, 2).contiguous().view(Bs, Fdim, T, -1)
    # (B, F, T, 1)
    enhanced = FC.einsum('...tc,...c->...t', [Ytilde, filter_matrix_conj])
    # enhanced = FC.matmul(Ytilde, filter_matrix_conj.unsqueeze(-1)).squeeze(-1)
    return enhanced


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

    B, Fdim, C, T = Z.shape

    # covariance matrix: (B, F, (btaps+1) * C, (btaps+1) * C)
    covariance_matrix = get_covariances(Z, inverse_power, bdelay, btaps, get_vector=False)

    # stacked speech signal PSD: (B, F, (btaps+1) * C, (btaps+1) * C)
    psd_x = get_stacked_covariance(Z, mask_speech, btaps, normalization=True)

    # reference vector: (B, C)
    ref_channel = 0
    u = torch.zeros(*(Z.size()[:-3] + (Z.size(-2),)),
                    device=Z.device)
    u[..., ref_channel].fill_(1)

    # (B, F, (btaps + 1) * C)
    WPD_filter_conj = get_WPD_filter_conj(covariance_matrix, psd_x, u)

    # (B, F, T)
    enhanced = perform_WPD_filtering(Z, WPD_filter_conj, bdelay, btaps)
