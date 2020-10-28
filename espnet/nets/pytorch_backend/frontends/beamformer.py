import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


def complex_norm(c: ComplexTensor) -> torch.Tensor:
    return torch.sqrt(
        (c.real ** 2 + c.imag ** 2).sum(dim=-1, keepdim=True) + 1e-10
    )

def power_iteration(
    matrix: ComplexTensor,
    vector: ComplexTensor,
    iterations: int = 1
) -> ComplexTensor:
    """Perform eigenvalue decomposition via the power iteration method.

    Args:
        matrix: (B, F, C, C) matrix to be eigenvalue decomposed
        vector: (B, F, C) initialized eigen-vector
        iterations: number of iterations
    Returns:
        eigenvec: (B, F, C) estimated eigenvector corresponding to the maximum eigenvalue
    """
    eigenvec = vector
    for i in range(iterations):
        eigenvec = FC.matmul(matrix, eigenvec.unsqueeze(-1)).squeeze(-1)
        eigenvec = eigenvec / complex_norm(eigenvec)
    return eigenvec


def hermite(x):
    return x.transpose(-2, -1).conj()

def ComplexTensor_to_Tensor(t):
    """
    Converts a third party complex tensor to a native complex torch tensor.
    >>> t = ComplexTensor(np.array([1., 2, 3]))
    >>> t
    ComplexTensor(
        real=tensor([1., 2., 3.], dtype=torch.float64),
        imag=tensor([0., 0., 0.], dtype=torch.float64),
    )
    >>> ComplexTensor_to_Tensor(t)
    tensor([(1.+0.j), (2.+0.j), (3.+0.j)], dtype=torch.complex128)
    """
    assert isinstance(t, ComplexTensor), type(t)
    return t.real + 1j * t.imag


def Tensor_to_ComplexTensor(t):
    """
    Converts a native complex torch tensor to a third party complex tensor.
    >>> t = torch.tensor(np.array([1., 2, 3]) + 0 * 1j)
    >>> t
    tensor([(1.+0.j), (2.+0.j), (3.+0.j)], dtype=torch.complex128)
    >>> Tensor_to_ComplexTensor(t)
    ComplexTensor(
        real=tensor([1., 2., 3.], dtype=torch.float64),
        imag=tensor([0., 0., 0.], dtype=torch.float64),
    )
    """
    assert isinstance(t, torch.Tensor), type(t)
    return ComplexTensor(t.real, t.imag)

def ComplexTensor_inverse(t):
    return Tensor_to_ComplexTensor(
            ComplexTensor_to_Tensor(t).inverse()
    )

def get_power_spectral_density_matrix(
    xs: ComplexTensor, mask: torch.Tensor, normalization=True, eps: float = 1e-15
) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (ComplexTensor): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = FC.einsum("...ct,...et->...tce", [xs, xs.conj()])

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


def get_weighted_power_spectral_density_matrix(
    xs: ComplexTensor, mask: torch.Tensor, normalization=True, eps: float = 1e-15
) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (ComplexTensor): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = FC.einsum("...ct,...et->...tce", [xs, xs.conj()])

    raise NotImplementedError
    Y_inverse_power = xs * inverse_power[..., None, :]
    Rd = np.matmul(Y_inverse_power, hermite(Y))

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

def get_mvdr_vector(
    psd_s: ComplexTensor,
    psd_n: ComplexTensor,
    reference_vector: torch.Tensor,
    eps: float = 1e-8,
    eps2: float = 1e-8,
) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        eps (float):
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # Add eps
#    C = psd_n.size(-1)
#    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
#    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
#    eye = eye.view(*shape)
#    psd_n += eps * eye

    # Add eps
    B, F = psd_n.shape[:2]
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape).repeat(B, F, 1, 1)
    with torch.no_grad():
        epsilon = FC.trace(psd_n).real.abs()[..., None, None] * eps
        # in case that correlation_matrix is all-zero
        epsilon = epsilon + eps2
#    try:
#        psd_n_i = (psd_n + epsilon * eye + eps).inverse2()
#        #psd_n_i = ComplexTensor_inverse(psd_n + eps * eye)
#    except:
#        eps2 = 1e-4
#        try:
#            psd_n = psd_n / 10e+4
#            psd_s = psd_s / 10e+4
#            psd_n += eps2 * eye
#            psd_n_i = psd_n.inverse2()
#            #psd_n_i = ComplexTensor_inverse(psd_n)
#        except:
#            try:
#                psd_n = psd_n / 10e+10
#                psd_s = psd_s / 10e+10
#                psd_n += eps2 * eye
#                psd_n_i = psd_n.inverse2()
#                #psd_n_i = ComplexTensor_inverse(psd_n)
#            except:
#                raise Exception('psd not invertable.')

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
#    numerator = FC.einsum("...ec,...cd->...ed", [psd_n.inverse(), psd_s])
#    numerator = FC.einsum("...ec,...cd->...ed", [psd_n_i, psd_s])
    numerator = FC.solve(psd_s, psd_n + epsilon * eye)[0]
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps2)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum("...fec,...c->...fe", [ws, reference_vector])
    return beamform_vector


def get_mvdr_vector_with_atf(
    psd_n: ComplexTensor,
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
        psd_n (ComplexTensor): (..., F, C, C)
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

    # numerator: (..., C_1, C_2) x (..., C_2, 1) -> (..., C_1)
    numerator = FC.solve(atf, psd_n)[0].squeeze(-1)
#     numerator = FC.einsum("...ec,...cd->...ed", [ComplexTensor(np.linalg.inv(psd_noise.numpy())), psd_speech])
    denominator = FC.einsum("...d,...d->...", [atf.squeeze(-1).conj(), numerator])
    if normalize_ref_channel is not None:
        scale = atf.squeeze(-1)[..., normalize_ref_channel, None].conj()
        beamforming_vector = numerator * scale / (denominator.real.unsqueeze(-1) + eps)
    else:
        beamforming_vector = numerator / (denominator.real.unsqueeze(-1) + eps)
    return beamforming_vector


def apply_beamforming_vector(
    beamform_vector: ComplexTensor, mix: ComplexTensor
) -> ComplexTensor:
    # (..., C) x (..., C, T) -> (..., T)
    es = FC.einsum("...c,...ct->...t", [beamform_vector.conj(), mix])
    return es
