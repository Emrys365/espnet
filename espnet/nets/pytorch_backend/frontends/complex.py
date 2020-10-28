import torch
from torch_complex.tensor import ComplexTensor


def hermite(a):
    return a.transpose(-2, -1).conj()

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

def matmul(t1, t2):
    real1, imag1 = t1.real, t1.imag
    real2, imag2 = t2.real, t2.imag
    o_real = torch.matmul(real1, real2) - torch.matmul(imag1, imag2)
    o_imag = torch.matmul(real1, imag2) + torch.matmul(imag1, real2)
    return o_real + 1j * o_imag

class Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        ctx.save_for_backward(input)
        x, _ = torch.solve(b, A)
        ctx.save_for_backward(A, x)
        return x
    @staticmethod
    def backward(ctx, grad_output):
        A, x = ctx.saved_tensors        
        gb, _ = torch.solve(grad_output, hermite(A))
        gA = - matmul(gb, hermite(x))
        return gA, gb

if __name__ == '__main__':
    import numpy as np
    M = np.random.randn(5, 4) + 1j * np.random.randn(5, 4)
    M = M @ M.T.conj()
    Solve.apply(torch.tensor(M, requires_grad=True), torch.tensor(M, requires_grad=True)).sum().real.backward()
