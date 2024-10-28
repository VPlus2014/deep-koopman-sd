import math
from typing import Callable
import torch
import torch.nn as nn
from torch.nn import functional as F

# RBF Layer
# https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer


class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Credit: JeremyLinux/PyTorch-Radial-Basis-Function-Layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis_func: Callable[[float], float],
    ):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a_param = nn.Parameter(torch.Tensor(out_features))
        self.a_sup = 2.0
        assert self.a_sup > 1
        self.basis_func = basis_func
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1e-1)
        nn.init.constant_(self.a_param, math.log(self.a_sup - 1))

    def calc_a(self):
        r"""0<a<a_sup"""
        a = self.a_sup * F.sigmoid(self.a_param)
        return a

    def forward(self, input: torch.Tensor):
        """(...,dimX)->(...,K)"""
        assert input.shape[-1] == self.in_features, "Input dimension does not match"
        x = input.unsqueeze(-2)  # (...,K,dimX)
        sz1_bc = [1] * (len(input.shape) - 2)
        c = self.centres.view(*sz1_bc, self.out_features, self.in_features)
        a = self.calc_a().view(*sz1_bc, self.out_features)

        dst = (x - c).square_().sum(-1).sqrt_()
        dst = dst * a
        y = self.basis_func(dst)
        return y

    @staticmethod
    def demo():
        n = 4
        t = 50
        dimX = 2
        dimY = 32
        x = torch.randn(n, t, dimX)
        rbf = RBF(dimX, dimY, basis_func=gaussian)
        y = rbf(x)
        l = nn.MSELoss()(y, torch.zeros_like(y))
        l.backward()


# RBFs


def gaussian(alpha: torch.Tensor):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha: torch.Tensor):
    phi = alpha
    return phi


def quadratic(alpha: torch.Tensor):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha: torch.Tensor):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha: torch.Tensor):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha: torch.Tensor):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha: torch.Tensor):
    phi = alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha))
    return phi


def poisson_one(alpha: torch.Tensor):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha: torch.Tensor):
    phi = (
        ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha))
        * alpha
        * torch.exp(-alpha)
    )
    return phi


def matern32(alpha: torch.Tensor):
    phi = (torch.ones_like(alpha) + 3**0.5 * alpha) * torch.exp(-(3**0.5) * alpha)
    return phi


def matern52(alpha: torch.Tensor):
    phi = (
        torch.ones_like(alpha) + 5**0.5 * alpha + (5 / 3) * alpha.pow(2)
    ) * torch.exp(-(5**0.5) * alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {
        "gaussian": gaussian,
        "linear": linear,
        "quadratic": quadratic,
        "inverse quadratic": inverse_quadratic,
        "multiquadric": multiquadric,
        "inverse multiquadric": inverse_multiquadric,
        "spline": spline,
        "poisson one": poisson_one,
        "poisson two": poisson_two,
        "matern32": matern32,
        "matern52": matern52,
    }
    return bases


if __name__ == "__main__":
    RBF.demo()
