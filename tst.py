import time
from typing import List, Union
import numpy as np
import torch
from torch import nn


def attach_dim(v: torch.Tensor, n_dim_to_prepend=0, n_dim_to_append=0):
    """oldshape-> (1,1,...,1,*old_shape,1,1,...,1)"""
    return v.view(*([1] * n_dim_to_prepend), *v.shape, *([1] * n_dim_to_append))


def block_diag(m: Union[List[torch.Tensor], torch.Tensor]):
    if isinstance(m, list):
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n, device=m.device).unsqueeze(-2), d - 3, 1)
    y = m2 * eye
    y = y.reshape(siz0 + torch.Size(torch.tensor(siz1) * n))
    return y


class KoopmanNet(nn.Module):
    def __init__(self, dimX=2, n_eig=16, use_complex=True):
        super(KoopmanNet, self).__init__()
        self.use_complex = use_complex
        if use_complex:
            self.reL = nn.Parameter(torch.randn([n_eig, 1, 1]) * 0.1)
            self.imL = nn.Parameter(torch.randn([n_eig, 1, 1]) * 0.1)
        else:
            self.reL = nn.Parameter(torch.randn([n_eig]) * 0.1)

    def form_koopman(self):
        print(self.reL.device)
        # k = torch.diag(self.reL)
        if self.use_complex:
            relam = self.reL
            imlam = self.imL
            Koopmat = torch.cat(
                [
                    torch.cat([relam, imlam], dim=-1),
                    torch.cat([-imlam, relam], dim=-1),
                ],
                dim=-2,
            )  # (dimH,2,2)
        else:
            Koopmat = self.reL
        return Koopmat


def find_proptions(vars: List[int], kmax: int = 10):
    x0 = vars[0]
    loss_opt = np.inf
    for k0 in range(1, kmax + 1):
        pass


from gymnasium import spaces


def main():
    x = np.asarray([1, 1, np.inf])
    xsp = spaces.Box(low=x, high=x)
    while True:
        x = xsp.sample()
        print(x)
        continue
        if abs(x[-1]) > 1:
            break
    print(x)
    raise

    use_cplx = True
    N = 32
    L = 50
    dimX = 2
    neig = 80
    dimH = neig * (2 if use_cplx else 1)
    dv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtp = torch.float32
    kpmodel = KoopmanNet(dimX=dimX, n_eig=neig, use_complex=use_cplx).to(dv, dtp)
    kpmodel.eval()

    hs = torch.randn([N, L, dimH]).to(dv, dtp)
    kop = kpmodel.form_koopman()  # (M,2,2)
    nitr = 100
    if use_cplx:
        with torch.no_grad():
            with torch.no_grad():
                kmat = (kpmodel.reL + kpmodel.imL * 1j).view(-1).diag()
                kmat = kmat.cpu().numpy()
                rst = np.linalg.eig(kmat)
                print(rst)
                rst

    for mode in range(10):
        valid_mode = True
        methodname = ""
        t0 = time.time()
        for _ in range(nitr):
            kop_ = kop.clone()
            if mode == 0:
                if use_cplx:  # for bmm
                    methodname = "for loop bmm"
                    kop_ = kop_.repeat(N, *([1] * (len(kop_.shape) - 1)))  # (N*M,2,2)
                    xs_ = hs.view(*hs.shape[:-1], neig, 2)
                    ys = []
                    for k in range(L):
                        xk = xs_[:, k]
                        xk = xk.flatten(0, 1).unsqueeze(-2)  # (N*M,1,2)
                        yk = torch.bmm(xk, kop_).view(xk.shape)
                        ys.append(yk)
                else:  # vec bmm with diag mat
                    methodname = "vec bmm with diag mat"
                    kop_ = kop_.view(-1).diag()  # (D,D)
                    kop_ = kop_.unsqueeze(0).repeat(N * L, 1, 1)  # (N*L,D,D)
                    xs_ = hs.view(N * L, 1, -1)
                    ys = torch.bmm(xs_, kop_)
            elif mode == 1:  # vectorized bmm
                if use_cplx:
                    methodname = "vectorized bmm"
                    kop_ = kop_.repeat(
                        N * L, *([1] * (len(kop_.shape) - 1))
                    )  # (N*L*M,2,2)
                    ys = hs.view(-1, 1, 2)
                    ys = torch.bmm(ys, kop_)
                else:  # vec with broadcasting
                    methodname = "vectorized broadcasting"
                    kop_ = kop_.view(*([1] * (len(hs.shape) - 1)), -1)  # (1,1,D)
                    ys = hs * kop_
                pass
            elif mode == 2:  # eisum
                ys = []
                if use_cplx:
                    methodname = "for loop eisum"
                    kop_ = kop_.unsqueeze_(0).repeat(N, 1, 1, 1)  # (N,M,2,2)
                    xs = hs.view(*hs.shape[:-1], neig, 2)
                    for k in range(L):
                        xk = xs[:, k].unsqueeze(-2)  # (N,M,1,2)
                        yk = torch.einsum("nmij,nmjk->nmik", xk, kop_)
                        ys.append(yk)
                else:
                    valid_mode = False
            else:
                valid_mode = False
            if not valid_mode:
                break
        if not valid_mode:
            continue
        t1 = time.time()
        print(f"mode {mode} time: {t1-t0:.6f} s {methodname}")

    return

    N = 128
    N = 32
    dimH = 16
    L = 1000
    r = (torch.randn([N, dimH, 1, 1]) * 0).exp()
    theta = torch.randn([N, dimH, 1, 1])
    rel = r * theta.cos()
    iml = r * theta.sin()
    Koopmat = torch.cat(
        [
            torch.cat([rel, iml], dim=-1),
            torch.cat([-iml, rel], dim=-1),
        ],
        dim=-2,
    )
    Koopmat.unsqueeze_(0)
    print(Koopmat.shape)
    Koopmat_ = Koopmat.tile(
        N, *([1] * (len(Koopmat.shape) - 1))
    )  # vectorized Koopman operator
    print(Koopmat_.shape)

    xphi = torch.randn([N, N, dimH, 2])
    for k in range(L):
        xphi = torch.matmul(xphi, Koopmat_)
        print(xphi.shape)

    pass


if __name__ == "__main__":
    main()
