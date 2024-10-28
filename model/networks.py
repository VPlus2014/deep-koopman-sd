from copy import deepcopy
from typing import Dict, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .listmodule import ListModule
from .torch_rbf import *


def get_encoder(params: Dict, shape: List[int], name="enc"):
    # returns encoder given its configuration and shape
    use_rbf = params.get("use_rbf") or False
    kernel = params.get("kernel") or None
    config = {"shape": shape, "use_rbf": use_rbf, "kernel": kernel}
    return Encoder(config, name)


class HReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1 + x.square()).sqrt() + x


supported_activation_dict = {
    "sigmoid": F.sigmoid,
    "relu": F.relu,
    "tanh": torch.tanh,
    "leaky_relu": F.leaky_relu,
    "elu": F.elu,
    "selu": F.selu,
    "softplus": F.softplus,
    "hswish": F.hardswish,
    "prelu": nn.PReLU(),
    "hrelu": HReLU(),
}


class Encoder(nn.Module):
    """
    General block for lifting the state-space dimension:
    y = h(x), where Dim(y)>Dim(x)

    !!! last affine layer has no bias

    Arguments:
        params: configuration dictionary
                requires:
                    "shape": (list of int)
                optional"
                    "drop_prob": (float) dropout probability
                    "use_rbf": (Bool) whether to use RBF as first layer
                    "kernel": type of kernel for RBF layer}

        name:   (str) encoder name
    """

    def __init__(self, params: Dict, name="encoder", activation: str = None):
        super().__init__()
        self.params = params
        self.shape = params["shape"]
        self.dimX = self.shape[0]
        self.dimY = self.shape[-1]
        self.drop_prob = params.get("drop_prob", None) or 0.0

        if activation is None:
            activation = "leaky_relu"
        activation = activation.lower()
        self.activation = supported_activation_dict.get(activation)

        self.dropout = nn.Dropout(p=self.drop_prob)

        # Can use a rbf kernel for the first layer
        self.use_rbf = params.get("use_rbf", None) or False
        if self.use_rbf:
            kernel_type = params.get("kernel", None) or "gaussian"
            str2base_func = basis_func_dict()
            kernel = str2base_func[kernel_type]
            self.rbf = RBF(self.dimX, self.shape[1], kernel)
            self.shape = self.shape[1:]

        # the (rest) of the layers are linear
        self.aux_layers = ListModule(self, "{}_".format(name))

        self.input_bn = nn.BatchNorm1d(self.dimX)
        self.output_scale = nn.Parameter(torch.ones(self.dimY))

        for j in range(len(self.shape) - 1):
            bias = bool(j + 2 != len(self.shape))  # last affine layer has no bias
            self.aux_layers.append(
                nn.Linear(self.shape[j], self.shape[j + 1], bias=bias)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp0 = x.shape[:-1]
        x = self.input_bn(x.view(-1, x.shape[-1]))
        x = x.view(*shp0, -1)

        if self.use_rbf:
            if len(x.size()) == 3:
                # handle 3D data (batch, timestep, dim)
                N, T, D = x.size()
                x = self.rbf(x.contiguous().view((N * T, D)))
                x = x.contiguous().view((N, T, x.shape[-1]))
            else:
                x = self.rbf(x)
            x = F.relu(x)

        for idx, layer in enumerate(self.aux_layers):
            x = layer(x)
            if idx <= len(self.aux_layers) - 2:
                x = self.activation(x)
            else:
                # only do dropout on the last layer
                x = self.dropout(x)

        x = x * self.output_scale.view(*([1] * (len(x.shape) - 1)), -1)
        return x


def xu_shape_valid(xs: torch.Tensor, us: torch.Tensor, axis_t=-2) -> bool:
    return xs.shape[axis_t] == us.shape[axis_t] + 1


class LREN(nn.Module):
    """
    Linearly Recurrent Encoder Network
    y_n = [x_n, g(x_n)]^T
    y_{n+1} = K y_n

    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_x_shape': (list of int),
                    'n_shifts': number of time shifts
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.params = params
        enc_x_shape = params["enc_x_shape"]
        self.dim = enc_x_shape[0]

        self.encoder = get_encoder(params, enc_x_shape, "enc_x")
        self.koopman = nn.Linear(
            enc_x_shape[-1] + self.dim,
            enc_x_shape[-1] + self.dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor, n_shifts: int = None, return_ko=False):
        if n_shifts is None:
            n_shifts = self.params["n_shifts"]
        assert n_shifts > 1, "expected n_shifts>1"
        assert x.shape[1] >= n_shifts, (
            f"expected x.shape[1]>={n_shifts}, got",
            x.shape[1],
        )

        # generate ground truth
        x = x[:, :n_shifts, :]
        y = self.encoder(x)
        # append observable state
        y = torch.cat((x, y), dim=-1)
        y_targ = y[:, 1:, :]

        # generate trajectories from initial state
        y1 = y[:, 0:1, :].clone()
        y_pred = []
        for i in range(n_shifts - 1):
            y_next = self.koopman(y1)
            y1 = y_next
            y_pred.append(y_next)
        y_pred = torch.cat(y_pred, axis=1)
        assert isinstance(y_pred, torch.Tensor)

        if return_ko:
            koopman = self.form_koopman(x)
            # koopman: np.ndarray = torch.Tensor(list(koopman)[0]).numpy()
            return y_targ, y_pred, koopman
        else:
            return y_targ, y_pred

    def form_koopman(self, x: torch.Tensor):
        """K^T"""
        # Generate parameterized Koopman operator
        return next(self.koopman.parameters()).data.T

    @property
    def dtype(self):
        return self.koopman.weight.dtype

    @property
    def device(self):
        return self.koopman.weight.device

    def predict(self, x: np.ndarray, n_shifts: int, return_ko=False):
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=self.dtype)
            y, y_pred, koopman = self.forward(x, n_shifts, True)
            x_pred: np.ndarray = y_pred.cpu().numpy()[:, :, : x.shape[-1]]
        if return_ko:
            return x_pred, koopman
        return x_pred

    def encode(self, x: torch.Tensor):
        isTensor = torch.is_tensor(x)
        if not isTensor:
            x = torch.tensor(x, dtype=self.dtype)
        with torch.no_grad():
            # Generate latent ground truth
            x = x[:, : self.params["n_shifts"], :]
            y = self.encoder.forward(x)
            y = torch.cat((x, y), dim=-1)
            assert isinstance(y, torch.Tensor)
            if not isTensor:
                y: np.ndarray = y.numpy()
        return y


class DENIS(nn.Module):
    """
    Deep Encoder Network with Initial State parameterisation
    y_n = [x_n, g(x_n)]^T
    K = h(x_0)
    y_{n+1} = K y_n

    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_x_shape': (list of int),
                    'aux_shape': (list of int),
                    'n_shifts': number of time shifts
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

        self.dim = params["enc_x_shape"][0]
        self.ldim = params["aux_shape"][-1]

        enc_x_shape = params["enc_x_shape"]
        aux_shape = deepcopy(params["aux_shape"])
        self.aux_rbf = params.get("aux_rbf", None) or False

        assert (
            aux_shape[-1] == enc_x_shape[0] + enc_x_shape[-1]
        ), "Invalid auxiliary network's output shape"
        assert aux_shape[0] == enc_x_shape[0], "Input state dimensions must match"

        aux_shape[-1] = aux_shape[-1] * aux_shape[-1]

        self.encoder = get_encoder(params, enc_x_shape, "enc_x")

        aux_params = params.copy()
        aux_params["use_rbf"] = self.aux_rbf
        self.aux_net = get_encoder(aux_params, aux_shape, "aux_")

    @property
    def dtype(self):
        return self.encoder.parameters().__next__().dtype

    def encode(self, x: torch.Tensor):
        isTensor = torch.is_tensor(x)
        if not isTensor:
            x = torch.tensor(x, dtype=self.dtype)
        with torch.no_grad():
            # Generate latent ground truth
            x = x[:, : self.params["n_shifts"], :]
            y = self.encoder.forward(x)
            y = torch.cat((x, y), dim=-1)
            assert isinstance(y, torch.Tensor)
            if not isTensor:
                y: np.ndarray = y.numpy()
        return y

    def form_koopman(self, x: torch.Tensor):
        """K^T"""
        # Generate parameterized Koopman operator
        x0 = x[:, 0, :]
        koopman = self.aux_net.forward(x0)
        koopman = koopman.view((koopman.shape[0], self.ldim, self.ldim))
        return koopman

    def forward(self, x: torch.Tensor, n_shifts: int = None, return_ko=False):
        if n_shifts is None:
            n_shifts = self.params["n_shifts"]
        assert n_shifts > 1, "expected n_shifts>1"
        assert x.shape[1] >= n_shifts, (
            f"expected x.shape[1]>={n_shifts}, got",
            x.shape[1],
        )
        koopman = self.form_koopman(x)
        ys = self.encode(x)  # (B,n_shifts,ldim)
        y_targ = ys[:, 1:, :]
        # generate trajectories from initial state
        y_pred = []
        y1 = ys[:, 0:1, :]
        for i in range(n_shifts - 1):
            y_next = torch.bmm(y1, koopman)
            y1 = y_next
            y_pred.append(y_next)
        y_pred = torch.cat(y_pred, axis=1)
        assert isinstance(y_pred, torch.Tensor)
        if return_ko:
            return y_targ, y_pred, koopman
        else:
            return y_targ, y_pred

    def predict(self, x: torch.Tensor, n_shifts: int, return_ko=False):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y, y_pred, koopman = self.forward(x, n_shifts, True)
            x_pred = y_pred.cpu().numpy()[:, :, : x.shape[-1]]
            koopman = koopman.cpu().numpy()
        if return_ko:
            return x_pred, koopman
        return x_pred


class DENIS_JBF(nn.Module):
    """
    Deep Encoder Network with Initial State parameterisation
    In Jordon block diagonal form, i.e. direct eigenfunction learning
    y_n = g(x_n)^T
    B_n = h_n(x_0)
    K = [[B_1, 0, ..., 0], [0, B_2, ...0], [0, ..., B_N]]
    y_{n+1} = K y_n
    x_{n+1} = C y_{n+1}

    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_x_shape': (list of int),
                    'aux_shape': (list of int),
                    'n_shifts': number of time shifts
                    'dt': sampling time
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
                    'aux_rbf': (bool), whether aux nets use RBF layers

    """

    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

        self.dim = params["enc_x_shape"][0]
        self.ldim = params["enc_x_shape"][-1]
        self.dt = params["dt"]
        self.aux_rbf = params.get("aux_rbf", None) or False

        enc_x_shape = params["enc_x_shape"]
        aux_shape = deepcopy(params["aux_shape"])

        self.n_aux = int(enc_x_shape[-1] // 2)
        self.koopman = None

        assert aux_shape[0] == enc_x_shape[0], "Invalid state dimensions"
        assert aux_shape[-1] == 2, "Each aux net must output omega and mu"
        assert self.ldim % 2 == 0, "Encoder output shape must be even"

        # initialize encoder network
        self.encoder = get_encoder(params, enc_x_shape, "enc_x")

        # initialize auxiliary networks (each one represent one Jacobian block)
        self.aux_nets = ListModule(self, "auxNets_")
        # option to not to use rbf for aux nets
        enc_params = params.copy()
        enc_params["use_rbf"] = self.aux_rbf
        for k in range(self.n_aux):
            self.aux_nets.append(get_encoder(enc_params, aux_shape, "auc_"))

        # initialize C matrix for casting to observed state
        self.C = nn.Linear(self.ldim, self.dim, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def block_diag(self, m):
        def attach_dim(v: torch.Tensor, n_dim_to_prepend=0, n_dim_to_append=0):
            """oldshape-> (1,1,...,1,*old_shape,1,1,...,1)"""
            return v.view(
                torch.Size([1] * n_dim_to_prepend)
                + v.shape
                + torch.Size([1] * n_dim_to_append)
            )

        if type(m) is list:
            m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

        d = m.dim()
        n = m.shape[-3]
        siz0 = m.shape[:-3]
        siz1 = m.shape[-2:]
        m2 = m.unsqueeze(-2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        eye = attach_dim(torch.eye(n).unsqueeze(-2).to(device), d - 3, 1)
        return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))

    def forward(self, xs: torch.Tensor, n_shifts=None, return_ko=False):
        # Generate Koopman operator (block parameters)
        x0 = xs[:, 0, :].clone()  # why clone?
        aux_out = torch.empty((x0.shape[0], self.n_aux, 2), device=x0.device)
        for idx, net in enumerate(self.aux_nets):
            aux_out[:, idx, :] = net(x0)

        # Form Koopman operator from Jordon blocks
        scale = torch.exp(aux_out[:, :, 0] * self.dt)
        cost = torch.cos(aux_out[:, :, 1] * self.dt) * scale
        sint = torch.sin(aux_out[:, :, 1] * self.dt) * scale
        row1 = torch.stack((cost, sint), dim=2)
        row2 = torch.stack((-sint, cost), dim=2)
        koopman = torch.stack((row1, row2), dim=2)
        koopman = list(map(self.block_diag, torch.unbind(koopman, 0)))
        koopman = torch.stack(koopman, 0)
        self.koopman = koopman

        # generate ground truth
        n_shifts = self.params["n_shifts"]
        xs = xs[:, :n_shifts, :]
        y = self.encoder(xs)
        y = torch.cat((xs, y), dim=-1)

        # generate trajectories from initial state
        y_pred = y[:, 0:1, self.dim :].clone()
        x_pred = y[:, 0:1, : self.dim].clone()

        for i in range(n_shifts - 1):
            y_next = torch.bmm(y_pred[:, -1:, :], koopman)
            y_pred = torch.cat((y_pred, y_next), dim=1)
            x_next = self.C(y_next)
            x_pred = torch.cat((x_pred, x_next), dim=1)

        y_pred = torch.cat((x_pred, y_pred), dim=2)

        if return_ko:
            return y, y_pred, koopman
        else:
            return y, y_pred


class DEINA(nn.Module):
    """
    Deep Encoder for Input Non-Affine Systems
    y_n = [x_n, g(x_n)]^T
    y_{n+1} = K y_n + B u_n
    or
    y_{n+1} = K y_n + B [u_n + h(u_n)+]^T

    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_x_shape': (list of int),
                    'n_shifts': number of time shifts
                optional:
                    'use_rbf': (bool),
                    'use_enc_u', (bool) whether to add
                                 non-linearities in u
                    'enc_u_shape': (list of int),
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
                    'enc_rbf': (bool), whether  use RBF layers
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

        self.enc_x_shape = params["enc_x_shape"]
        self.use_enc_u = params.get("use_enc_u", None) or False

        self.dim = self.enc_x_shape[0]
        self.ldim = self.enc_x_shape[-1] + self.dim

        self.encoder_x = get_encoder(params, self.enc_x_shape, "enc_x")

        # option to add non-linearities in u
        if self.use_enc_u:
            self.enc_u_shape = params["enc_u_shape"]
            # option to add a rbf layer
            enc_params = params.copy()
            enc_params["use_rbf"] = params.get("enc_rbf", None) or False
            self.encoder_u = get_encoder(enc_params, self.enc_u_shape, "enc_u")
            self.B = nn.Linear(self.enc_u_shape[-1] + self.dim, self.ldim, bias=False)
            # B*[u; enc(u)]
        else:
            self.B = nn.Linear(self.dim, self.ldim, bias=False)

        self.koopman = nn.Linear(self.ldim, self.ldim, bias=False)

    def forward(self, xs: torch.Tensor, us: torch.Tensor, return_ko=False):
        n_shifts = self.params["n_shifts"]
        assert xs.shape[1] >= n_shifts, (
            f"expected x.shape[1]>={n_shifts}, got",
            xs.shape[1],
        )
        assert us.shape[1] + 1 >= n_shifts, (
            f"expected us.shape[1]>={n_shifts}-1, got",
            us.shape[1],
        )
        # generate latent ground truth
        x1s = xs[:, :n_shifts, :]
        y1s = self.encoder_x(x1s)
        y1s = torch.cat((x1s, y1s), dim=-1)
        y2s = y1s[:, 1:]

        # generate encoded inputs
        us = us[:, : n_shifts - 1, :]
        if self.use_enc_u:
            v = self.encoder_u(us)
            us = torch.cat((us, v), dim=-1)
        Bu: torch.Tensor = self.B(us)

        # generate predicted trajectories
        y2s_pred = []
        y1 = y1s[:, 0:1, :]
        for i in range(n_shifts - 1):
            y2 = self.koopman(y1) + Bu[:, i : i + 1, :]
            y1 = y2
            y2s_pred.append(y2)
        y2s_pred = torch.cat(y2s_pred, dim=1)

        if return_ko:
            koopman = self.koopman.parameters()
            koopman = torch.Tensor(list(koopman)[0])
            return y2s, y2s_pred, koopman.detach()
        else:
            return y2s, y2s_pred


class DEINA_SD(nn.Module):
    r"""
    Deep Encoder for Input Affine Systems in Spectral Decomposition

    g(x,u) &= \sum_i [g(x,u)]_i^\Phi v^i\\
    &= \sum_i [g_i(x)+[A_i(x)+B_i]*[u;h(u)]] v^i\\
    
    \Phi:=(v^i)_{i\geq 1} is unkown linearly independent eigenvectors
    
    (K\odot g)(x,u) = \sum_i [\lambda_i g_i(x)] v^i\\
    
    \hat{x}=C g(x)
    \hat{y}=[x;g(x)]

    g: encoder for x
    h: encoder for u s.t. h(0)=0
    B: linear operator for u
    A: non-linear operator for u

    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_x_shape': (list of int),
                    'aux_shape': (list of int),
                    'n_shifts': number of time shifts
                    'dt': sampling time
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
                    'aux_rbf': (bool), whether aux nets use RBF layers

    """

    def __init__(self, params: Dict):
        super().__init__()
        params = self.params = deepcopy(params)

        enc_x_shape: List[int] = params["enc_x_shape"]
        enc_u_shape: List[int] = params["enc_u_shape"]
        aux_shape: List[int] = params["aux_shape"]
        self.dimX = enc_x_shape[0]
        self.dimU = enc_u_shape[0]
        self.dimH = enc_x_shape[-1]
        self.use_enc_u = params.get("use_enc_u", None) or False
        self.use_complex = params.get("use_complex", None) or False
        self.use_affine = params.get("use_affine", None) or False

        # X encoder
        if self.use_complex:
            assert (
                self.dimH % 2 == 0
            ), f"complex X encoder output shape must be even, but got {self.dimH}"
        self.encoder_x = get_encoder(params, enc_x_shape, "enc_x")

        # U encoder
        dimUh = self.dimU
        if self.use_enc_u:
            dimUh = dimUh + enc_u_shape[-1]
            # option to add a rbf layer
            enc_u_params = deepcopy(params)
            enc_u_params["use_rbf"] = params.get("enc_rbf", False)
            self.encoder_u = get_encoder(enc_u_params, enc_u_shape, "enc_u")
        else:
            self.encoder_u = None

        # nonlinear affine operator
        # g(x,u)=g(x) + (A(x)+B)*[\psi(u)-\psi(0)]
        if self.use_affine:
            assert len(aux_shape) >= 2
            # auto IO size
            aux_shape[0] = self.dimX + self.dimH
            aux_shape[-1] = self.dimH * dimUh
            aux_params = deepcopy(params)
            aux_params["use_rbf"] = params.get("enc_rbf", None) or False
            self.aux_net = get_encoder(aux_params, aux_shape, "affine_x")
        else:
            self.aux_net = None
        self.B = nn.Linear(dimUh, self.dimH, bias=False)

        self.dt = params["dt"]
        self.koopman = None

        # observation operator
        dimY = self.dimX
        self.C = nn.Linear(self.dimH, dimY, bias=False)

        # Koopman eigenvalue parameters
        self.n_eig = (self.dimH // 2) if self.use_complex else self.dimH
        if self.use_complex:
            lshape = [self.n_eig, 1, 1]
        else:
            lshape = [self.n_eig]
        self.reL = nn.Parameter(torch.zeros(lshape))
        if self.use_complex:
            self.imL = nn.Parameter(torch.zeros(lshape))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        lam0lb = 0.5
        lam0ub = 1 + 1e-2
        lnorm = torch.clip(
            torch.normal(
                (lam0lb + lam0ub) / 2, (lam0ub - lam0lb) / 6, size=self.reL.shape
            ),
            lam0lb,
            lam0ub,
        )
        if self.use_complex:
            thetas = torch.randn_like(self.reL) * (2 * torch.pi)
            self.reL.data.copy_(lnorm * torch.cos(thetas))
            self.imL.data.copy_(lnorm * torch.sin(thetas))
        else:
            self.reL.data.copy_(lnorm)
        # nn.init.uniform_(self.reL, -2,1)
        # nn.init.uniform_(self.imL, 0, 2 * torch.pi)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def form_koopman(self):
        """K"""
        use_cplx = False
        if use_cplx:
            # lnorm = self.reL.exp()
            # relam = torch.cos(self.imL) * lnorm
            # imlam = torch.sin(self.imL) * lnorm
            relam = self.reL
            imlam = self.imL
            Koopmat = torch.cat(
                [
                    torch.cat([relam, -imlam], dim=-1),
                    torch.cat([imlam, relam], dim=-1),
                ],
                dim=-2,
            )  # (D,2,2)
        else:
            Koopmat = self.reL  # (D,)
        return Koopmat

    def _Phi2phi(self, phi: torch.Tensor):
        return phi.sum(-3).flatten(-2, -1)

    def forward(self, xs: torch.Tensor, us: torch.Tensor, return_ko=False):
        n_shifts = self.params["n_shifts"]
        assert xs.shape[1] >= n_shifts, (
            f"expected xs.shape[1]>={n_shifts}, got",
            xs.shape[1],
        )
        assert us.shape[1] + 1 >= n_shifts, (
            f"expected us.shape[1]>={n_shifts}-1, got",
            us.shape[1],
        )
        xs = xs[:, :n_shifts]  # (N,L+1,dimX)
        us = us[:, : n_shifts - 1]  # (N,L,dimU)
        N = xs.shape[0]
        L = xs.shape[1] - 1
        dimH = self.dimH  # D
        n_eig = self.n_eig  # M

        phis: torch.Tensor = self.encoder_x(xs)  # (N,L+1,D)
        phi1s = phis[:, :-1]  # (N,L,D)
        g2s_t = phis[:, 1:]

        if self.use_enc_u:
            uzero = torch.zeros_like(us)
            us_ = self.encoder_u(us) - self.encoder_u(uzero)
            us_ = torch.cat((us, us_), dim=-1)  # (N,L,DU')
        else:
            us_ = us
        # (N,L,DU')

        Bu: torch.Tensor = self.B(us_)
        g1s = phi1s + Bu  # (N,L,D)
        if self.use_affine:
            # 这段用于计算依赖于状态的仿射项 A(x)@u ， bmm : eisum 耗时约 1:2
            Ax: torch.Tensor = self.aux_net(xs[:, :-1])  # (N,L,D*DU')
            Ax = Ax.view(-1, dimH, us_.shape[-1])  # (N*L,D,DU')
            Axu = torch.bmm(Ax, us_.view(-1, us_.shape[-1], 1))  # -> (N*L,D,1)
            Axu = Axu.view(Bu.shape)  # (N,L,D)
            g1s = g1s + Axu

        # Generate Koopman operator
        kop = self.form_koopman()  # (D,2,2) | (D,)

        # Koopman transform
        shp_NL = g1s.shape[:-1]
        if self.use_complex:
            # 利用 X 已知, 张量版:for循环 耗时 1:40
            kop = kop.repeat(N * L, 1, 1)  # (N*L*M,2,2)
            g1s = g1s.view(-1, 2, 1)  # (N*L*M,1,2)
            g2s_p = torch.bmm(kop, g1s)  # (N*L*M,1,2)
            g2s_p = g2s_p.view(*shp_NL, dimH)  # (N*L,M*2)
            assert g2s_p.shape == g2s_t.shape, (g2s_p.shape, g2s_t.shape)
            print(f"g2s_p.shape: {g2s_p.shape}")
        else:
            # 广播乘:对角阵乘 耗时 1:5
            kop = kop.view(*([1] * len(shp_NL)), n_eig)  # (1,1,D)
            g2s_p = g1s * kop

        # observation operator
        x2_p = self.C(g2s_p)  # (N,L,dimX)

        y2_p = torch.cat((x2_p, g2s_p), dim=-1)  # (N,L,dimX+dimH)
        y2_t = torch.cat((xs[:, 1:], g2s_t), dim=-1)  # (N,L,dimX+dimH)
        if return_ko:
            with torch.no_grad():
                if self.use_complex:
                    kmat = (self.reL + 1j * self.imL).view(-1).diag()
                else:
                    kmat = self.reL.view(-1).diag()
                kmat = kmat.cpu().numpy()
            return y2_t, y2_p, kmat
        else:
            return (
                y2_t,
                y2_p,
            )


class DEINA_JBF(nn.Module):
    """
    Deep Encoder for Input Non-Affine Systems
    y_n = [phi(x_n)+psi(u_n)-psi(u_n)]^T
    y_{n+1} = K y_n

    In Jordon block diagonal form, i.e. direct eigenfunction learning
    y_n = g(x_n)^T
    B_n = h_n(x_0)
    K = [[B_1, 0, ..., 0], [0, B_2, ...0], [0, ..., B_N]]
    y_{n+1} = K y_n
    x_{n+1} = C y_{n+1}

    Arguments:
        params: configuration dictionary,
                requires:
                    'enc_x_shape': (list of int),
                    'aux_shape': (list of int),
                    'n_shifts': number of time shifts
                    'dt': sampling time
                optional:
                    'use_rbf': (bool)
                    'kernel'(str): type of RBF kernel
                    'drop_prob': (float)
                    'return_ko': (bool)
                    'aux_rbf': (bool), whether aux nets use RBF layers

    """

    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

        enc_x_shape: List[int] = deepcopy(params["enc_x_shape"])
        self.n_aux = int(
            enc_x_shape[-1] // 2
        )  # 复值特征函数个数 M, 依照线性无关假设, 它们在任意正交基下的坐标维度应不低于 M
        self.dim = enc_x_shape[0]
        self.ldim = enc_x_shape[-1]
        assert self.ldim % 2 == 0, "X Encoder output shape must be even"

        enc_u_shape: List[int] = params["enc_u_shape"]
        self.dimU = enc_u_shape[0]
        assert (
            enc_u_shape[-1] == self.ldim
        ), "U Encoder output shape must be the same as X Encoder output shape"

        self.use_enc_u = params.get("use_enc_u", None) or False
        if self.use_enc_u:
            self.enc_u_shape = params["enc_u_shape"]
            # option to add a rbf layer
            enc_params = params.copy()
            enc_params["use_rbf"] = params.get("enc_rbf", None) or False
            self.encoder_u = get_encoder(enc_params, self.enc_u_shape, "enc_u")

            self.B = nn.Linear(self.dimU + self.enc_u_shape[-1], self.ldim, bias=False)
        else:
            self.B = nn.Linear(self.dimU, self.ldim, bias=False)

        self.dt = params["dt"]
        self.aux_rbf = params.get("aux_rbf", None) or False

        # enc_x_shape = params["enc_x_shape"]

        self.koopman = None

        self.aux_dout = enc_x_shape[-1]

        self.reL = nn.Parameter(torch.zeros(1, self.n_aux, 1, 1))
        self.imL = nn.Parameter(torch.zeros_like(self.reL))

        # initialize encoder network
        # self.encoder = get_encoder(params, enc_x_shape, "enc_x")

        # initialize auxiliary networks (each one represent one Jacobian block)
        self.x_aux_nets = ListModule(self, "XAuxNets_")
        self.u_aux_nets = ListModule(self, "UAuxNets_")
        # option to not to use rbf for aux nets
        enc_params = params.copy()
        enc_params["use_rbf"] = self.aux_rbf
        for k in range(self.n_aux):
            self.x_aux_nets.append(get_encoder(enc_params, enc_x_shape, "auc_"))
            self.u_aux_nets.append(get_encoder(enc_params, enc_u_shape, "auc_"))

        # initialize C matrix for casting to observed state
        self.C = nn.Linear(self.ldim, self.dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        lam0lb = 0.8
        lam0ub = 1.1
        lnorm = torch.clip(
            torch.normal(
                (lam0lb + lam0ub) / 2, (lam0ub - lam0lb) / 6, size=self.reL.shape
            ),
            lam0lb,
            lam0ub,
        )
        thetas = torch.randn_like(self.reL) * (2 * torch.pi)

        self.reL.copy_(lnorm * torch.cos(thetas))
        self.imL.copy_(lnorm * torch.sin(thetas))
        # nn.init.uniform_(self.reL, -2,1)
        # nn.init.uniform_(self.imL, 0, 2 * torch.pi)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def form_koopman(self):
        """"""
        # lnorm = self.reL.exp()
        # relam = torch.cos(self.imL) * lnorm
        # imlam = torch.sin(self.imL) * lnorm
        relam = self.reL
        imlam = self.imL
        Koopmat = torch.cat(
            [
                torch.cat([relam, imlam], dim=-1),
                torch.cat([-imlam, relam], dim=-1),
            ],
            dim=-2,
        )  # (1,M,2,2)
        return Koopmat

    def _Phi2phi(self, phi: torch.Tensor):
        return phi.sum(-3).flatten(-2, -1)

    def forward(self, xs: torch.Tensor, us: torch.Tensor, return_ko=False):
        n_shifts = self.params["n_shifts"]
        assert xs.shape[1] >= n_shifts, (
            f"expected xs.shape[1]>={n_shifts}, got",
            xs.shape[1],
        )
        assert us.shape[1] + 1 >= n_shifts, (
            f"expected us.shape[1]>={n_shifts}-1, got",
            us.shape[1],
        )
        xs = xs[:, :n_shifts]
        us = us[:, : n_shifts - 1]
        n = xs.shape[0]
        uzero = torch.zeros_like(us)  # (N,1,dimU)

        phi1s = []
        for i, net in enumerate(self.x_aux_nets):
            phi_x_i: torch.Tensor = net(xs)  # (N,L+1,M*2)
            phi_x_i.unsqueeze_(-2)  # (N,L+1,1,M*2)
            phi1s.append(phi_x_i)
        phi1s = torch.cat(phi1s, dim=-2)  # (N,L,M,M*2)
        phi1s = phi1s.view(*phi1s.shape[:-1], self.n_aux, 2)  # (N,L+1,M,M,2)

        ures = []
        for i, net in enumerate(self.u_aux_nets):
            psi_u_i: torch.Tensor = net(us) - net(uzero)  # (N,L,M*2)
            psi_u_i.unsqueeze_(-2)  # (N,L,1,M*2)
            ures.append(psi_u_i)
        ures = torch.cat(ures, dim=-2)
        ures = ures.view(*ures.shape[:-1], self.n_aux, 2)  # (N,L,M,M,2)

        # Generate Koopman operator (block parameters)
        koopmat = self.form_koopman()  # (1,M,2,2)
        koopmat = koopmat.repeat(n, 1, 1, 1)  # (N,M,2,2)
        koopmat = koopmat.flatten(0, 1)  # (N*M,2,2)

        # generate trajectories from initial state
        y_pred = []
        phi1 = phi1s[:, 0]  # (N,M,M,2)
        for i in range(n_shifts - 1):
            phi1 = phi1 + ures[:, i]  # (N,M,M,2)
            phi1 = phi1.flatten(0, 1)
            phi2 = torch.bmm(phi1, koopmat)  # (N*M,M,2)
            phi2 = phi2.view(n, self.n_aux, self.n_aux, 2)  # (N,M,M,2)
            phi1 = phi2
            # print(phi2.abs().max())
            y_pred.append(phi2)

        y_pred = torch.stack(y_pred, dim=1)  # (N,L,M,M,2)
        y_pred = self._Phi2phi(y_pred)  # (N,L,M*2)
        y_pred = self.C(y_pred)  # (N,L,dimY)

        # generate latent ground truth
        y = phi1s[:, 1:]  # (N,L,M,M,2)
        y = self._Phi2phi(y)  # (N,L,M*2)
        y = self.C(y)  # (N,L,dimY)

        if return_ko:
            return y, y_pred, koopmat
        else:
            return y, y_pred
