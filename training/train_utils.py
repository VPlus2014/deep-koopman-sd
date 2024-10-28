from dataclasses import dataclass
import math
from pathlib import Path
import traceback
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from .config_protol import RunConfig


def count_parameters(net: nn.Module):
    from functools import reduce
    from collections import defaultdict
    import pandas as pd

    p = list(net.parameters())
    total = sum([reduce(lambda x, y: x * y, i.shape) for i in p])

    inp = [
        (i[0].split("_")[0], reduce(lambda x, y: x * y, i[1].shape))
        for i in list(net.named_parameters())
    ]
    output = defaultdict(int)
    for k, v in inp:
        output[k] += v

    df = pd.DataFrame(dict(output), index=[0])
    df["total"] = total
    return df


def affine_model_configurer(config: dict):
    model_config = {
        "enc_x_shape": config["enc_x_shape"],
        "aux_shape": config.get("aux_shape"),
        "n_shifts": config["n_shifts"],
        "dt": config.get("dt"),
        "use_rbf": config.get("use_rbf"),
        "drop_prob": config.get("drop_prob"),
        "aux_rbf": config.get("aux_rbf"),
        "activation": config.get("activation", None),
    }
    return model_config


def non_affine_model_configurer(config: dict):
    model_config = {
        "enc_x_shape": config["enc_x_shape"],
        "enc_u_shape": config.get("enc_u_shape"),
        "aux_shape": config.get("aux_shape"),
        "n_shifts": config["n_shifts"],
        "dt": config.get("dt"),
        "use_rbf": config.get("use_rbf"),
        "enc_rbf": config.get("enc_rbf"),
        "drop_prob": config.get("drop_prob"),
        "use_enc_u": config.get("use_enc_u"),
        "activation": config.get("activation", None),
    }
    return model_config


def get_lr_scheduler(optimizer, config: RunConfig):
    step = config.lr_sch_step
    gamma = config.lr_sch_gamma
    return optim.lr_scheduler.StepLR(optimizer, step, gamma=gamma)


def get_batch(data: torch.Tensor, btach_size: int, config: RunConfig):
    batch = data[torch.randint(len(data), size=(btach_size,))]
    if config.lw_zero != 0.0:
        zero = torch.zeros((1, *batch.shape[1:]))
        batch = torch.cat((zero, batch), dim=0)
    return batch


def get_l2_reg(model: nn.Module, w_bound: float = 1.0) -> torch.Tensor:
    w_bound = w_bound**2
    assert w_bound >= 0.0
    l_reg = None
    for w in model.parameters():
        wl = w.square()
        if w_bound:
            wl = F.relu(wl - w_bound)
        wl = wl.sum()
        l_reg = wl if l_reg is None else (l_reg + wl)
    l_reg = 0.5 * l_reg
    assert isinstance(l_reg, torch.Tensor)
    return l_reg


def get_l1_reg(model: nn.Module, w_bound: float = 1.0) -> torch.Tensor:
    assert w_bound >= 0.0
    l_reg = None
    for w in model.parameters():
        wl = w.abs()
        if w_bound:
            wl = F.relu(wl - w_bound)
        wl = wl.sum()
        l_reg = wl if l_reg is None else (l_reg + wl)
    assert isinstance(l_reg, torch.Tensor)
    return l_reg


@dataclass
class LossMetrics:
    loss_no_w_reg: float
    loss_w_reg: float

    loss_state: float
    loss_latent: float
    loss_x_inf_2: float
    loss_h_inf_2: float

    loss_w_l1: float
    loss_w_l2: float


K_train_loss = "train_loss"
K_train_reg = "train_reg"
K_train_state_mse = "train_state_mse"
K_train_latent_mse = "train_latent_mse"
K_train_state_maxtse = "train_state_max_tse"
K_train_latent_maxtse = "train_latent_max_tse"
K_val_loss = "val_loss"
K_val_reg = "val_reg"
K_val_state_mse = "val_state_mse"
K_val_latent_mse = "val_latent_mse"
K_val_state_maxtse = "val_state_max_tse"
K_val_latent_maxtse = "val_latent_max_tse"


class TinySummrayWriter:
    def __init__(self, fname: str, model_name: str):
        self._fn = Path(fname).resolve()
        df = pd.DataFrame(
            columns=[
                "model_name",
                "iter",
                "lr",
                #
                K_train_loss,
                K_train_reg,
                K_train_state_mse,
                K_train_latent_mse,
                K_train_state_maxtse,
                K_train_latent_maxtse,
                #
                K_val_loss,
                K_val_reg,
                K_val_state_mse,
                K_val_latent_mse,
                K_val_state_maxtse,
                K_val_latent_maxtse,
            ]
        )
        if self._fn.exists():  # merge
            df_ = pd.read_csv(fname)
            try:
                df = pd.concat([df, df_], ignore_index=True)
            except Exception as e:
                print(e, traceback.format_exc())
        self._df = df
        self.model_name = model_name

    def push(
        self,
        train_vals: List[float],
        val_vals: List[float],
        lr: float,
        to_file=True,
    ):
        model_name = self.model_name
        df = self._df
        idxs = df["model_name"] == model_name
        itr_last = -1
        if idxs.any():
            itr_last = df[idxs]["iter"].max()

        df.loc[len(df) + 1] = [model_name, itr_last + 1, lr, *train_vals, *val_vals]
        if to_file:
            self._fn.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self._fn, index=False)

    def filter_by_model(self):
        df = self._df
        return df[df["model_name"] == self.model_name]

    def query_min(self, key="val_loss"):
        df = self.filter_by_model()
        df = df[key]
        if df.empty:
            min_val = math.inf
        else:
            min_val = float(df.min())
        return min_val


def koopman_loss(
    xh_pred: torch.Tensor,
    xh_target: torch.Tensor,
    config: RunConfig,
    model,
):
    dim = config.enc_x_shape[0]
    x_targ = xh_target[:, :, :dim]
    x_pred = xh_pred[:, :, :dim]
    h_targ = xh_target[:, :, dim:]
    h_pred = xh_pred[:, :, dim:]
    # if config.lw_zero != 0.0:
    #     zero_loss = torch.mean((xh_target[0] - xh_pred[0]) ** 2)
    #     idx = 1
    #     x_targ = x_targ[idx:]
    #     x_pred = x_pred[idx:]
    #     h_targ = h_targ[idx:]
    #     h_pred = h_pred[idx:]
    # else:
    #     zero_loss = torch.zeros(1)

    if config.use_huber:
        # nn.SmoothL1Loss
        floss = F.smooth_l1_loss
    else:
        # nn.MSELoss
        floss = F.mse_loss
    sse_x = floss(x_pred, x_targ, reduction="none").sum(dim=-1)  # (N,L)
    sse_h = floss(h_pred, h_targ, reduction="none").sum(dim=-1)  # (N,L)

    x_mse = sse_x.mean()
    h_mse = sse_h.mean()

    tspan_sup = min(10000, x_targ.shape[1])
    Lx_inf_2 = sse_x[:, :tspan_sup].max(-1)[0].mean()
    Lh_inf_2 = sse_h[:, :tspan_sup].max(-1)[0].mean()

    penal_l1 = get_l1_reg(model)
    penal_l2 = get_l2_reg(model)

    loss_no_reg = config.lw_state * x_mse
    for lw, loss_ent in [
        [config.lw_latent, h_mse],
        [config.lw_x_inf_2, Lx_inf_2],
        [config.lw_h_inf_2, Lh_inf_2],
    ]:
        if lw > 0.0:
            loss_no_reg = loss_no_reg + lw * loss_ent

    loss_reg = config.lw_l2reg * penal_l2
    for lw, loss_ent in [
        [config.lw_l1reg, penal_l1],
    ]:
        if lw > 0.0:
            loss_reg = loss_reg + lw * loss_ent

    loss = loss_no_reg + loss_reg

    metrics = np.array(
        [
            loss_no_reg.detach().cpu().numpy(),
            loss_reg.detach().cpu().numpy(),
            x_mse.detach().cpu().numpy(),
            h_mse.detach().cpu().numpy(),
            Lx_inf_2.detach().cpu().numpy(),
            Lh_inf_2.detach().cpu().numpy(),
            # penal_l1.detach().cpu().numpy(),
            # penal_l2.detach().cpu().numpy(),
        ],
        dtype=np.float32,
    )
    return loss, metrics


# used for Floydhub
def print_metrics(train_metrics_mean, val_metrics_mean, epoch, lr):
    print(
        '{{"metric": "Train Loss", "value": {}, "epoch": {}}}'.format(
            train_metrics_mean[0], epoch
        )
    )
    print(
        '{{"metric": "Val Loss", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[0], epoch
        )
    )
    print(
        '{{"metric": "Val State MSE", "value": {}, "epoch": {}}}'.format(
            train_metrics_mean[1], epoch
        )
    )
    print(
        '{{"metric": "Val Latent MSE", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[2], epoch
        )
    )
    print(
        '{{"metric": "Val Infinity MSE", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[3], epoch
        )
    )
    print(
        '{{"metric": "Val Zeros MSE", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[4], epoch
        )
    )
    print(
        '{{"metric": "Regularization", "value": {}, "epoch": {}}}'.format(
            val_metrics_mean[5], epoch
        )
    )
    print('{{"metric": "Learning Rate", "value": {}, "epoch": {}}}'.format(lr, epoch))
