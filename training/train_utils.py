from dataclasses import dataclass
import math
from pathlib import Path
import traceback
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from .protol4config import RunConfig


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


@dataclass
class KoopmanMetrics:
    loss_total: float = 0.0
    """非正则化项加权+正则化项加权"""
    loss_no_reg: float = 0.0
    """非正则化项的加权"""
    loss_reg: float = 0.0
    """正则化项的加权"""
    x_mse: float = 0.0
    r"""\sum_{t,j}\frac{1}{NL} |\delta x_j|^2"""
    h_mse: float = 0.0
    r"""\sum_{t,j}\frac{1}{NL} |\delta h_j|^2"""
    x_inf_sse: float = 0.0
    r"""max_t \sum_j |\delta x_j|^2"""
    h_inf_sse: float = 0.0
    r"""max_t \sum_j |\delta h_j|^2"""


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

    _meta: List[Tuple[float, torch.Tensor]] = [
        [config.lw_latent, h_mse],
        [config.lw_x_inf_2, Lx_inf_2],
        [config.lw_h_inf_2, Lh_inf_2],
    ]
    for lw, loss_entry in _meta:
        if lw > 0.0:
            loss_no_reg = loss_no_reg + lw * loss_entry

    loss_reg = config.lw_l2reg * penal_l2
    _meta: List[Tuple[float, torch.Tensor]] = [
        [config.lw_l1reg, penal_l1],
    ]
    for lw, loss_entry in _meta:
        if lw > 0.0:
            loss_reg = loss_reg + lw * loss_entry

    loss = loss_no_reg + loss_reg

    with torch.no_grad():
        metrics = KoopmanMetrics(
            loss_total=loss.cpu().item(),
            loss_no_reg=loss_no_reg.cpu().item(),
            loss_reg=loss_reg.cpu().item(),
            x_mse=x_mse.cpu().item(),
            h_mse=h_mse.cpu().item(),
            x_inf_sse=Lx_inf_2.cpu().item(),
            h_inf_sse=Lh_inf_2.cpu().item(),
        )
    return loss, metrics

    # used for Floydhub
    # def print_metrics(train_metrics_mean, val_metrics_mean, epoch, lr):
    #     print(
    #         '{{"metric": "Train Loss", "value": {}, "epoch": {}}}'.format(
    #             train_metrics_mean[0], epoch
    #         )
    #     )
    #     print(
    #         '{{"metric": "Val Loss", "value": {}, "epoch": {}}}'.format(
    #             val_metrics_mean[0], epoch
    #         )
    #     )
    #     print(
    #         '{{"metric": "Val State MSE", "value": {}, "epoch": {}}}'.format(
    #             train_metrics_mean[1], epoch
    #         )
    #     )
    #     print(
    #         '{{"metric": "Val Latent MSE", "value": {}, "epoch": {}}}'.format(
    #             val_metrics_mean[2], epoch
    #         )
    #     )
    #     print(
    #         '{{"metric": "Val Infinity MSE", "value": {}, "epoch": {}}}'.format(
    #             val_metrics_mean[3], epoch
    #         )
    #     )
    #     print(
    #         '{{"metric": "Val Zeros MSE", "value": {}, "epoch": {}}}'.format(
    #             val_metrics_mean[4], epoch
    #         )
    #     )
    #     print(
    #         '{{"metric": "Regularization", "value": {}, "epoch": {}}}'.format(
    #             val_metrics_mean[5], epoch
    #         )
    #     )
    #     print('{{"metric": "Learning Rate", "value": {}, "epoch": {}}}'.format(lr, epoch))


def print_metrics(
    trn_ms: KoopmanMetrics,
    val_ms: KoopmanMetrics,
    epoch: int,
    lr: float,
):
    msgs = [
        f"Epoch {epoch}, lr={lr:.06g}",
        f"train_loss: {trn_ms.loss_total:.06g}",
        f"val_loss: {val_ms.loss_total:.06g}",
        f"val_x_mse: {val_ms.x_mse:.06g}",
        f"val_h_mse: {val_ms.h_mse:.06g}",
        f"val_reg: {val_ms.loss_reg:.06g}",
        "",
    ]
    msg = "\n".join(msgs)

    print(msg)


WD1_train = "train"
WD1_val = "val"
#
WD_loss_total = "loss_total"
WD_loss_noreg = "loss_no_reg"
WD_loss_reg = "loss_reg"
WD_state_mse = "state_mse"
WD_latent_mse = "latent_mse"
WD_state_maxtse = "state_max_tse"
WD_latent_maxtse = "latent_max_tse"
WD_lr = "lr"
WD_iter = "iter"
#
K_train_state_mse = f"{WD1_train}/{WD_state_mse}"
K_train_latent_mse = f"{WD1_train}/{WD_latent_mse}"
K_train_state_maxtse = f"{WD1_train}/{WD_state_maxtse}"
K_train_latent_maxtse = f"{WD1_train}/{WD_latent_maxtse}"
K_train_loss_total = f"{WD1_train}/{WD_loss_total}"
K_train_loss_noreg = f"{WD1_train}/{WD_loss_noreg}"
K_train_loss_reg = f"{WD1_train}/{WD_loss_reg}"
#
K_val_state_mse = f"{WD1_val}/{WD_state_mse}"
K_val_latent_mse = f"{WD1_val}/{WD_latent_mse}"
K_val_state_maxtse = f"{WD1_val}/{WD_state_maxtse}"
K_val_latent_maxtse = f"{WD1_val}/{WD_latent_maxtse}"
K_val_loss_total = f"{WD1_val}/{WD_loss_total}"
K_val_loss_noreg = f"{WD1_val}/{WD_loss_noreg}"
K_val_loss_reg = f"{WD1_val}/{WD_loss_reg}"


class KoopmanSummrayWriter:
    def __init__(self, fname: str, model_name: str, use_tensorboard=True):
        self._fn_csv = Path(fname).resolve()
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter as TBSWriter

            self._tbsw = TBSWriter(str(self._fn_csv.parent / "tensorboard"))
        else:
            self._tbsw = None
        colums = [
            "model_name",
            WD_iter,
            WD_lr,
        ]
        for wd1 in [WD1_train, WD1_val]:
            for wd2 in [
                WD_loss_total,
                WD_loss_noreg,
                WD_loss_reg,
                WD_state_mse,
                WD_latent_mse,
                WD_state_maxtse,
                WD_latent_maxtse,
            ]:
                colums.append(f"{wd1}/{wd2}")
        df = pd.DataFrame(columns=colums)
        if self._fn_csv.exists():  # merge
            df_ = pd.read_csv(fname)
            try:
                df = pd.concat([df, df_], ignore_index=True)
            except Exception as e:
                print(e, traceback.format_exc())
        self._df = df
        self.model_name = model_name

    def push(
        self,
        train_ms: KoopmanMetrics,
        val_ms: KoopmanMetrics,
        lr: float,
        to_csv=True,
    ):
        model_name = self.model_name
        df = self._df
        idxs = df["model_name"] == model_name
        itr_last = -1
        if idxs.any():
            itr_last = df[idxs][WD_iter].max()
        itr_new = itr_last + 1

        meta: Dict[str, Union[float, int, str]] = {}
        for wd1, ms in [
            (WD1_train, train_ms),
            (WD1_val, val_ms),
        ]:
            meta[f"{wd1}/{WD_loss_total}"] = ms.loss_total
            meta[f"{wd1}/{WD_loss_noreg}"] = ms.loss_no_reg
            meta[f"{wd1}/{WD_loss_reg}"] = ms.loss_reg
            meta[f"{wd1}/{WD_state_mse}"] = ms.x_mse
            meta[f"{wd1}/{WD_latent_mse}"] = ms.h_mse
            meta[f"{wd1}/{WD_state_maxtse}"] = ms.x_inf_sse
            meta[f"{wd1}/{WD_latent_maxtse}"] = ms.h_inf_sse
        # 添加到 tensorboard
        if self._tbsw is not None:
            add_scalar = self._tbsw.add_scalar
            for name, v in meta.items():
                add_scalar(f"{model_name}/{name}", v, itr_new)

        # 添加到 csv
        meta["model_name"] = model_name
        meta[WD_lr] = lr
        meta[WD_iter] = itr_new
        new_row = pd.Series(meta)
        df.loc[len(df) + 1] = new_row
        if to_csv:
            self._fn_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self._fn_csv, index=False)

    def filter_by_model(self):
        df = self._df
        return df[df["model_name"] == self.model_name]

    def query_min(self, key=K_val_loss_noreg):
        df = self.filter_by_model()
        df = df[key]
        if df.empty:
            min_val = math.inf
        else:
            min_val = float(df.min())
        return min_val

    def close(self):
        if self._tbsw is not None:
            self._tbsw.close()
            self._tbsw = None
        self._df = None

    def __del__(self):
        self.close()
