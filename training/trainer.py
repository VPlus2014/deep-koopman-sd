# coding: utf-8
# -- !(!/usr/bin/env python)

import math
import os
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import traceback
from typing import List
import pandas as pd
import numpy as np
from numpy import linalg as LA
import sys
import matplotlib.pyplot as plt

DIR_WS = Path(__file__).resolve().parent.parent
sys.path.append(f"{DIR_WS}")
from model.networks import *
from data.data_helper_fns import KoopmanData
from tqdm import tqdm
from .protol4config import RunConfig
from .train_utils import *


def now2str():
    return datetime.now().strftime("%m%d_%H%M%S")


def path_resolve(path: str) -> Path:
    if path.startswith("~") or path.startswith("/") or path.startswith("."):
        return Path(path).expanduser().resolve()
    return (DIR_WS / path).resolve()


def path_addext(path: str, ext: str) -> str:
    p = os.path.splitext(path)[0]
    if not ext.startswith("."):
        ext = "." + ext
    if not p.endswith(ext):
        p += ext
    return p


def path_is_relative(path: str) -> bool:
    return not any([path.startswith(c) for c in ["/", "\\", "~"]])


def check_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


name2model = {
    "lren": LREN,
    "denis": DENIS,
    "denis_jbf": DENIS_JBF,
    "deina": DEINA,
    "deina_jbf": DEINA_JBF,
    "deina_sd": DEINA_SD,
}


def get_args():
    parser = argparse.ArgumentParser("DKoopman", description="Training Models")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/"
        + [
            "config_deina_jbf_pendulum.json",
            "config_deina_demo.json",
            "config_deina_sd_pend.json",
            "pend_deina_sd.json",
            "dof6plane_deina_rsd.json",
        ][-1],
        help="config file directory",
    )
    # parser.add_argument(
    #     "--taskname",
    #     type=str,
    #     default="",
    #     help="task folder name in runs directory, default is current timestamp",
    # )
    args = parser.parse_args()
    return args


class IntLinspaceTimer:
    def __init__(self, ini: int, end: int, n: int):
        assert end >= ini
        self._n = n
        self._x0 = ini
        self._span = end - ini
        self.reset()

    def reset(self):
        self._k = 0
        self._xk = self._x0
        self._x = self._x0 - 1

    def step(self) -> bool:
        self._x += 1
        if self._x >= self._xk:
            n = self._n
            self._k += 1
            self._xk = (self._x0 * n + self._span * self._k) // n
            return True
        return False


class VizProcess:
    def __init__(
        self,
        figs_dir: str,
        use_random=False,
        use_itr=False,
        use_gui=True,
        seed=None,
    ):
        fig = plt.figure(figsize=(24, 16), dpi=300, facecolor="white")
        self.fig = fig
        self.ax_x = fig.add_subplot(331, frameon=False)
        self.ax_y = fig.add_subplot(332, frameon=False)
        self.ax_phase = fig.add_subplot(333, frameon=False)
        self.ax_ko = fig.add_subplot(336, frameon=False)
        self.ax_x_mse = fig.add_subplot(334, frameon=False)
        self.ax_h_mse = fig.add_subplot(335, frameon=False)
        self.ax_reg = fig.add_subplot(339, frameon=False)
        self.ax_x_inf = fig.add_subplot(337, frameon=False)
        self.ax_h_inf = fig.add_subplot(338, frameon=False)
        self.ax_lr = None  # fig.add_subplot(339, frameon=False)

        plt.rcParams.update({"font.size": 10})
        if use_gui:
            plt.show(block=False)

        self.use_random = use_random
        self._rng = np.random.default_rng(seed)

        self.save_itr = use_itr
        self.use_gui = use_gui
        self.figs_dir = Path(figs_dir).resolve()

    def visualize(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        iter: int,
        model_name: str,
        df_stat: pd.DataFrame,
        model_desc: str = "",
        i_phx=0,
        i_phy=1,
    ):
        rng = self._rng
        use_random = self.use_random
        n_samp = 1
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            try:
                outputs = model.forward(*model_input, return_ko=True)
            except:
                outputs = model.forward(model_input, return_ko=True)
            outputs: List[torch.Tensor]

        xy_gt, xy_pred, kos = outputs
        xy_gt = check_np(xy_gt)
        xy_pred = check_np(xy_pred)
        kos = check_np(kos)
        # import pdb; pdb.set_trace()

        fig = self.fig
        ax_x = self.ax_x
        ax_y = self.ax_y
        ax_phase = self.ax_phase
        ax_ko = self.ax_ko
        ax_x_mse = self.ax_x_mse
        ax_h_mse = self.ax_h_mse
        ax_x_inf = self.ax_x_inf
        ax_h_inf = self.ax_h_inf
        ax_reg = self.ax_reg
        ax_lr = self.ax_lr

        clr_trn = "purple"
        clr_val = "grey"
        clr_CKg = "cyan"
        clr_CgF = "green"
        formal_Ckg = r"$\mathcal{C}\circ\mathcal{K}\circ g$"
        formal_CgF = r"$\mathcal{C}\circ g\circ F$"

        def minmax2lim(xmin: float, xmax: float, lscale=1.1, hscale=1.1, eps=1e-6):
            xc = (xmin + xmax) * 0.5 + eps
            xr = abs(xmax - xmin) * 0.5
            if xr == 0:
                xr = 1.0
            return xc - xr * lscale, xc + xr * hscale

        idx = rng.integers(len(xy_gt)) if use_random else -1

        # pred&gt in 2 dims for phase
        ax_y.cla()
        for ax, i_dim in [
            (ax_x, i_phx),
            (ax_y, i_phy),
        ]:
            ax.cla()

            _ys_gt = xy_gt[idx, :, i_dim]
            _ys_pred = xy_pred[idx, :, i_dim]
            ax.plot(
                _ys_gt,
                ls="-",
                color=clr_CgF,
                linewidth=2,
                alpha=0.8,
                label=formal_CgF,
            )
            ax.plot(
                _ys_pred,
                ls="--",
                color=clr_CKg,
                linewidth=2,
                alpha=0.5,
                label=formal_Ckg,
            )
            #
            _ys_gt: np.ndarray = xy_gt[..., i_dim]
            _ys_pred: np.ndarray = xy_pred[..., i_dim]
            ymin = min(_ys_pred.min(), _ys_gt.min())
            ymax = max(_ys_pred.max(), _ys_gt.max())
            ax.set_ylim(*minmax2lim(ymin, ymax))
            ax.set_title(f"Trajectories in dim {i_dim}")
            ax.set_xlabel("Sim Time")
            ax.set_ylabel(rf"$\mathbf{{X}}_{i_dim}$")
            ax.legend()

        if len(kos.shape) == 3:
            w, v = LA.eig(kos[idx])
        else:
            w, v = LA.eig(kos)
        ax_ko.cla()
        ax_ko.scatter(np.real(w), np.imag(w), marker="+", color="purple", s=30)
        ax_ko.set_title("Koopman Eigenvalues")
        ax_ko.set_xlabel(r"Re$(\lambda)$")
        ax_ko.set_ylabel(r"Im$(\lambda)$")

        # phase portrait
        N_cap = len(xy_gt)
        n_samp = min(n_samp, N_cap)
        idicies = rng.integers(N_cap, size=n_samp) if use_random else range(n_samp)
        ls_CKg = "--"
        ls_CgF = "-"
        ax_phase.cla()
        for i in idicies:
            ax_phase.plot(
                xy_gt[i, :, i_phx],
                xy_gt[i, :, i_phy],
                ls_CgF,
                color=clr_CgF,
                linewidth=2,
                alpha=0.8,
                label="$\mathcal{C}\circ g\circ F$",
            )
            ax_phase.scatter(
                xy_gt[i, 0, i_phx], xy_gt[i, 0, i_phy], color=clr_CgF, s=15
            )

            ax_phase.plot(
                xy_pred[i, :, i_phx],
                xy_pred[i, :, i_phy],
                ls_CKg,
                color=clr_CKg,
                alpha=0.5,
                linewidth=2,
                label="$\mathcal{C}\circ\mathcal{K}\circ g$",
            )
            ax_phase.scatter(
                xy_gt[i, 0, i_phx], xy_gt[i, 0, i_phy], color=clr_CKg, s=15
            )

        ax_phase.spines["top"].set_visible(False)
        ax_phase.spines["right"].set_visible(False)

        idicies2 = slice(None, None)
        xmin = min(xy_gt[idicies2, :, i_phx].min(), xy_pred[idicies2, :, i_phx].min())
        xmax = max(xy_gt[idicies2, :, i_phx].max(), xy_pred[idicies2, :, i_phx].max())
        ymin = min(xy_gt[idicies2, :, i_phy].min(), xy_pred[idicies2, :, i_phy].min())
        ymax = max(xy_gt[idicies2, :, i_phy].max(), xy_pred[idicies2, :, i_phy].max())
        ax_phase.set_xlim(*minmax2lim(xmin, xmax))
        ax_phase.set_ylim(*minmax2lim(ymin, ymax))
        ax_phase.set_title("Phase Portrait")
        ax_phase.set_xlabel(rf"$X_{{{i_phx}}}$")
        ax_phase.set_ylabel(rf"$X_{{{i_phy}}}$")
        ax_phase.legend()

        t_axis = np.asarray(df_stat["iter"])
        lrs = np.asarray(df_stat["lr"])
        #
        x_mse_trn = np.asarray(df_stat[K_train_state_mse])
        h_mse_trn = np.asarray(df_stat[K_train_latent_mse])
        x_inf_trn = np.asarray(df_stat[K_train_state_maxtse])
        h_inf_trn = np.asarray(df_stat[K_train_latent_maxtse])
        reg_trn = np.asarray(df_stat[K_train_reg])
        #
        x_mse_val = np.asarray(df_stat[K_val_state_mse])
        h_mse_val = np.asarray(df_stat[K_val_latent_mse])
        x_inf_val = np.asarray(df_stat[K_val_state_maxtse])
        h_inf_val = np.asarray(df_stat[K_val_latent_maxtse])
        reg_val = np.asarray(df_stat[K_val_reg])

        def _plot_lines(
            ax: plt.Axes,
            ys_trn,
            ys_val,
            title: str = "",
            xlabel: str = "Iterations",
            ylabel: str = "",
            yscale="log",
        ):
            ax.cla()
            ax.plot(t_axis, ys_trn, label="train", color=clr_trn, linewidth=2)
            ax.scatter(t_axis[-1], ys_trn[-1], color=clr_trn, s=25)
            ax.plot(t_axis, ys_val, label="val", color=clr_val, linewidth=2)
            ax.scatter(t_axis[-1], ys_val[-1], color=clr_val, s=25)

            ymin = min(np.min(ys_trn), np.min(ys_val))
            ymax = max(np.max(ys_trn), np.max(ys_val))
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            lscale = 1.0 if yscale == "log" else 1.1
            ax.set_ylim(*minmax2lim(ymin, ymax, lscale))
            ax.set_yscale(yscale)
            ax.legend()

        _plot_lines(
            ax=ax_x_mse,
            ys_trn=x_mse_trn,
            ys_val=x_mse_val,
            title="State MSE",
        )
        _plot_lines(
            ax=ax_h_mse,
            ys_trn=h_mse_trn,
            ys_val=h_mse_val,
            title="Latent MSE",
        )

        _plot_lines(
            ax=ax_x_inf,
            ys_trn=x_inf_trn,
            ys_val=x_inf_val,
            title="Max Temporal State Deviation",
        )
        _plot_lines(
            ax=ax_h_inf,
            ys_trn=h_inf_trn,
            ys_val=h_inf_val,
            title="Max Temporal Latent Deviation",
        )

        _plot_lines(
            ax=ax_reg,
            ys_trn=reg_trn,
            ys_val=reg_val,
            title="Regularization Loss",
            yscale="linear",
        )

        if ax_lr is not None:
            ax_lr.cla()
            ax_lr.plot(t_axis, lrs, color=clr_trn, linewidth=2)
            ax_lr.set_title("Learning Rate")
            ax_lr.set_xlabel("Iterations")

        fig.tight_layout()
        fig.suptitle("{}:\n{}".format(model_name, model_desc))

        fig.subplots_adjust(top=0.9, hspace=0.5, wspace=0.5)

        fn = model_name
        if self.save_itr:
            fn += f"_{iter}"
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.figs_dir / path_addext(fn, ".png"))
        fig.canvas.draw()  # 确保出图

        if self.use_gui:
            # plt.draw()
            plt.pause(0.01)
        return


def save_model(
    model: nn.Module,
    config: Dict,
    itr: int,
    model_dir: str,
    model_name: str,
    device_src: torch.device,
):
    model_head = os.path.join(model_name, model_name)

    model_dir = path_addext(model_head, ".pt")
    Path(model_dir).parent.mkdir(parents=True, exist_ok=True)

    model.cpu()
    torch.save(model.state_dict(), model_dir)
    model.to(device_src)

    json_dir = path_addext(model_head, ".json")
    Path(json_dir).parent.mkdir(parents=True, exist_ok=True)
    with open(json_dir, "w") as f:
        json.dump(config, f, indent=4)


def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass


def main():
    now_str = now2str()
    args = get_args()

    config_fn = args.config
    assert config_fn, "Please provide a config file directory"
    config_fn = Path(config_fn).resolve()
    print(f"config<< {config_fn}")
    cfg = RunConfig.load(config_fn)
    cfg_meta = cfg.meta()
    cfg.save(config_fn)

    RUNS_DIR = DIR_WS / "outputs"
    DUMP_DIR = RUNS_DIR / f"{cfg.task_head}_{now_str}"
    assert not DUMP_DIR.exists(), f"Task path {DUMP_DIR} already exists"
    print(f"task>> {DUMP_DIR}")

    SUMMARY_DIR = f"{DUMP_DIR}/summary.csv"
    MODEL_DIR = f"{DUMP_DIR}"
    FIGURES_DIR = f"{DUMP_DIR}/figures"
    MODEL_NAME: str = "model"

    rng = np.random.default_rng(cfg.seed)
    INT32_MAX = np.iinfo(np.int32).max
    init_seed(rng.integers(INT32_MAX))

    EPOCHS: int = cfg.epochs
    BATCH_SIZE: int = cfg.batch_size
    VAL_FEQ: int = cfg.val_feq

    dv = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtp = torch.float64

    AFFINE_FLAG = True
    network_name = cfg.network
    if "deina" in network_name:
        AFFINE_FLAG = False
    if AFFINE_FLAG:
        model_config = affine_model_configurer(cfg_meta)
    else:
        model_config = non_affine_model_configurer(cfg_meta)

    model: Union[
        LREN,
        DENIS,
        DENIS_JBF,
        DEINA,
        DEINA_SD,
    ] = name2model[
        network_name
    ](model_config)
    if cfg.weights_name:
        try:
            model_w_fn = Path(RUNS_DIR) / cfg.weights_name
            with open(model_w_fn, "rb") as fp:
                # print(fp.read())
                model.load_state_dict(torch.load(fp))
            print(f"model<< {model_w_fn}\n")
        except Exception as e:
            print(f"failed to load weights: {e}")
    model.to(dv, dtp)

    if cfg.verbose:
        print("Entering loop\n")
        print("Model configuration: {}\n".format(model_config))
        print(count_parameters(model))
        print("\n")

    optimizer = optim.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    lr_scheduler = get_lr_scheduler(optimizer, cfg)

    try:
        train_x = torch.tensor(
            np.load(cfg.fn_train_x),
            device=dv,
            dtype=dtp,
        )
    except Exception as e:
        print(f"failed to load train_x: {e}")
        raise e
    val_x = torch.tensor(
        np.load(cfg.fn_val_x),
        device=dv,
        dtype=dtp,
    )
    with_u = not AFFINE_FLAG
    if with_u:
        train_u = torch.tensor(np.load(cfg.fn_train_u), device=dv, dtype=dtp)
        val_u = torch.tensor(np.load(cfg.fn_val_u), device=dv, dtype=dtp)
    else:
        train_u = None
        val_u = None

    sw = TinySummrayWriter(SUMMARY_DIR, model_name=MODEL_NAME)
    viz = cfg.viz
    if viz:
        vp = VizProcess(
            figs_dir=FIGURES_DIR,
            use_gui=cfg.viz_gui,
            use_itr=cfg.viz_save_itr,
            seed=rng.integers(INT32_MAX),
        )
    from torch.nn.utils import clip_grad_value_

    grad_max = cfg.grad_max

    def _proc_batch(
        xs: torch.Tensor,
        us: torch.Tensor = None,
        metrics_mean: List[np.ndarray] = [],
        loss_mean: List[float] = [],
        training=True,
    ):
        # if config["lw_zero"] != 0.0:
        #     batch_x = zero_pad(batch_x)
        #     if with_u:
        #         batch_u = zero_pad(batch_u)
        if training:
            optimizer.zero_grad()

        if with_u:
            xy_targ, xy_pred = model(xs, us)
        else:
            xy_targ, xy_pred = model(xs)

        loss, metrics = koopman_loss(xy_pred, xy_targ, cfg, model)

        if training:
            loss.backward()

            # grad clip
            if grad_max > 0.0:
                clip_grad_value_(model.parameters(), grad_max)

            optimizer.step()

        metrics_mean.append(metrics)
        loss_mean.append(metrics[0])
        return loss_mean, metrics_mean

    N = len(train_x)
    model_desc = cfg.description
    use_floyd = cfg.use_floyd
    training = cfg.train
    pbar_e = range(EPOCHS)
    if not use_floyd:
        pbar_e = tqdm(pbar_e)
    for epoch in pbar_e:
        pbar_b = range(0, N, BATCH_SIZE)
        # dont show tqdm on floydhub
        if not use_floyd:
            pbar_b = tqdm(pbar_b)
        ilt = IntLinspaceTimer(0, len(pbar_b) - 1, VAL_FEQ)

        # permute training samples across each epoch
        trn_ids = torch.randperm(N)

        # training starts
        trn_metrics = []
        trn_losses = []
        IDX_x_mse = 2
        IDX_h_mse = IDX_x_mse + 1
        for idx, i in enumerate(pbar_b):
            b_ids = trn_ids[i : i + BATCH_SIZE]

            model.train(True)
            trn_losses, trn_metrics = _proc_batch(
                train_x[b_ids],
                train_u[b_ids] if with_u else None,
                trn_metrics,
                trn_losses,
                training=True,
            )

            # validation starts
            is_final_batch = (idx + 1) == len(pbar_b)
            _tick = ilt.step()
            if _tick or is_final_batch:
                val_metrics = []
                val_losses = []
                model.train(False)
                with torch.no_grad():
                    # batching validation set
                    for j in range(0, len(val_x), BATCH_SIZE):
                        val_ids = slice(j, j + BATCH_SIZE)
                        batch_val_x = val_x[val_ids]
                        batch_val_u = val_u[val_ids] if with_u else None
                        val_losses, val_metrics = _proc_batch(
                            batch_val_x,
                            batch_val_u,
                            val_metrics,
                            val_losses,
                            training=False,
                        )

                # average loss components across validation batches
                trn_metrics_mean = np.asarray(trn_metrics).mean(axis=0)
                trn_loss_mean = np.mean(trn_losses)
                val_metrics_mean = np.asarray(val_metrics).mean(axis=0)
                val_loss_mean = np.mean(val_losses)
                lr_ = lr_scheduler.get_last_lr()[0]

                # saves model weights if validation loss is the lowest ever
                val_loss_cur = val_metrics_mean[0]
                val_loss_opt = sw.query_min()  # 历史最优(排除当前)
                # print(
                #     " ".join(
                #         [
                #             f"val_loss_cur: {val_loss_cur:.06g}",
                #             f"val_loss_opt: {val_loss_opt:.06g}",
                #             f"diff: {val_loss_cur-val_loss_opt:.06g}",
                #         ]
                #     )
                # )
                if training:
                    if val_loss_cur <= val_loss_opt:
                        save_model(
                            model=model,
                            config=cfg_meta,
                            itr=epoch,
                            model_dir=MODEL_DIR,
                            model_name=MODEL_NAME,
                            device_src=dv,
                        )

                    #
                sw.push(trn_metrics_mean, val_metrics_mean, lr_)

                if viz and is_final_batch:
                    if with_u:
                        viz_data = [batch_val_x, batch_val_u]
                    else:
                        viz_data = batch_val_x

                    df4viz = sw.filter_by_model()
                    vp.visualize(
                        model,
                        viz_data,
                        epoch,
                        model_name=MODEL_NAME,
                        model_desc=model_desc,
                        df_stat=df4viz,
                        i_phx=cfg.viz_phase_x_idx,
                        i_phy=cfg.viz_phase_y_idx,
                    )

                if not use_floyd:
                    # update progress bar
                    trn_x_mse_ = trn_metrics_mean[IDX_x_mse]
                    val_x_mse_ = val_metrics_mean[IDX_x_mse]
                    trn_h_mse_ = trn_metrics_mean[IDX_h_mse]
                    val_h_mse_ = val_metrics_mean[IDX_h_mse]
                    fmt_flt_ = "{:0.01e}"
                    fmt_tup_ = fmt_flt_ + "/" + fmt_flt_
                    desc = " ".join(
                        [
                            # f"{epoch}/{EPOCHS}",
                            (f"L:{fmt_tup_}").format(trn_loss_mean, val_loss_mean),
                            (f"Xmse:{fmt_tup_}").format(trn_x_mse_, val_x_mse_),
                            (f"Hmse:{fmt_tup_}").format(trn_h_mse_, val_h_mse_),
                        ]
                    )
                    pbar_b.set_description(desc)
                    pbar_b.refresh()

        if not use_floyd:
            # pbar_e.set_description(f"Epoch {epoch}/{EPOCHS}")
            pbar_e.refresh()
        else:
            print_metrics(trn_metrics, val_metrics_mean, epoch, lr_)
        lr_scheduler.step()


if __name__ == "__main__":
    raise Exception(
        f"do not run this file directly, use {__file__}.{main.__name__}() instead"
    )
