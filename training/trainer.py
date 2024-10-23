# coding: utf-8
# -- !(!/usr/bin/env python)

import os
import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import linalg as LA

DIR_PARENT = Path(__file__).resolve().parent
DIR_WS = DIR_PARENT.parent
sys.path.append(f"{DIR_PARENT.parent}")
from model.networks import *
from train_utils import *
from data.data_helper_fns import KoopmanData
from tqdm import tqdm


def path_resolve(path: str) -> Path:
    if path.startswith("~") or path.startswith("/") or path.startswith("."):
        return Path(path).expanduser().resolve()
    return (DIR_WS / path).resolve()


def path_basename(path: str) -> str:
    p = path_resolve(path)
    p = os.path.basename(p)
    return p


def path_addext(path: str, ext: str) -> str:
    p = path_basename(path)
    if not ext.endswith("."):
        ext = "." + ext
    if not p.endswith(ext):
        p += ext
    return p


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

parser = argparse.ArgumentParser("DKoopman", description="Training Models")
parser.add_argument(
    "--config_dir",
    type=str,
    default="training/configs/"
    + [
        "config_deina_jbf_pendulum.json",
        "config_deina_demo.json",
        "config_deina_sd_pend.json",
    ][-1],
    help="config file directory",
)
parser.add_argument(
    "--name",
    type=str,
    help="model file name to save (pure name, without extension and directory)",
    default="tmp",
)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--viz", action="store_true", help="visualize the process")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument(
    "--val_feq", type=int, default=4, help="number of times to validate per epoch"
)
parser.add_argument(
    "--dump_dir", type=str, default="saved/logs", help="directory to save outputs"
)
parser.add_argument("--v", action="store_true", help="verbosity")
parser.add_argument("--random", action="store_true", help="randomize visualization")
parser.add_argument(
    "--load_weights",
    type=str,
    default=None,
    help="dir for a model checkpoint (.pt file)",
)
parser.add_argument(
    "--record", action="store_true", help="save figure at every iteration"
)
args = parser.parse_args()

now = datetime.now().strftime("%Y%m%d_%H%M%S")
EPOCHS: int = args.epochs
# Uncomment if you want time stamps for model names
# MODEL_NAME = args.name + "-{}".format(now)
MODEL_NAME: str = path_basename(args.name)
BATCH_SIZE: int = args.batch_size
VAL_FEQ: int = args.val_feq
DUMP_DIR = path_resolve("runs") / now
VERBOSE: bool = args.v
VIS: bool = args.viz  # or True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float

SUMMARY_DIR = f"{DUMP_DIR}/summary.csv"
MODEL_DIR = f"{DUMP_DIR}/models"
FIGURE_DIR = f"{DUMP_DIR}/figures"

for d in [DUMP_DIR, MODEL_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)
    assert os.path.isdir(d)
if VIS:
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    plt.rcParams.update({"font.size": 10})
    ax_phase = fig.add_subplot(332, frameon=False)
    ax_traj = fig.add_subplot(331, frameon=False)
    ax_x_mse = fig.add_subplot(334, frameon=False)
    ax_y_mse = fig.add_subplot(335, frameon=False)
    ax_x_inf = fig.add_subplot(336, frameon=False)
    ax_0_mse = fig.add_subplot(337, frameon=False)
    ax_reg = fig.add_subplot(338, frameon=False)
    ax_ko = fig.add_subplot(333, frameon=False)
    ax_lr = fig.add_subplot(339, frameon=False)
    plt.show(block=False)


def summary_writer(train_val, val_val, lr: float):
    if not os.path.isfile(SUMMARY_DIR):
        df = pd.DataFrame(
            columns=[
                "model_name",
                "iter",
                "lr",
                "train_loss",
                "state_loss_train",
                "latent_loss_train",
                "inf_loss_train",
                "zero_loss_train",
                "reg_loss_train",
                "val_loss",
                "state_loss_val",
                "latent_loss_val",
                "inf_loss_val",
                "zero_loss_val",
                "reg_loss_val",
            ]
        )
        df.to_csv(SUMMARY_DIR, index=False)

    df = pd.read_csv(SUMMARY_DIR)
    if not df["model_name"].str.contains(MODEL_NAME).any():
        df.loc[len(df) + 1] = [MODEL_NAME, 0, lr, *train_val, *val_val]
    else:
        idx = df[df["model_name"] == MODEL_NAME]["iter"].max()
        df.loc[len(df) + 1] = [MODEL_NAME, idx + 1, lr, *train_val, *val_val]
    df.to_csv(SUMMARY_DIR, index=False)


def save_model(val_loss: float, model: nn.Module, config: Dict):
    # saves model weights if validation loss is the lowest ever
    df = pd.read_csv(SUMMARY_DIR)
    min_val = np.min(df[df["model_name"] == MODEL_NAME]["val_loss"])
    if val_loss <= min_val:
        model_dir = os.path.join(MODEL_DIR, "{}.pt".format(MODEL_NAME))
        model.cpu()
        torch.save(model.state_dict(), model_dir)
        model.to(DEVICE)

        json_dir = os.path.join(MODEL_DIR, "{}.json".format(MODEL_NAME))
        with open(json_dir, "w") as fp:
            json.dump(config, fp)


def zero_pad(data: torch.Tensor):
    # pad first location with zero input (for calculating zero loss)
    zero = torch.zeros((1, *data.shape[1:]), device=DEVICE)
    data_w_zeros = torch.cat((zero, data), dim=0)
    return data_w_zeros


def visualize(model: nn.Module, data: torch.Tensor, iter: int, model_desc: str = ""):
    df = pd.read_csv(SUMMARY_DIR)
    df = df[df["model_name"] == MODEL_NAME]
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        try:
            outputs = model.forward(*data, return_ko=True)
        except:
            outputs = model.forward(data, return_ko=True)
        outputs: List[torch.Tensor]

    xy_gt, xy_pred, kos = outputs
    xy_gt = check_np(xy_gt)
    xy_pred = check_np(xy_pred)
    kos = check_np(kos)
    # import pdb; pdb.set_trace()

    use_random = args.random

    idx = np.random.randint(len(xy_gt)) if use_random else 1
    ax_traj.cla()
    ax_traj.plot(xy_gt[idx, :, 0], "--", color="purple", linewidth=2, alpha=0.5)
    ax_traj.plot(xy_pred[idx, :, 0], color="purple", linewidth=2)
    ax_traj.plot(xy_gt[idx, :, 1], "--", color="grey", linewidth=2, alpha=0.5)
    ax_traj.plot(xy_pred[idx, :, 1], color="grey", linewidth=2)
    ax_traj.set_ylim(np.min(xy_gt) * 1.1, np.max(xy_gt) * 1.1)
    ax_traj.set_title("Trajectories")
    ax_traj.set_xlabel("Time Steps")
    ax_traj.set_ylabel(r"$\mathbf{x}$")

    ax_phase.cla()
    idicies = np.random.randint(len(xy_gt), size=10) if use_random else range(10)
    for i in idicies:
        ax_phase.plot(
            xy_gt[i, :, 0],
            xy_gt[i, :, 1],
            "--",
            color="purple",
            linewidth=2,
            alpha=0.5,
        )
        ax_phase.plot(xy_pred[i, :, 0], xy_pred[i, :, 1], color="grey", linewidth=2)
        ax_phase.scatter(xy_gt[i, 0, 0], xy_gt[i, 0, 1], color="grey", s=15)
    ax_phase.spines["top"].set_visible(False)
    ax_phase.spines["right"].set_visible(False)
    ax_phase.set_xlim(np.min(xy_gt[:, :, 0] * 1.1), np.max(xy_gt[:, :, 0] * 1.1))
    ax_phase.set_ylim(np.min(xy_gt[:, :, 1] * 1.1), np.max(xy_gt[:, :, 1] * 1.1))
    ax_phase.set_title("Phase Portrait")
    ax_phase.set_xlabel(r"$x_1$")
    ax_phase.set_ylabel(r"$x_2$")

    x_axis = np.array(df["iter"])
    x_mse_train = np.array(df["state_loss_train"])
    x_mse_val = np.array(df["state_loss_val"])
    y_mse_train = np.array(df["latent_loss_train"])
    y_mse_val = np.array(df["latent_loss_val"])
    x_inf_train = np.array(df["inf_loss_train"])
    x_inf_val = np.array(df["inf_loss_val"])
    zero_train = np.array(df["zero_loss_train"])
    zero_val = np.array(df["zero_loss_val"])
    reg_train = np.array(df["reg_loss_train"])
    reg_val = np.array(df["reg_loss_val"])
    reg_val = np.array(df["reg_loss_val"])
    lrs = np.array(df["lr"])

    ax_x_mse.cla()
    ax_x_mse.plot(x_axis, x_mse_train, label="train", color="purple", linewidth=2)
    ax_x_mse.scatter(x_axis[-1], x_mse_train[-1], color="purple", s=25)
    ax_x_mse.plot(x_axis, x_mse_val, label="val", color="grey", linewidth=2)
    ax_x_mse.scatter(x_axis[-1], x_mse_val[-1], color="grey", s=25)
    ax_x_mse.set_title("State MSE")
    ax_x_mse.set_yscale("log")
    ax_x_mse.set_xlabel("Iterations")
    ax_x_mse.set_ylim([None, np.min((np.max(x_mse_train), 2))])
    ax_x_mse.legend()

    ax_y_mse.cla()
    ax_y_mse.plot(x_axis, y_mse_train, label="train", color="purple", linewidth=2)
    ax_y_mse.scatter(x_axis[-1], y_mse_train[-1], s=25, color="purple")
    ax_y_mse.plot(x_axis, y_mse_val, label="val", color="grey", linewidth=2)
    ax_y_mse.scatter(x_axis[-1], y_mse_val[-1], s=25, color="grey")
    ax_y_mse.set_title("Latent MSE")
    ax_y_mse.set_yscale("log")
    ax_y_mse.set_xlabel("Iterations")
    ax_y_mse.set_ylim([None, np.min((np.max(y_mse_train), 1))])
    ax_y_mse.legend()

    ax_x_inf.cla()
    ax_x_inf.plot(x_axis, x_inf_train, label="train", color="purple", linewidth=2)
    ax_x_inf.scatter(x_axis[-1], x_inf_train[-1], s=25, color="purple")
    ax_x_inf.plot(x_axis, x_inf_val, label="val", color="grey", linewidth=2)
    ax_x_inf.scatter(x_axis[-1], x_inf_val[-1], s=25, color="grey")
    ax_x_inf.set_title("Max State Deviation")
    ax_x_inf.set_yscale("log")
    ax_x_inf.set_xlabel("Iterations")
    ax_x_inf.legend()

    ax_0_mse.cla()
    ax_0_mse.plot(x_axis, zero_train, label="train", color="purple", linewidth=2)
    ax_0_mse.scatter(x_axis[-1], zero_train[-1], s=25, color="purple")
    ax_0_mse.plot(x_axis, zero_val, label="val", color="grey", linewidth=2)
    ax_0_mse.scatter(x_axis[-1], zero_val[-1], s=25, color="grey")
    ax_0_mse.set_title("Zero Loss")
    ax_0_mse.set_yscale("log")
    ax_0_mse.set_xlabel("Iterations")
    ax_0_mse.legend()
    ax_0_mse.set_ylim([np.min(zero_train) + 1e-11, None])

    ax_reg.cla()
    ax_reg.plot(x_axis, reg_train, label="train", color="purple", linewidth=2)
    ax_reg.scatter(x_axis[-1], reg_train[-1], s=25, color="purple")
    ax_reg.plot(x_axis, reg_val, label="val", color="grey", linewidth=2)
    ax_reg.scatter(x_axis[-1], reg_val[-1], s=25, color="grey")
    ax_reg.set_title("Regularization Loss")
    ax_reg.set_yscale("log")
    ax_reg.set_xlabel("Iterations")
    ax_reg.legend()

    ax_lr.cla()
    ax_lr.plot(x_axis, lrs, color="purple", linewidth=2)
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Iterations")

    idx = np.random.randint(len(xy_gt)) if use_random else 1
    if len(kos.shape) == 3:
        w, v = LA.eig(kos[idx])
    else:
        w, v = LA.eig(kos)
    ax_ko.cla()
    ax_ko.scatter(np.real(w), np.imag(w), marker="+", color="purple", s=30)
    ax_ko.set_title("Koopman Eigenvalues")
    ax_ko.set_xlabel(r"$\mathcal{R}(\lambda)$")
    ax_ko.set_ylabel(r"$\mathcal{I}(\lambda)$")

    fig.tight_layout()
    plt.suptitle("{}:\n{}".format(MODEL_NAME, model_desc))

    plt.subplots_adjust(top=0.9, hspace=0.5, wspace=0.5)
    if args.record:
        plt.savefig("{}/{}_{}.png".format(FIGURE_DIR, MODEL_NAME, iter))
    else:
        plt.savefig("{}/{}.png".format(FIGURE_DIR, MODEL_NAME))
    plt.draw()
    plt.pause(0.001)


def main():
    assert args.config_dir is not None, "Please provide a config file directory"
    config_fn = path_resolve(args.config_dir)
    with open(config_fn, "r") as fp:
        config: Dict = json.load(fp)
    AFFINE_FLAG = True
    network_name = config["network"]
    if "deina" in network_name:
        AFFINE_FLAG = False
    if AFFINE_FLAG:
        model_config = affine_model_configurer(config)
    else:
        model_config = non_affine_model_configurer(config)

    model: Union[
        LREN,
        DENIS,
        DENIS_JBF,
        DEINA,
        DEINA_SD,
    ] = name2model[
        network_name
    ](model_config)
    if args.load_weights is not None:
        try:
            model_w_fn = path_resolve(args.load_weights)
            print(f"{model_w_fn}\n")
            with open(model_w_fn, "rb") as fp:
                # print(fp.read())
                model.load_state_dict(torch.load(fp))
            print(f"All keys match, model<<{model_w_fn}\n")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    model.to(DEVICE, DTYPE)

    if args.v:
        print("Entering training loop\n")
        print("Model configuration: {}\n".format(model_config))
        print(count_parameters(model))
        print("\n")

    optimizer = optim.Adam(model.parameters(), config["lr"])
    lr_scheduler = get_lr_scheduler(optimizer, config)

    train_x = torch.tensor(
        np.load(path_resolve(config["train_data"])),
        device=DEVICE,
        dtype=DTYPE,
    )
    val_x = torch.tensor(
        np.load(path_resolve(config["val_data"])),
        device=DEVICE,
        dtype=DTYPE,
    )
    with_u = not AFFINE_FLAG
    if with_u:
        train_u = torch.tensor(
            np.load(path_resolve(config["train_data_inputs"])),
            device=DEVICE,
            dtype=DTYPE,
        )
        val_u = torch.tensor(
            np.load(path_resolve(config["val_data_inputs"])),
            device=DEVICE,
            dtype=DTYPE,
        )
    else:
        train_u = None
        val_u = None

    from torch.nn.utils import clip_grad_value_

    def _proc_batch(
        xs: torch.Tensor,
        us: torch.Tensor = None,
        metrics_mean: List[np.ndarray] = [],
        loss_mean: List[float] = [],
        training=True,
    ):
        # if config["zero_loss"] != 0.0:
        #     batch_x = zero_pad(batch_x)
        #     if with_u:
        #         batch_u = zero_pad(batch_u)
        if training:
            optimizer.zero_grad()

        if with_u:
            xy_targ, xy_pred = model(xs, us)
        else:
            xy_targ, xy_pred = model(xs)

        loss, loss_array = koopman_loss(xy_pred, xy_targ, config, model)

        if training:
            loss.backward()

            # grad clip
            # clip_grad_value_(model.parameters(), 10.0)

            optimizer.step()

        metrics_mean.append(loss_array)
        loss_mean.append(loss.item())
        return metrics_mean, loss_mean

    N = len(train_x)
    model_desc = config["description"]
    is_floyd = config.get("mode") == "floyd"
    for epoch in range(EPOCHS):
        # dont show tqdm on floydhub
        if is_floyd:
            pbar = range(0, N, BATCH_SIZE)
        else:
            pbar = tqdm(range(0, N, BATCH_SIZE))
        # permute training samples across each epoch
        permutation = torch.randperm(N)

        # training starts
        trn_metrics = []
        trn_losses = []
        for idx, i in enumerate(pbar):
            trn_ids = permutation[i : i + BATCH_SIZE]

            model.train(True)
            trn_metrics, trn_losses = _proc_batch(
                train_x[trn_ids],
                train_u[trn_ids] if with_u else None,
                trn_metrics,
                trn_losses,
                training=True,
            )

            # validation starts
            if (idx + 1) % (len(pbar) // VAL_FEQ) == 0 or idx == len(pbar) - 1:
                val_metrics = []
                val_losses = []
                model.train(False)
                with torch.no_grad():
                    # batching validation set
                    for j in range(0, len(val_x), BATCH_SIZE):
                        val_ids = slice(j, j + BATCH_SIZE)
                        batch_val_x = val_x[val_ids]
                        batch_val_u = val_u[val_ids] if with_u else None
                        val_metrics, val_losses = _proc_batch(
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
                lr = lr_scheduler.get_last_lr()[0]
                summary_writer(trn_metrics_mean, val_metrics_mean, lr)

                save_model(val_metrics_mean[0], model=model, config=config)

                if VIS:
                    if with_u:
                        visualize(
                            model,
                            [batch_val_x, batch_val_u],
                            epoch,
                            model_desc=model_desc,
                        )
                    else:
                        visualize(model, batch_val_x, epoch, model_desc=model_desc)

                if not is_floyd:
                    # update progress bar
                    desc = (
                        "{}/{}] loss: {:.4f}/{:.4f}, state_mse: {:.4f}/{:.4f}".format(
                            epoch,
                            EPOCHS,
                            trn_loss_mean,
                            val_loss_mean,
                            trn_metrics_mean[0],
                            val_metrics_mean[0],
                        )
                    )
                    pbar.set_description(desc)
                    pbar.refresh()

        if is_floyd:
            print_metrics(trn_metrics, val_metrics_mean, epoch, lr)
        print(epoch)
        lr_scheduler.step()


if __name__ == "__main__":
    main()
