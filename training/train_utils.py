from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def count_parameters(net):
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
        "enc_shape": config["enc_shape"],
        "aux_shape": config.get("aux_shape"),
        "n_shifts": config["n_shifts"],
        "dt": config.get("dt"),
        "use_rbf": config.get("use_rbf"),
        "drop_prob": config.get("drop_prob"),
        "aux_rbf": config.get("aux_rbf"),
    }
    return model_config


def non_affine_model_configurer(config: dict):
    model_config = {
        "enc_shape": config["enc_shape"],
        "enc_u_shape": config.get("enc_u_shape"),
        "aux_shape": config.get("aux_shape"),
        "n_shifts": config["n_shifts"],
        "dt": config.get("dt"),
        "use_rbf": config.get("use_rbf"),
        "drop_prob": config.get("drop_prob"),
        "enc_rbf": config.get("enc_rbf"),
        "use_enc_u": config.get("use_enc_u"),
    }
    return model_config


def get_lr_scheduler(optimizer, config: dict):
    step = config.get("lr_sch_step", 1000)
    gamma = config.get("lr_sch_gamma", 0.1)
    return optim.lr_scheduler.StepLR(optimizer, step, gamma=gamma)


def get_batch(data, btach_size, config):
    batch = data[torch.randint(len(data), size=(btach_size,))]
    if config["zero_loss"] != 0.0:
        zero = torch.zeros((1, *batch.shape[1:]))
        batch = torch.cat((zero, batch), dim=0)
    return batch


def get_l2_weights(model: nn.Module) -> torch.Tensor:
    l2_reg = 0.0
    for W in model.parameters():
        l2_reg = l2_reg + W.norm(2)
    assert isinstance(l2_reg, torch.Tensor)
    return l2_reg


def koopman_loss(
    xy_pred: torch.Tensor,
    xy_target: torch.Tensor,
    config,
    model,
):
    dim = config["enc_shape"][0]
    x_targ, x_pred = xy_target[:, :, :dim], xy_pred[:, :, :dim]
    y_targ, y_pred = xy_target[:, :, dim:], xy_pred[:, :, dim:]
    if config["zero_loss"] != 0.0:
        zero_loss = torch.mean((xy_target[0] - xy_pred[0]) ** 2)
        idx = 1
    else:
        zero_loss = torch.zeros(1)
        idx = 0
    x_error = x_targ[idx:] - x_pred[idx:]
    y_error = y_targ[idx:] - y_pred[idx:]
    x_mse = torch.mean(x_error**2)
    y_mse = torch.mean(y_error**2)
    min_idx = np.min((config["n_shifts"], 5))
    x_mse_1 = torch.mean(x_error**2, dim=-1)[:, : int(min_idx)]
    x_inf_mse = torch.mean(x_mse_1.norm(p=float("inf"), dim=-1))
    reg_loss = get_l2_weights(model)

    loss = (
        float(config["state_loss"]) * x_mse
        + float(config["latent_loss"]) * y_mse
        + float(config["inf_loss"]) * x_inf_mse
        + float(config["zero_loss"]) * zero_loss
        + float(config["reg_loss"]) * reg_loss
    )

    r_arry = np.array(
        [
            loss.detach().cpu().numpy(),
            x_mse.detach().cpu().numpy(),
            y_mse.detach().cpu().numpy(),
            x_inf_mse.detach().cpu().numpy(),
            zero_loss.detach().cpu().numpy(),
            reg_loss.detach().cpu().numpy(),
        ],
        dtype=np.double,
    )
    return loss, r_arry


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
