from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

WS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WS_DIR / "data" / "raw_data"
RUNS_DIR = WS_DIR / "runs"


@dataclass
class RunConfig:
    data_name: str
    """full path = $data/raw_data/$data_name"""

    # architecture
    network: str
    enc_x_shape: List[int]
    enc_u_shape: List[int] = field(default_factory=lambda: [])
    aux_shape: List[int] = field(default_factory=lambda: [])
    activation: str = "hrelu"

    task_head: str = ""
    """folder's head name, full path = $runs/$task_head_$timestamp"""
    description: str = ""
    """description of the experiment, is to show in figure title"""
    weights_name: str = ""
    """full path = $runs/$weights_name"""

    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    lr_sch_step: int = 1000
    lr_sch_gamma: float = 0.5
    weight_decay: float = 0.0
    device: str = "cuda:0"
    seed: int = 42

    # log_interval: int = 100
    # save_interval: int = 1000
    # save_dir: str = "checkpoints"
    # model_name: str = "koopman"
    # dataset_name: str = "pendulum"
    # num_workers: int = 4
    # use_wandb: bool = False
    # wandb_project: str = "deep-koopman"
    # wandb_entity: str = "vplus"
    # wandb_name: str = "pendulum-koopman"

    train: bool = True
    val_feq: int = 4
    """number of times to validate per epoch"""

    n_shifts: int = 51
    use_rbf: bool = False
    enc_rbf: bool = False
    use_enc_x: bool = True
    use_enc_u: bool = False
    use_complex: bool = False
    use_affine: bool = False
    kernel: str = "gaussian"
    drop_prob: float = 0.0

    lw_state: float = 1.0
    r"""$\frac{1}{N*L}\sum_{t,i} |e_x(t,i)|^2$"""
    lw_latent: float = 1.0
    r"""$\frac{1}{N*L}\sum_{t,i} |e_h(t,i)|^2$"""
    # lw_zero: float = 0.0
    lw_x_inf_2: float = 1e-02
    r"""$\max_{t}\sum_{i}|e_x(t,i)|^2$"""
    lw_h_inf_2: float = 1e-02
    r"""$\max_{t}\sum_{i}|e_h(t,i)|^2$"""
    lw_l1reg: float = 0.0
    r"""$\frac{1}{2}\sum_i max(0,w_i^2-\delta^2)$"""
    lw_l2reg: float = 1e-03
    r"""$\sum_i max(0,|w_i|-\delta)$"""
    use_huber: bool = False
    """use huber function instead of squared error"""

    verbose: bool = True
    viz: bool = True
    viz_random: bool = False
    """random choice in visualization"""
    viz_gui: bool = True
    """ouput to GUI"""
    viz_save_itr: bool = False
    """save with iteration number"""
    viz_phase_x_idx: int = 0
    """index of the phase variable in x axis"""
    viz_phase_y_idx: int = 1
    """index of the phase variable in y axis"""
    use_floyd: bool = False
    """use floydhub"""

    seed: Optional[int] = None
    """random seed"""

    @property
    def _data_dir(self):
        return DATA_DIR / self.data_name

    @property
    def fn_train_x(self) -> Path:
        return self._data_dir / "train_x.npy"

    @property
    def fn_train_u(self) -> Path:
        return self._data_dir / "train_u.npy"

    @property
    def fn_val_x(self) -> Path:
        return self._data_dir / "val_x.npy"

    @property
    def fn_val_u(self) -> Path:
        return self._data_dir / "val_u.npy"

    @staticmethod
    def load(fn: str):
        fn = Path(fn).resolve()
        with open(fn, "r") as f:
            meta: Dict = json.load(f)
        fd = RunConfig.__dataclass_fields__
        ks = meta.keys()
        meta = {k: meta[k] for k, v in fd.items() if k in ks}
        cfg = RunConfig(**meta)
        return cfg

    def meta(self) -> Dict[str, Any]:
        ks = self.__dataclass_fields__.keys()
        d = {k: getattr(self, k) for k in ks}
        return d

    def save(self, fn: str):
        fn = Path(fn).resolve()
        meta = self.meta()
        with open(fn, "w") as f:
            json.dump(meta, f, indent=4)

    def copy(self):
        return deepcopy(self)
