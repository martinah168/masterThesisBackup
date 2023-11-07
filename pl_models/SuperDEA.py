import sys

from models.unet import BeatGANsEncoderModel
from pl_models.DEA import DAE_LitModel
from utils.enums_model import OptimizerType

sys.path.append("..")
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.tensorboard
from pytorch_lightning.callbacks import EarlyStopping
from torch.cuda.amp import autocast
from torch.optim import Optimizer

from .pl_utils.dist import get_world_size
from monai.utils.misc import set_determinism
from dataloader.dataset_factory import get_dataset
from utils.arguments import DAE_Option

from utils.enums import TrainMode
import pickle
from collections import deque


class SuperDAE_LitModel(pl.LightningModule):
    ###### INIT PROCESS ######
    def __init__(self, conf: DAE_Option, checkpoint_old: str):
        super().__init__()
        assert conf.train_mode.value != TrainMode.manipulate.value
        if conf.seed is not None:
            pl.seed_everything(conf.seed)
            set_determinism(seed=conf.seed)

        self.save_hyperparameters()

        self.conf = conf
        self.model_old: DAE_LitModel = DAE_LitModel.load_from_checkpoint(checkpoint_old)
        self.encoder: BeatGANsEncoderModel = pickle.loads(pickle.dumps(self.model_old.model.encoder))
        self.model_old.requires_grad_(False)
        self.encoder.requires_grad_(False)
        self.model_old.requires_grad_(True)

        self.crit_mse = torch.nn.MSELoss()
        self.crit_cos = torch.nn.CosineSimilarity()
        self.last_100_loss = deque(maxlen=100)
        self.model_old.log = self.log

    def prepare_data(self):
        self.model_old.train_data = get_dataset(self.conf, split="train", super_res=True)
        self.model_old.val_data = get_dataset(self.conf, split="val", super_res=True)

        print("train data:", len(self.model_old.train_data))
        print("val data:", len(self.model_old.val_data))

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("local seed:", seed)
        ##############################################

    ####### DATA LOADER #######
    def train_dataloader(self):
        return self.model_old.train_dataloader()

    def val_dataloader(self):
        return self.model_old.val_dataloader()

    ###### Training #####
    def forward(self, noise=None, x_start=None, ema_model: bool = False, **qargs):
        self.model_old.forward(noise=noise, x_start=x_start, ema_model=ema_model, **qargs)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        self.last_100_loss.append(loss["loss"].detach().cpu().numpy())
        self.log("train/avg_loss", value=np.mean(np.array(self.last_100_loss)).item(), prog_bar=True)
        
        return loss

    def _shared_step(self, batch, batch_idx, step_mode: str):
        with autocast(False):
            losses = {}
            if self.conf.train_mode == TrainMode.diffusion:
                # get augmented version of x_start if given
                img = batch["img"]
                img_lr = batch["img_lr"]
                changed = self.model_old.model.encoder(img_lr)
                with torch.no_grad():
                    gt = self.encoder(img).detach()
                loss_mse = self.crit_mse(gt, changed)
                loss_cos = 1 + (-1 * self.crit_cos(gt, changed))
                losses["loss_cos"] = loss_cos  # TODO test different losses
                losses["loss_mse"] = loss_mse
                losses["loss"] = loss_cos * 0.5 + loss_mse * 0.5
            else:
                raise NotImplementedError()
            losses = {k: v.detach() if k != "loss" else v for k, v in losses.items()}
            loss_keys = list(losses.keys())
            ## divide by accum batches to make the accumulated gradient exact!
            for key in loss_keys:
                losses[key] = self.all_gather(losses[key]).mean()  # type: ignore
            for key in loss_keys:
                self.log(f"loss/{step_mode}_{key}", losses[key].item(), rank_zero_only=True, prog_bar=True)
        return losses

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.model_old.validation_step(batch, batch_idx)

    def on_train_start(self):
        super().on_train_start()
        early_stopping = next(c for c in self.trainer.callbacks if isinstance(c, EarlyStopping))  # type: ignore
        early_stopping.patience = self.conf.early_stopping_patience

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.model_old.on_train_batch_end(outputs, batch, batch_idx)

    #### Optimizer ####
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        self.model_old.on_before_optimizer_step(optimizer)

    def configure_optimizers(self):
        out = {}
        m = self.model_old.model.encoder
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(m.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        elif self.conf.optimizer == OptimizerType.adamW:
            optim = torch.optim.AdamW(m.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        else:
            raise NotImplementedError()
        out["optimizer"] = optim

        return out

    def encode(self, x):
        return self.model_old.encode(x)

    def encode_stochastic(self, x, cond, T=None):
        return self.model_old.encode_stochastic(x, cond, T=T)

    def render(self, noise, cond, x_start=None, T=None):
        return self.model_old.render(noise, cond, x_start=x_start, T=T)
