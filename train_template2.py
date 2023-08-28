from pathlib import Path
import sys

from pytorch_lightning.utilities.types import STEP_OUTPUT

root = str(Path(__file__).parent.parent)
sys.path.append(root)
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor


from dataloader.dataset_factory import get_data_2D
from models.model_factory import get_model_2D
from utils import arguments
from typing import Any, Optional, Tuple
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


def mean(a):
    return sum(a) / len(a)


class Age_prediction(pl.LightningModule):
    def __init__(self, opt: arguments.Age_Prediction_Option, max_v, min_v, labels) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.learning_rate = opt.lr
        # TODO load network
        self.model = get_model_2D(opt, labels)
        self.loss_func = nn.L1Loss()  # .MSELoss()
        self.max_v = max_v
        self.min_v = min_v
        self.labels = labels
        self.reset_loss()
        assert len(labels) == len(min_v), (len(labels), len(min_v), len(max_v))
        assert len(labels) == len(max_v), (len(labels), len(min_v), len(max_v))
        self.num_batches = -1

    @torch.no_grad()
    def reset_loss(self):
        self.running_abs_loss_train = [[] for _ in self.labels]
        self.running_abs_loss_val = [[] for _ in self.labels]
        self.running_abs_loss_gt = [[] for _ in self.labels]
        self.running_abs_loss_pred = [[] for _ in self.labels]
        self.train_loss = []
        self.val_loss = []

    def configure_optimizers(self):
        lr = self.opt.lr = self.learning_rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.opt.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, "min", verbose=True)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, "min", verbose=True, factor=0.5, patience=2),
            "monitor": "val/PatientAge",  # Default: val_loss
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    @torch.no_grad()
    def compute_abs_loss(self, gt: Tensor, pred: Tensor, out: list[list], log):
        for jm, (mi, ma) in enumerate(zip(self.min_v, self.max_v)):
            gt2 = (gt[:, jm] + 1) / 2 * (ma - mi) + mi
            pred2 = (pred[:, jm] + 1) / 2 * (ma - mi) + mi
            abs_loss = torch.abs(gt2 - pred2)
            out[jm].append(abs_loss.mean().item())
            if log != "train":
                self.running_abs_loss_gt[jm].append(gt2.reshape(-1).cpu())
                self.running_abs_loss_pred[jm].append(pred2.reshape(-1).cpu())
        for jm, a in enumerate(out):
            self.log(f"{log}/{self.labels[jm]}", mean(a[-100 * self.opt.batch_size :]), prog_bar=True)

    def training_step(self, train_batch: Batch_Type, batch_idx):
        x, gt = train_batch
        assert len(gt.shape) == 2, gt.shape
        pred = self.model(x)
        self.compute_abs_loss(gt, pred, self.running_abs_loss_train, log="train")
        loss = self.loss_func(pred, gt)
        loss: Tensor = loss.mean()
        self.train_loss.append(loss.cpu().item())
        self.log("train/loss", mean(self.train_loss[-100:]), prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch: Batch_Type, batch_idx):
        vx, gt = batch
        pred = self.model(vx)  #
        self.compute_abs_loss(gt, pred, self.running_abs_loss_val, log="val")
        loss: Tensor = self.loss_func(pred.squeeze(), gt.squeeze()).mean()
        self.val_loss.append(loss.cpu().item())
        self.log("val/loss", loss.cpu().item(), prog_bar=True)

        return loss

    def make_scatter(self, phase="val"):
        print("make_scatter", phase)
        for id, label in enumerate(self.labels):
            make_scatter(
                str(Path(self.opt.log_dir, self.opt.experiment_name)),
                self.running_abs_loss_gt[id],
                self.running_abs_loss_pred[id],
                alpha=0.5,
                prefix_epoch=self.opt.model_name.name + f"_{label}_{phase}",
                epoch="",
            )

    @torch.no_grad()
    def on_validation_end(self):
        val = sum(self.val_loss) / len(self.val_loss)
        self.logger: TensorBoardLogger

        self.logger.experiment.add_scalar("val/loss_total", val, self.current_epoch)  # type: ignore
        self.make_scatter(phase="val")
        self.reset_loss()
        return val

    @torch.no_grad()
    def test_step(self, batch: Batch_Type, idx) -> STEP_OUTPUT | None:
        self.validation_step(batch, idx)
        if idx == int(self.num_batches) - 1:
            self.log("num_batches", float(idx))
            self.test_end()

    def test_end(self):
        out = {}
        for jm, a in enumerate(self.running_abs_loss_val):
            a.append(jm)
            out[self.labels[jm]] = mean(a)
            self.log(f"test/{self.labels[jm]}", out[self.labels[jm]])

        val = sum(self.val_loss) / len(self.val_loss)
        self.logger: TensorBoardLogger
        self.logger.experiment.add_scalar("val/loss_total", val)  # type: ignore
        self.make_scatter(phase="test")
        self.reset_loss()


def main(opt: arguments.Age_Prediction_Option, limit_train_batches=1):
    #### Define dataset ####
    from torch.utils.data import DataLoader

    train_ds, val_ds, _ = get_data_2D(opt)
    train_loader = DataLoader(
        train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_cpu, drop_last=True, persistent_workers=True
    )
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size_val, shuffle=False, num_workers=opt.num_cpu, persistent_workers=True)

    model = Age_prediction(opt=opt, max_v=train_ds.max_v, min_v=train_ds.min_v, labels=train_ds.target_label)  # type: ignore

    # Get last checkpoint. If there is non or --new was called this returns None and starts a new model.
    last_checkpoint = arguments.get_latest_Checkpoint(opt, log_dir_name=opt.log_dir, best=False)
    ### We do not reload with trainer.fit, as my
    model.load_from_checkpoint(last_checkpoint) if last_checkpoint is not None else None
    last_checkpoint = None

    # Define Last and best Checkpoints to be saved.
    mc_last = ModelCheckpoint(
        filename="{epoch}-{step}-loss_{train/loss:.8f}_latest",
        monitor="step",
        mode="max",
        every_n_train_steps=min(500, len(train_loader)),
        save_top_k=3,
        auto_insert_metric_name=False,
    )

    mc_best = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        filename="{epoch}-{step}-loss={val/loss:.8f}_best",
        every_n_train_steps=len(train_loader) + 1,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    # This sets the experiment name. The model is in /lightning_logs/{opt.exp_nam}/version_*/checkpoints/
    logger = TensorBoardLogger(opt.log_dir, name=opt.experiment_name, default_hp_metric=False)
    limit_train_batches = limit_train_batches if limit_train_batches != 1 else None

    gpus = opt.gpus
    accelerator = "gpu"
    if gpus is None:
        gpus = 1
    elif -1 in gpus:
        gpus = None
        accelerator = "cpu"
    # training

    early_stop_callback = EarlyStopping(monitor="val/PatientAge", patience=20, verbose=False, mode="min")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=gpus,  # type: ignore
        # num_nodes=1,  # Train on 'n' GPUs; 0 is CPU
        limit_train_batches=limit_train_batches,  # Train only x % (if float) or train only on x batches (if int)
        max_epochs=opt.num_epochs,  # Stopping epoch
        logger=logger,
        callbacks=[mc_last, mc_best, early_stop_callback],  # mc_best
        detect_anomaly=opt.prevent_nan,
        val_check_interval=1 / 3.0
        # auto_lr_find=opt.auto_lr_find,
    )
    # if opt.auto_lr_find:
    #    trainer.tune(
    #        model,
    #        train_loader,
    #        val_loader,
    #    )
    #    model.learning_rate *= 0.5
    #    try:
    #        next(Path().glob(".lr_find*")).unlink()
    #    except StopIteration:
    #        pass

    trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint)


def get_opt(config=None) -> arguments.Age_Prediction_Option:
    torch.cuda.empty_cache()
    opt = arguments.Age_Prediction_Option().get_opt(None, config)
    opt = arguments.Age_Prediction_Option.from_kwargs(**opt.parse_args().__dict__)
    opt.experiment_name = "Age_" + opt.experiment_name
    return opt


if __name__ == "__main__":
    main(get_opt())
