from typing import Literal
from pytorch_lightning import Trainer
import torch
from pl_models.DEA import DAE_LitModel
from pl_models.SuperDEA import SuperDAE_LitModel
from utils import arguments
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pprint import pprint


def train(opt: arguments.DAE_Option, mode: Literal["train", "eval"] = "train"):
    pprint(opt.__dict__)
    new = opt.new
    opt.new = False
    checkpoint_path = arguments.get_latest_Checkpoint(opt, "*", opt.log_dir)
    assert checkpoint_path is not None
    opt.experiment_name = "SuperDAE_" + opt.experiment_name
    opt.new = new

    model: SuperDAE_LitModel = SuperDAE_LitModel(opt, checkpoint_path)

    if not opt.debug:
        try:
            pass
            model = torch.compile(model)  # type: ignore
        except Exception:
            print("Could not compile, running normally")

    # monitor_str = "loss/val_loss"
    monitor_str = "loss/train_loss"

    checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}_latest",
        monitor=monitor_str,
        mode="min",
        save_last=True,
        save_top_k=3,
        auto_insert_metric_name=True,
        every_n_train_steps=opt.save_every_samples // opt.batch_size_effective,
    )

    early_stopping = EarlyStopping(
        monitor=monitor_str,
        mode="min",
        verbose=False,
        patience=opt.early_stopping_patience,
        # check_on_train_epoch_end=True,
    )
    resume = None
    if not opt.new:
        checkpoint_path = arguments.get_latest_Checkpoint(opt, "*", opt.log_dir)
        if checkpoint_path is not None:
            resume = checkpoint_path
            print(f"Resuming from {resume}")
    logger = TensorBoardLogger(opt.log_dir, name=opt.experiment_name, default_hp_metric=False)

    n_overfit_batches = 1 if opt.overfit else 0.0

    log_every_n_steps = 1 if opt.overfit else opt.log_every_n_steps // opt.batch_size_effective
    gpus = opt.gpus
    accelerator = "gpu"
    if gpus is None:
        gpus = 1
        nodes = 1
    elif -1 in gpus:
        gpus = None
        nodes = 1
        accelerator = "cpu"
    else:
        nodes = len(gpus)
    trainer = Trainer(
        max_steps=opt.total_samples // opt.batch_size_effective,
        devices=gpus,  # type: ignore
        num_nodes=nodes,
        accelerator=accelerator,
        precision=16 if not opt.fp32 else 32,
        callbacks=[checkpoint, early_stopping],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        overfit_batches=n_overfit_batches,
        fast_dev_run=opt.fast_dev_run,
        check_val_every_n_epoch=2500,
    )

    if mode == "train":
        trainer.fit(model, ckpt_path=resume)
    elif mode == "eval":
        raise NotImplementedError(mode)
        ## load the latest checkpoint
        ## perform lpips
        ## dummy loader to allow calling 'test_step'
        # dummy = DataLoader(TensorDataset(torch.tensor([0.0] * opt.batch_size)), batch_size=opt.batch_size)
        # eval_path = opt.eval_path or checkpoint_path
        ## conf.eval_num_images = 50
        # print("loading from:", eval_path)
        # state = torch.load(eval_path, map_location="cpu")
        # print("step:", state["global_step"])
        # model.load_state_dict(state["state_dict"])
        ## trainer.fit(model)
        # out = trainer.test(model, dataloaders=dummy)
        # if len(out) == 0:
        #    # no results where returned
        #    return
        ## first (and only) loader
        # out = out[0]
        # print(out)
        #
        # if get_rank() == 0:
        #    # save to tensorboard
        #    for k, v in out.items():
        #        model.log(k, v)
        #
        #    # # save to file
        #    # # make it a dict of list
        #    # for k, v in out.items():
        #    #     out[k] = [v]
        #    tgt = f"evals/{opt.name}.txt"
        #    dirname = os.path.dirname(tgt)
        #    if not os.path.exists(dirname):
        #        os.makedirs(dirname)
        #    with open(tgt, "a") as f:
        #        f.write(json.dumps(out) + "\n")
        #    # pd.DataFrame(out).to_csv(tgt)
    else:
        raise NotImplementedError()


def get_opt(config=None) -> arguments.DAE_Option:
    torch.cuda.empty_cache()
    opt = arguments.DAE_Option().get_opt(None, config)
    opt = arguments.DAE_Option.from_kwargs(**opt.parse_args().__dict__)
    opt.experiment_name = "DAE_" + opt.experiment_name
    return opt


if __name__ == "__main__":
    train(get_opt())
