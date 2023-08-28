import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Literal

import monai
import monai.networks.nets
import numpy as np
import pytorch_lightning as pl
import torch
import torch._dynamo as dynamo
import torch._dynamo.config
import torch.utils.data
import torchmetrics
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

import wandb
from training.config.train import TrainConfig, data_paths
from training.data.celeb import (CelebAttrDataset, CelebD2CAttrFewshotDataset,
                                 CelebHQAttrDataset)
from training.data.glioma_public import PublicGliomaDataset
from training.data.wrapper import Repeat
from training.dist import get_world_size
from training.mode.clf import ClfMode
from training.mode.loss import ManipulateLossType
from training.mode.manipulate import ManipulateMode
from training.mode.train import TrainMode
from training.models.unet_autoenc import BeatGANsAutoencModel
from training.scheduler.cosine import LightningCosineLRScheduler
from training.vis.confmat import get_confmat_image

torch._dynamo.config.verbose = True
torch._dynamo.config.verify_correctness = True


class ZipLoader:

    def __init__(self, loaders):
        self.loaders = loaders

    def __len__(self):
        return len(self.loaders[0])

    def __iter__(self):
        for each in zip(*self.loaders):
            yield each


class ClsModel(pl.LightningModule):

    def __init__(self, conf: TrainConfig):
        super().__init__()
        assert conf.train_mode.is_manipulate()
        if conf.seed is not None:
            pl.seed_everything(conf.seed)
            monai.utils.misc.set_determinism(seed=conf.seed)

        self.save_hyperparameters(conf.as_dict_jsonable())
        if self.logger is not None:
            print('hyperparameters will be saved to',
                  os.path.join(conf.logdir, 'hparams.yaml'))
        self.conf = conf

        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)

        # preparations
        if conf.train_mode == TrainMode.manipulate:
            # this is only important for training!
            # the latent is freshly inferred to make sure it matches the image
            # manipulating latents require the base model
            self.model = conf.make_model_conf().make_model()
            self.ema_model = copy.deepcopy(self.model)

            self.model.requires_grad_(False)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

            self.encoder = self.model.encoder if hasattr(
                self.model, 'encoder') else self.model
            self.ema_encoder = self.ema_model.encoder if hasattr(
                self.ema_model, 'encoder') else self.ema_model

            if self.conf.full_fine_tuning:
                # fine-tune the whole model
                self.encoder.requires_grad_(True)

            if conf.pretrain is not None:
                print(
                    f'loading pretrain ... {conf.pretrain.name} from {conf.pretrain.path}'
                )
                state = torch.load(conf.pretrain.path, map_location='cpu')
                print('step:', state['global_step'])
                self.load_state_dict(state['state_dict'], strict=False)

            # load the latent stats
            if conf.manipulate_znormalize:
                print('loading latent stats ...')
                state = torch.load(conf.latent_infer_path)
                self.conds = state['conds']
                self.register_buffer('conds_mean', state['conds_mean'][None, :])
                self.register_buffer('conds_std', state['conds_std'][None, :])
            else:
                self.conds_mean = None
                self.conds_std = None

        if conf.manipulate_mode in [ManipulateMode.celebahq_all]:
            num_cls = len(CelebAttrDataset.id_to_cls)
        elif conf.manipulate_mode.is_single_class() or conf.clf_mode in [
                ClfMode.one_vs_all, ClfMode.one_vs_one
        ] or conf.use_healthy:
            num_cls = 1
        elif conf.manipulate_mode == ManipulateMode.gliomapublic:
            num_cls = PublicGliomaDataset.num_classes
        else:
            raise NotImplementedError()

        # classifier
        if conf.train_mode == TrainMode.manipulate:
            # latent manipluation requires only a linear classifier
            self.classifier = nn.Linear(conf.style_ch, num_cls)
        elif conf.train_mode == TrainMode.supervised:
            first_densenet_block = (6,) if conf.img_size > 32 else ()
            cls_init = {
                'resnet18':
                    lambda num_classes: monai.networks.nets.resnet18(
                        n_input_channels=len(conf.mri_sequences),
                        num_classes=num_classes),
                'resnet50':
                    lambda num_classes: monai.networks.nets.resnet50(
                        n_input_channels=len(conf.mri_sequences),
                        num_classes=num_classes),
                'vit':
                    lambda num_classes: monai.networks.nets.ViT(
                        in_channels=len(conf.mri_sequences),
                        num_classes=num_classes,
                        img_size=conf.img_size,
                        patch_size=8,  # 16
                        hidden_size=512,  # 768
                        num_heads=8,  # 12
                        mlp_dim=2048,  # 3072
                        classification=True,
                        post_activation=None  # make sure to use logits
                    ),
                'densenet':
                    lambda num_classes: monai.networks.nets.DenseNet121(
                        spatial_dims=3,
                        in_channels=len(conf.mri_sequences),
                        out_channels=num_classes,
                        block_config=first_densenet_block +
                        (12, 24, 16),  # (6, 12, 24, 16),
                        init_features=64,
                        growth_rate=32,
                    ),
                'densenet169':
                    lambda num_classes: monai.networks.nets.DenseNet169(
                        spatial_dims=3,
                        in_channels=len(conf.mri_sequences),
                        out_channels=num_classes,
                        block_config=first_densenet_block +
                        (12, 32, 32),  # (6, 12, 32, 32)
                        init_features=64,
                        growth_rate=32,
                    ),
                'densenet201':
                    lambda num_classes: monai.networks.nets.DenseNet201(
                        spatial_dims=3,
                        in_channels=len(conf.mri_sequences),
                        out_channels=num_classes,
                        block_config=first_densenet_block +
                        (12, 48, 32),  #  (6, 12, 48, 32),
                        init_features=64,
                        growth_rate=32,
                    ),
                'densenet264':
                    lambda num_classes: monai.networks.nets.DenseNet264(
                        spatial_dims=3,
                        in_channels=len(conf.mri_sequences),
                        out_channels=num_classes,
                        block_config=first_densenet_block +
                        (12, 64, 48),  #  (6, 12, 64, 48),
                        init_features=64,
                        growth_rate=32,
                    ),
            }[conf.clf_arch]

            self.classifier = cls_init(num_classes=num_cls)

        else:
            raise NotImplementedError()

        # num params
        num_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        self.ema_classifier = copy.deepcopy(self.classifier)

        # create class histogram for train and val
        run_modes = ['train', 'val', 'test']

        for mode in run_modes:
            # create dataset
            data = self.load_dataset(split=mode)
            setattr(self, f'{mode}_data', data)

            cls_counter = Counter(data.cls_labels.values())
            cls_hist = [cls_counter[k] for k in sorted(cls_counter.keys())]
            cls_hist_dist = [c / sum(cls_hist) for c in cls_hist]

            print(f'{mode} class histogram:',
                  [f'{c:.2f}' for c in cls_hist_dist])

        #metrics

        self.clf_task = 'multiclass'
        metric_kwargs = dict(task=self.clf_task,
                             num_classes=num_cls,
                             average='macro')

        if self.conf.clf_mode in [ClfMode.one_vs_all, ClfMode.one_vs_one
                                 ] or conf.use_healthy:
            self.clf_task = 'binary'
            metric_kwargs = dict(task=self.clf_task)

        self.metric_names = {
            'acc': torchmetrics.Accuracy,
            'prec': torchmetrics.Precision,
            'rec': torchmetrics.Recall,
            'f1': torchmetrics.F1Score,
            'mcc': torchmetrics.MatthewsCorrCoef,
        }

        # create metrics for each run mode
        for m in run_modes:
            for metric_name, metric_fn in self.metric_names.items():
                if metric_name == 'mcc':
                    average = metric_kwargs.pop('average')
                setattr(self, f'{m}_{metric_name}', metric_fn(**metric_kwargs))
                if metric_name == 'mcc':
                    metric_kwargs['average'] = average

            if m != 'train':
                # create confusion matrix for each run mode
                average = metric_kwargs.pop('average')

                metric_obj = torchmetrics.ConfusionMatrix(**metric_kwargs)
                setattr(self, f'{m}_conf_mat', metric_obj)
                metric_kwargs['average'] = average

                split_mode = getattr(self, f'{m}_data').split_mode
                # create metrics for each study
                for study in PublicGliomaDataset.subset_names_dict[split_mode][
                        m]:
                    # replace tum_glioma with glioma_epic
                    study = study.replace('tum_glioma', 'glioma_epic')

                    for metric_name, metric_fn in self.metric_names.items():
                        study_metric_name = f'{m}_{metric_name}_{study}'
                        if metric_name == 'mcc':
                            average = metric_kwargs.pop('average')

                        setattr(self, study_metric_name,
                                metric_fn(**metric_kwargs))
                        if metric_name == 'mcc':
                            metric_kwargs['average'] = average

        if self.clf_task == 'multiclass':
            # add additional metrics which calculate each metric on a per-class basis
            metric_kwargs['average'] = None

            for m in run_modes:
                for metric_name, metric_fn in ((m, v) for (
                        m, v) in self.metric_names.items() if m != 'mcc'):
                    setattr(self, f'{m}_{metric_name}_per_class',
                            metric_fn(**metric_kwargs))

    def state_dict(self, *args, **kwargs):
        # don't save the base model
        out = {}
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('model.'):
                pass
            elif k.startswith('ema_model.'):
                pass
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = None):
        if self.conf.train_mode == TrainMode.manipulate:
            # change the default strict => False
            if strict is None:
                strict = False
        else:
            if strict is None:
                strict = True
        return super().load_state_dict(state_dict, strict=strict)

    def normalize(self, cond):
        cond = (cond - self.conds_mean.to(self.device)) / self.conds_std.to(
            self.device)
        return cond

    def denormalize(self, cond):
        cond = (cond * self.conds_std.to(self.device)) + self.conds_mean.to(
            self.device)
        return cond

    def load_dataset(self, **kwargs):
        if self.conf.manipulate_mode == ManipulateMode.d2c_fewshot:
            return CelebD2CAttrFewshotDataset(
                cls_name=self.conf.manipulate_cls,
                K=self.conf.manipulate_shots,
                img_folder=data_paths['celeba'],
                img_size=self.conf.img_size,
                seed=self.conf.manipulate_seed,
                all_neg=False,
                do_augment=True,
            )
        elif self.conf.manipulate_mode == ManipulateMode.d2c_fewshot_allneg:
            # positive-unlabeled classifier needs to keep the class ratio 1:1
            # we use two dataloaders, one for each class, to stabiliize the training
            img_folder = data_paths['celeba']

            return [
                CelebD2CAttrFewshotDataset(
                    cls_name=self.conf.manipulate_cls,
                    K=self.conf.manipulate_shots,
                    img_folder=img_folder,
                    img_size=self.conf.img_size,
                    only_cls_name=self.conf.manipulate_cls,
                    only_cls_value=1,
                    seed=self.conf.manipulate_seed,
                    all_neg=True,
                    do_augment=True),
                CelebD2CAttrFewshotDataset(
                    cls_name=self.conf.manipulate_cls,
                    K=self.conf.manipulate_shots,
                    img_folder=img_folder,
                    img_size=self.conf.img_size,
                    only_cls_name=self.conf.manipulate_cls,
                    only_cls_value=-1,
                    seed=self.conf.manipulate_seed,
                    all_neg=True,
                    do_augment=True),
            ]
        elif self.conf.manipulate_mode == ManipulateMode.celebahq_all:
            return CelebHQAttrDataset(path=data_paths['celebahq'],
                                      image_size=self.conf.img_size,
                                      attr_path=data_paths['celebahq_anno'],
                                      do_augment=True)
        elif self.conf.manipulate_mode == ManipulateMode.gliomapublic:
            return PublicGliomaDataset(data_dir=Path(
                data_paths[self.conf.data_name]),
                                       img_size=self.conf.img_size,
                                       mri_sequences=self.conf.mri_sequences,
                                       mri_crop=self.conf.mri_crop,
                                       train_mode=self.conf.train_mode,
                                       split_ratio=self.conf.split_ratio,
                                       manipulate_cls=self.conf.manipulate_cls,
                                       use_healthy=self.conf.use_healthy,
                                       with_data_aug=self.conf.with_data_aug,
                                       split_mode=self.conf.split_mode,
                                       data_aug_prob=self.conf.data_aug_prob,
                                       **kwargs)

        else:
            raise NotImplementedError()

    def setup(self, stage=None) -> None:
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        if self.conf.manipulate_mode.is_fewshot():
            # repeat the dataset to be larger (speed up the training)
            if isinstance(self.train_data, list):
                # fewshot-allneg has two datasets
                # we resize them to be of equal sizes
                a, b = self.train_data
                self.train_data = [
                    Repeat(a, max(len(a), len(b))),
                    Repeat(b, max(len(a), len(b))),
                ]
            else:
                self.train_data = Repeat(self.train_data, 100_000)

    def _shared_loader(self, mode: str, batch_size: int = None):
        conf = self.conf.clone()

        if batch_size is None:
            if mode == 'train':
                batch_size = self.batch_size
            else:
                batch_size = self.conf.batch_size_eval
        data = getattr(self, f'{mode}_data')

        if isinstance(data, list):
            dataloader = []
            for each in data:
                dataloader.append(
                    conf.make_loader(each, shuffle=True, drop_last=True))
            dataloader = ZipLoader(dataloader)
        else:
            dataloader = conf.make_loader(data,
                                          shuffle=mode == 'train',
                                          drop_last=mode == 'train',
                                          batch_size=batch_size,
                                          mode=mode)
        return dataloader

    def train_dataloader(self, batch_size: int = None):
        # make sure to use the fraction of batch size
        # the batch size is global!
        return self._shared_loader('train', batch_size)

    def val_dataloader(self, batch_size: int = None):
        # make sure to use the fraction of batch size
        # the batch size is global!
        return self._shared_loader('val', batch_size)

    def test_dataloader(self, batch_size: int = None):
        return self._shared_loader('test', batch_size)

    @property
    def batch_size(self):
        ws = get_world_size()
        assert self.conf.batch_size % ws == 0
        return self.conf.batch_size // ws

    def _shared_step(self, batch, batch_idx: int, step_name: str):
        self.ema_model: BeatGANsAutoencModel
        if isinstance(batch, tuple):
            a, b = batch
            imgs = torch.cat([a['img'], b['img']])
            labels = torch.cat([a['labels'], b['labels']])
        else:
            imgs = batch['img']
            # print(f'({self.global_rank}) imgs:', imgs.shape)
            labels = batch['labels']

        if self.conf.train_mode.is_manipulate():
            if not self.conf.train_mode == TrainMode.supervised:
                # self.ema_encoder.eval()
                self.encoder.eval()

                with torch.no_grad():
                    # (n, c)
                    # latent = self.ema_encoder(imgs)
                    latent = self.encoder(imgs)

                if self.conf.manipulate_znormalize:
                    latent = self.normalize(latent)
            else:
                # classify images directly
                latent = imgs

            # (n, cls)
            with torch.set_grad_enabled(step_name == 'train'):
                pred = self.classifier(latent)
            with torch.no_grad():
                # we dont need gradients for the ema updated classifier
                pred_ema = self.ema_classifier(latent)

        elif self.conf.train_mode == TrainMode.manipulate_img:
            # (n, cls)
            pred = self.classifier(imgs)
            pred_ema = None
        elif self.conf.train_mode == TrainMode.manipulate_imgt:
            t, weight = self.T_sampler.sample(len(imgs), imgs.device)
            imgs_t = self.sampler.q_sample(imgs, t)
            pred = self.classifier(imgs_t, t=t)
            pred_ema = None
            print('pred:', pred.shape)
        else:
            raise NotImplementedError()

        if self.conf.clf_arch == 'vit':
            # throw away hidden activations
            pred = pred[0]
            pred_ema = pred_ema[0]

        if self.conf.clf_mode == ClfMode.one_vs_one or self.conf.use_healthy:
            # multi-class classification
            self.train_data: PublicGliomaDataset
            # convert labels to one hot encoding
            gt = labels.float().view(-1, 1)
        elif self.conf.clf_mode == ClfMode.one_vs_all:
            # one vs all classification
            # map int labels to one hot labels

            # manipulate cls has length one ==> take the first element
            manipulate_cls = int(self.conf.manipulate_cls[0])
            gt = torch.where(labels == manipulate_cls, 1., 0.).view(pred.size())

        elif self.conf.clf_mode == ClfMode.multi_class:
            # multi-class classification
            self.train_data: PublicGliomaDataset
            gt = labels.long()
        else:
            raise NotImplementedError(f'Unknown clf_mode: {self.conf.clf_mode}')

        if self.conf.manipulate_loss == ManipulateLossType.bce:
            if self.conf.clf_mode in [ClfMode.one_vs_all, ClfMode.one_vs_one
                                     ] or self.conf.use_healthy:
                loss_fn = F.binary_cross_entropy_with_logits
            elif self.conf.clf_mode == ClfMode.multi_class:
                loss_fn = F.cross_entropy  # -> with logits
            else:
                raise NotImplementedError(
                    f'Unknown clf_mode: {self.conf.clf_mode}')

            loss = loss_fn(pred, gt)
            if pred_ema is not None:
                loss_ema = loss_fn(pred_ema, gt)
        elif self.conf.manipulate_loss == ManipulateLossType.mse:
            loss = F.mse_loss(pred, gt)
            if pred_ema is not None:
                loss_ema = F.mse_loss(pred_ema, gt)
        else:
            raise NotImplementedError()

        self.log(
            f'loss/{step_name}_bce_loss',
            loss.item(),
        )
        self.log(
            f'loss/{step_name}_bce_loss_ema',
            loss_ema.item(),
        )
        if self.training:
            self.log('lr', self.optimizers().param_groups[0]['lr'])

        pred_oh = pred.argmax(
            dim=1) if pred.size(1) > 1 else torch.sigmoid(pred)

        # log metric averages
        for metric_name in self.metric_names:
            m_metric_name = f'{step_name}_{metric_name}'
            metric_fn = getattr(self, m_metric_name)
            metric_val = metric_fn(pred_oh, gt)
            self.log(
                f'cls_metrics/{m_metric_name}',
                metric_fn,
            )

        if self.clf_task == 'multiclass':
            # additionally log per class metrics
            for metric_name in (m for m in self.metric_names if m != 'mcc'):
                m_metric_name = f'{step_name}_{metric_name}_per_class'
                metric_fn: torchmetrics.Metric = getattr(self, m_metric_name)
                if step_name == 'train':
                    metric_val = metric_fn(pred_oh, gt)

                    self._log_per_class(step_name, metric_name, metric_val)
                else:
                    # update metric
                    metric_fn.update(pred_oh, gt)

        if step_name != 'train':
            # add logging confusion matrix
            m_metric_name = f'{step_name}_conf_mat'
            metric_fn = getattr(self, m_metric_name)
            metric_val = metric_fn(pred_oh, gt)
            # aggregate metrics by study during validation and test
            batch_studies = [s.split('/')[0] for s in batch['patient_id']]

            # determine the indices in the batch for each study
            batch_study_indices = {}
            for i_study, study in enumerate(batch_studies):
                if study not in batch_study_indices:
                    batch_study_indices[study] = []
                batch_study_indices[study].append(i_study)

            # calculate metrics per study
            for study, indices in batch_study_indices.items():
                # get predictions and gt for this study
                study_pred = pred_oh[indices]
                study_gt = gt[indices]

                # calculate metrics for this study
                for metric_name in self.metric_names:
                    m_metric_name = f'{step_name}_{metric_name}_{study}'

                    metric_fn = getattr(self, m_metric_name)
                    metric_fn(study_pred, study_gt)
                    self.log(
                        f'cls_metrics_per_study/{m_metric_name}',
                        metric_fn,
                    )

        return loss

    def _log_per_class(self, step_name: str, metric_name: str,
                       metric_val: torch.Tensor) -> None:

        for i_cls, val in enumerate(metric_val):
            # log per class metrics
            m_metric_name = f'{step_name}_{metric_name}'
            # during training: log val
            self.log(
                f'cls_metrics_per_class/{m_metric_name}_{i_cls}',
                val.item(),
            )

    def training_step(self, batch, batch_idx):
        # explanation, out_guards, graphs, ops_per_graph = dynamo.explain(self._shared_step, batch, batch_idx, 'train')
        # print(explanation)
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')

    def _on_shared_eval_epoch_end(self, step_name: str):
        # add logging confusion matrix
        m_metric_name = f'{step_name}_conf_mat'
        confmat_obj: torchmetrics.Metric = getattr(self, m_metric_name)
        # get image of confusion matrix
        img_confmat = get_confmat_image(confmat_obj.compute(), mode=step_name)
        # save image to logdir
        save_image(img_confmat, f'{self.conf.logdir}/{m_metric_name}.png')

        self.log_image(f'cls_metrics/{m_metric_name}',
                       img_confmat,
                       step=self.global_step)

        confmat_obj.reset()

        # log per class metrics
        if self.clf_task == 'multiclass':
            # additionally log per class metrics
            for metric_name in (m for m in self.metric_names if m != 'mcc'):
                m_metric_name = f'{step_name}_{metric_name}_per_class'
                metric_fn: torchmetrics.Metric = getattr(self, m_metric_name)
                # aggregate metrics during validation and test
                metric_val_epoch = metric_fn.compute()
                self._log_per_class(step_name, metric_name, metric_val_epoch)
                # reset metric after one epoch
                metric_fn.reset()

    def on_validation_epoch_end(self):
        self._on_shared_eval_epoch_end(step_name='val')

    def on_test_epoch_end(self):
        self._on_shared_eval_epoch_end(step_name='test')

    def on_train_epoch_end(self):
        for metric_name in (m for m in self.metric_names if m != 'mcc'):
            # reset per class metrics
            m_metric_name = f'train_{metric_name}_per_class'
            metric_fn: torchmetrics.Metric = getattr(self, m_metric_name)
            metric_fn.reset()

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        ema(self.classifier, self.ema_classifier, self.conf.ema_decay)
        if self.conf.full_fine_tuning:
            ema(self.encoder, self.ema_encoder, self.conf.ema_decay)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.classifier.parameters(),
                                 lr=self.conf.lr,
                                 weight_decay=self.conf.weight_decay)
        out = {
            'optimizer': optim,
        }
        if self.conf.clf_arch == 'vit':
            out['lr_scheduler'] = {
                'scheduler':
                    LightningCosineLRScheduler(
                        optim,
                        warmup_t=self.conf.warmup,
                        t_initial=self.conf.total_samples //
                        len(self.train_data),
                    ),
                'interval':
                    'epoch',
            }

        return out

    def on_train_start(self):
        super().on_train_start()
        early_stopping = next(
            c for c in self.trainer.callbacks if isinstance(c, EarlyStopping))
        early_stopping.patience = self.conf.clf_early_stopping_patience

    def log_image(self,
                  tag: str,
                  image: torch.Tensor,
                  step: int,
                  on_step: bool = None,
                  on_epoch: bool = None) -> None:

        if self.global_rank == 0:
            experiment = self.logger.experiment
            if isinstance(experiment, torch.utils.tensorboard.SummaryWriter):
                experiment.add_image(tag, image, step)
            elif isinstance(experiment, wandb.sdk.wandb_run.Run):
                experiment.log({tag: [wandb.Image(image.cpu())]},
                               #step=step,
                              )


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


def get_trainer(conf: TrainConfig, gpus: int):
    print('conf:', conf.name, 'version:', conf.version)
    model = ClsModel(conf)
    # model = torch.compile(model , mode="reduce-overhead")

    monitor_str = 'loss/val_bce_loss'

    checkpoint = ModelCheckpoint(monitor=monitor_str,
                                 mode='min',
                                 dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=1,
                                 verbose=True,
                                 auto_insert_metric_name=True)
    early_stopping = EarlyStopping(monitor=monitor_str,
                                   mode='min',
                                   verbose=True,
                                   patience=conf.clf_early_stopping_patience)

    checkpoint_path = f'{conf.logdir}/last.ckpt'
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print(f'Resuming from {resume}')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None
    wandb_resume_kwargs = {}
    wandb_id_fp = f'{conf.logdir}/wandb_id.txt'
    if os.path.exists(wandb_id_fp):
        # read wandb id
        wandb_resume_kwargs = {'resume': 'must'}
        # load wandb id
        with open(wandb_id_fp, 'r') as f:
            conf.wandb_id = f.read().strip()
    else:
        # save wandb id to checkpoints dir
        with open(wandb_id_fp, 'w') as f:
            f.write(conf.wandb_id)

    n_overfit_batches = 1 if conf.overfit else 0.0
    log_every_n_steps = 1 if conf.overfit else min(
        10,
        len(model.train_data) // conf.batch_size)

    fast_dev_run = 'debug' in conf.name
    if fast_dev_run:
        log_every_n_steps = 1

    logger = None
    if not fast_dev_run:
        logger = pl_loggers.WandbLogger(project=conf.wandb_project,
                                        config=conf.as_dict_jsonable(),
                                        save_code=False,
                                        log_model=False,
                                        job_type='cls',
                                        id=conf.wandb_id,
                                        **wandb_resume_kwargs)
        _ = logger.experiment  # init wandb

    trainer = pl.Trainer(max_steps=conf.total_samples //
                         conf.batch_size_effective,
                         precision='16-mixed',
                         callbacks=[checkpoint, early_stopping],
                         logger=logger,
                         accumulate_grad_batches=conf.accum_batches,
                         log_every_n_steps=log_every_n_steps,
                         overfit_batches=n_overfit_batches,
                         fast_dev_run=fast_dev_run,
                         gradient_clip_val=conf.grad_clip,
                         gradient_clip_algorithm='norm')
    return trainer, model


def run_cls(conf: TrainConfig, gpus):
    trainer, model = get_trainer(conf, gpus)
    if not conf.test_only:

        trainer.fit(model, ckpt_path=conf.continue_from)
    else:

        if conf.ckpt_state == 'best':
            checkpoint = torch.load(f'{conf.logdir}/last.ckpt',
                                    map_location=lambda storage, loc: storage)

            load_ckpt = checkpoint['callbacks'][
                trainer.checkpoint_callback.state_key]['best_model_path']
            # if model was trained on another machine, the path will be wrong
            load_ckpt = (Path(conf.logdir) /
                         Path(load_ckpt).name).resolve().as_posix()
            print(f'Loading best checkpoint: {load_ckpt}')
        elif conf.ckpt_state == 'last':
            load_ckpt = f'{conf.logdir}/last.ckpt'
        else:
            raise ValueError(f'Unknown ckpt_state: {conf.ckpt_state}')

        checkpoint = torch.load(f'{conf.logdir}/last.ckpt',
                                map_location=lambda storage, loc: storage)
        model = ClsModel.load_from_checkpoint(load_ckpt, conf=conf)
        # last_model = ClsModel.load_from_checkpoint(f'{conf.logdir}/last.ckpt', conf=conf)

    if 'debug' in conf.name and not conf.test_only:
        print('Skipping test due to debug mode')
        return
    results = test(conf, trainer, model)

    results_json = json.dumps(results, indent=4)

    print('results:', results_json)
    # save results json
    results_fp = f'{conf.logdir}/results{"_last" if conf.ckpt_state == "last" else ""}.json'
    results_fp_new = results_fp
    i_exists = 0
    while os.path.exists(results_fp_new):
        # add a number to the end to avoid overwriting existing results file
        results_fp_new = results_fp.replace('.json', f'_{i_exists}.json')
        i_exists += 1

    with open(results_fp_new, 'w') as json_f:
        json_f.write(results_json)
    print(f'Saved results to {results_fp_new}')

    # return final val loss wandb sweeps
    return results['loss/val_bce_loss']


def test(conf: TrainConfig, trainer: pl.Trainer,
         model: ClsModel) -> dict[str, float]:
    print('conf:', conf.name, 'version:', conf.version)
    params = dict(
        model=model,  # already loaded from best checkpoint
        verbose=True,
    )
    results = {}
    val_results = trainer.validate(
        dataloaders=model.val_dataloader(batch_size=conf.batch_size_eval),
        **params,
    )[0]
    results |= val_results

    test_results = trainer.test(
        dataloaders=model.test_dataloader(batch_size=conf.batch_size_eval),
        **params)[0]
    results |= test_results

    return results
