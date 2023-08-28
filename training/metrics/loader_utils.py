import os
from multiprocessing import get_context

import torchvision
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.autonotebook import tqdm

from training.config.train import TrainConfig
from training.data.wrapper import SubsetDataset
from training.vis.renderer import *


def make_subset_loader(conf: TrainConfig,
                       dataset: Dataset,
                       batch_size: int,
                       shuffle: bool,
                       parallel: bool,
                       drop_last=True):
    dataset = SubsetDataset(dataset, size=conf.eval_num_images)
    if parallel and distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=conf.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context('fork'),
    )


def loader_to_path(loader: DataLoader, path: str, denormalize: bool):
    # not process safe!

    if not os.path.exists(path):
        os.makedirs(path)

    # write the loader to files
    i = 0
    for batch in tqdm(loader, desc='copy images'):
        imgs = batch['img']
        if denormalize:
            imgs = (imgs + 1) / 2
        for j in range(len(imgs)):
            torchvision.utils.save_image(imgs[j],
                                         os.path.join(path, f'{i+j}.png'))
        i += len(imgs)
