import os
import shutil

import torch
import torchvision
from pytorch_fid import fid_score
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm, trange

from training.config.train import TrainConfig
from training.dist import (barrier, broadcast, chunk_size, get_rank,
                           get_world_size)
from training.metrics.loader_utils import loader_to_path, make_subset_loader
from training.mode.model import ModelType
from training.models import Model
from training.models.diffusion import Sampler
from training.models.unet_autoenc import BeatGANsAutoencModel
from training.vis.renderer import render_condition, render_uncondition


def evaluate_fid(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    train_data: Dataset,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    conds_mean=None,
    conds_std=None,
    remove_cache: bool = True,
    clip_latent_noise: bool = False,
):
    assert conf.fid_cache is not None
    if get_rank() == 0:
        # no parallel
        # validation data for a comparing FID
        val_loader = make_subset_loader(conf,
                                        dataset=val_data,
                                        batch_size=conf.batch_size_eval,
                                        shuffle=False,
                                        parallel=False)

        # put the val images to a directory
        cache_dir = f'{conf.fid_cache}_{conf.eval_num_images}'
        if (os.path.exists(cache_dir) and
                len(os.listdir(cache_dir)) < conf.eval_num_images):
            shutil.rmtree(cache_dir)

        if not os.path.exists(cache_dir):
            # write files to the cache
            # the images are normalized, hence need to denormalize first
            loader_to_path(val_loader, cache_dir, denormalize=True)

        # create the generate dir
        if os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)
        os.makedirs(conf.generate_dir)

    barrier()

    world_size = get_world_size()
    rank = get_rank()
    batch_size = chunk_size(conf.batch_size_eval, rank, world_size)

    def filename(idx):
        return world_size * idx + rank

    model.eval()
    with torch.no_grad():
        if conf.model_type.can_sample():
            eval_num_images = chunk_size(conf.eval_num_images, rank, world_size)
            desc = 'generating images'
            for i in trange(0, eval_num_images, batch_size, desc=desc):
                batch_size = min(batch_size, eval_num_images - i)
                x_T = torch.randn((batch_size, 3, conf.img_size, conf.img_size),
                                  device=device)
                batch_images = render_uncondition(conf=conf,
                                                  model=model,
                                                  x_T=x_T,
                                                  sampler=sampler,
                                                  latent_sampler=latent_sampler,
                                                  conds_mean=conds_mean,
                                                  conds_std=conds_std).cpu()

                batch_images = (batch_images + 1) / 2
                # keep the generated images
                for j in range(len(batch_images)):
                    img_name = filename(i + j)
                    torchvision.utils.save_image(
                        batch_images[j],
                        os.path.join(conf.generate_dir, f'{img_name}.png'))
        elif conf.model_type == ModelType.autoencoder:
            if conf.train_mode.is_latent_diffusion():
                # evaluate autoencoder + latent diffusion (doesn't give the images)
                model: BeatGANsAutoencModel
                eval_num_images = chunk_size(conf.eval_num_images, rank,
                                             world_size)
                desc = 'generating images'
                for i in trange(0, eval_num_images, batch_size, desc=desc):
                    batch_size = min(batch_size, eval_num_images - i)
                    x_T = torch.randn(
                        (batch_size, 3, conf.img_size, conf.img_size),
                        device=device)
                    batch_images = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                        conds_mean=conds_mean,
                        conds_std=conds_std,
                        clip_latent_noise=clip_latent_noise,
                    ).cpu()
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
            else:
                # evaulate autoencoder (given the images)
                # to make the FID fair, autoencoder must not see the validation dataset
                # also shuffle to make it closer to unconditional generation
                train_loader = make_subset_loader(conf,
                                                  dataset=train_data,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  parallel=True)

                i = 0
                for batch in tqdm(train_loader, desc='generating images'):
                    imgs = batch['img'].to(device)
                    x_T = torch.randn(
                        (len(imgs), 3, conf.img_size, conf.img_size),
                        device=device)
                    batch_images = render_condition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        x_start=imgs,
                        cond=None,
                        sampler=sampler,
                        latent_sampler=latent_sampler).cpu()
                    # model: BeatGANsAutoencModel
                    # # returns {'cond', 'cond2'}
                    # conds = model.encode(imgs)
                    # batch_images = sampler.sample(model=model,
                    #                               noise=x_T,
                    #                               model_kwargs=conds).cpu()
                    # denormalize the images
                    batch_images = (batch_images + 1) / 2
                    # keep the generated images
                    for j in range(len(batch_images)):
                        img_name = filename(i + j)
                        torchvision.utils.save_image(
                            batch_images[j],
                            os.path.join(conf.generate_dir, f'{img_name}.png'))
                    i += len(imgs)
        else:
            raise NotImplementedError()
    model.train()

    barrier()

    if get_rank() == 0:
        fid = fid_score.calculate_fid_given_paths(
            [cache_dir, conf.generate_dir],
            batch_size,
            device=device,
            dims=2048)

        # remove the cache
        if remove_cache and os.path.exists(conf.generate_dir):
            shutil.rmtree(conf.generate_dir)

    barrier()

    if get_rank() == 0:
        # need to float it! unless the broadcasted value is wrong
        fid = torch.tensor(float(fid), device=device)
        broadcast(fid, 0)
    else:
        fid = torch.tensor(0., device=device)
        broadcast(fid, 0)
    fid = fid.item()
    print(f'fid ({get_rank()}):', fid)

    return fid
