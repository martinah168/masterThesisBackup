import lpips
import torch
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

from training.config.train import TrainConfig
from training.data.mri import extract_slices_from_volume
from training.dist import all_gather, barrier, get_world_size
from training.metrics.loader_utils import make_subset_loader
from training.metrics.psnr import psnr
from training.metrics.ssim import ssim
from training.mode.model import ModelType
from training.models import Model
from training.models.diffusion import Sampler
from training.vis.renderer import render_condition, render_uncondition


def evaluate_lpips(
    sampler: Sampler,
    model: Model,
    conf: TrainConfig,
    device,
    val_data: Dataset,
    latent_sampler: Sampler = None,
    use_inverted_noise: bool = False,
):
    """
    compare the generated images from autoencoder on validation dataset
    Args:
        use_inversed_noise: the noise is also inverted from DDIM
    """
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    val_loader = make_subset_loader(conf,
                                    dataset=val_data,
                                    batch_size=conf.batch_size_eval,
                                    shuffle=False,
                                    parallel=True)

    model.eval()
    with torch.no_grad():
        scores = {
            'lpips': [],
            'mse': [],
            'ssim': [],
            'psnr': [],
        }
        for batch in tqdm(val_loader, desc='lpips'):
            imgs = batch['img'].to(device)

            if use_inverted_noise:
                # inverse the noise
                # with condition from the encoder
                model_kwargs = {}
                if conf.model_type.has_autoenc():
                    with torch.no_grad():
                        model_kwargs = model.encode(imgs)
                x_T = sampler.ddim_reverse_sample_loop(
                    model=model,
                    x=imgs,
                    clip_denoised=True,
                    model_kwargs=model_kwargs)
                x_T = x_T['sample']
            else:
                x_T = torch.randn_like(imgs, device=device)

            if conf.model_type == ModelType.ddpm:
                # the case where you want to calculate the inversion capability of the DDIM model
                assert use_inverted_noise
                pred_imgs = render_uncondition(
                    conf=conf,
                    model=model,
                    x_T=x_T,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                )
            else:
                pred_imgs = render_condition(conf=conf,
                                             model=model,
                                             x_T=x_T,
                                             x_start=imgs,
                                             cond=None,
                                             sampler=sampler)

            # extract slices when images are 3d
            if imgs.dim() == 5:
                com = batch['com'].to(imgs.device)
                imgs = extract_slices_from_volume(imgs, com)
                pred_imgs = extract_slices_from_volume(pred_imgs, com)

            # (n, 1, 1, 1) => (n, )
            scores['lpips'].append(lpips_fn.forward(imgs, pred_imgs).view(-1))

            # need to normalize into [0, 1]
            norm_imgs = (imgs + 1) / 2
            norm_pred_imgs = (pred_imgs + 1) / 2
            # (n, )
            scores['ssim'].append(
                ssim(norm_imgs, norm_pred_imgs, size_average=False))
            # (n, )
            scores['mse'].append(
                (norm_imgs - norm_pred_imgs).pow(2).mean(dim=[1, 2, 3]))
            # (n, )
            scores['psnr'].append(psnr(norm_imgs, norm_pred_imgs))
        # (N, )
        for key in scores.keys():
            scores[key] = torch.cat(scores[key]).float()
    model.train()

    barrier()

    # support multi-gpu
    outs = {
        key: [
            torch.zeros(len(scores[key]), device=device)
            for i in range(get_world_size())
        ] for key in scores.keys()
    }
    for key in scores.keys():
        all_gather(outs[key], scores[key])

    # final scores
    for key in scores.keys():
        scores[key] = torch.cat(outs[key]).mean().item()

    # {'lpips', 'mse', 'ssim'}
    return scores
