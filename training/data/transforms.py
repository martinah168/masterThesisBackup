import timeit
from typing import Callable, Iterator

import monai.transforms as mon_transforms
import numpy as np
import torch
import torchvision.transforms.functional as Ftrans
from scipy.spatial import Delaunay
from torchvision import transforms
from torchvision.utils import save_image

from training.data.mri import calc_center_of_mass, extract_slices_from_volume


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)


def make_transform(
    image_size,
    flip_prob=0.5,
    crop_d2c=False,
):
    if crop_d2c:
        transform = [
            d2c_crop(),
            transforms.Resize(image_size),
        ]
    else:
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
    transform.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    return transform


class Crop:

    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + '(x1={}, x2={}, y1={}, y2={})'.format(
            self.x1, self.x2, self.y1, self.y2)


def get_simclr_transform(size, s=1):
    """Original simclr transform for reference."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor()
    ])
    raise NotImplementedError('You should use the MRI version below.')
    return data_transforms


def RandomResizedCropd(size, *args, **kwargs):
    return mon_transforms.Compose([
        mon_transforms.RandScaleCropd(
            roi_scale=0.08,
            max_roi_scale=1.,
            # TODO: add random center while ensuring having tumor in the crop
            random_center=False,
            *args,
            **kwargs),
        mon_transforms.Resized(spatial_size=[size, size, size], *args, **kwargs)
    ])


class CenterOfMassTumorCropd:

    def __init__(
            self, *args, **kwargs
    ) -> Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]:

        self.args = args
        self.kwargs = kwargs

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        roi_center = calc_center_of_mass(batch['seg'] > 0)[0]
        batch_transformed = mon_transforms.SpatialCropd(roi_center=roi_center,
                                                        *self.args,
                                                        **self.kwargs)(batch)
        return batch_transformed


def TwoStageCenterOfMassTumorCropd(roi_size_com, roi_size_rand, *args,
                                   **kwargs):
    stage_one_com_crop = CenterOfMassTumorCropd(roi_size=roi_size_com,
                                                *args,
                                                **kwargs)

    def stage_one(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # crop larger area around center of mass
        batch = stage_one_com_crop(batch)
        return batch

    # random crop to roi_size_rand
    stage_two_rand_crop = mon_transforms.RandSpatialCropd(
        roi_size=roi_size_rand,
        random_center=True,
        random_size=False,
        *args,
        **kwargs)

    def stage_two(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # crop randomly withing the already cropped region
        batch_transformed = stage_two_rand_crop(batch)
        return batch_transformed

    def f(batch):
        batch = stage_one(batch)
        batch = stage_two(batch)
        return batch

    return f


def crop_convex_hull(batch, roi_center_tumor, roi_center_healthy,
                     healthy_dir_vector_norm, *args, **kwargs):

    # find hull to tumor center vector in  uncropped image
    points = np.stack(np.array(batch['seg'][0] > 0).nonzero(), axis=1)

    delauney = Delaunay(points)
    # walk from tumor center to healthy center until we hit the hull
    # of the tumor
    for i in range(100):
        hull_check_point = roi_center_tumor + i * healthy_dir_vector_norm
        hull_check_point = hull_check_point.floor().long()

        # simplex = -1 if not batch['seg'][0][tuple(
        #     hull_check_point)] > 0 else 0
        simplex = delauney.find_simplex(hull_check_point)
        if simplex == -1:
            break

        # vector from tumor center to hull
    step_vector = i * healthy_dir_vector_norm

    # move roi_center away from tumor center the amount from tumor centere to hull
    # times two because we assume tumor is symmetric
    roi_center_healthy = roi_center_healthy + step_vector * 2
    roi_center_healthy = roi_center_healthy.floor().long()

    batch_transformed = mon_transforms.SpatialCropd(
        roi_center=roi_center_healthy, *args, **kwargs)(batch)
    brainmask_healthy = get_healthy_brain_mask(batch_transformed)

    roi_center_healthy_crop = calc_center_of_mass(
        brainmask_healthy)[0].floor().long()

    return batch_transformed, roi_center_healthy_crop


def crop_iter(batch, roi_center_healthy, healthy_dir_vector_norm, *args,
              **kwargs):
    # initialize updating variables
    batch_transformed = batch.copy()
    roi_center_healthy_crop = roi_center_healthy.clone()

    cnt = 0
    while has_tumor(batch_transformed['seg']):
        # move roi_center away from tumor center
        batch_transformed = mon_transforms.SpatialCropd(
            roi_center=roi_center_healthy, *args, **kwargs)(batch)
        brainmask_healthy = get_healthy_brain_mask(batch_transformed)

        roi_center_healthy_crop = calc_center_of_mass(
            brainmask_healthy)[0].floor().long()

        roi_center_healthy = roi_center_healthy + healthy_dir_vector_norm
        # add some noise to the direction vector

        # visualize
        save_image(extract_slices_from_volume(batch_transformed['t2'],
                                              roi_center_healthy_crop),
                   'test_img/healthy_crop.png',
                   normalize=True,
                   nrow=3)

        vis_seg_mask = batch_transformed['seg'].clone()
        vis_seg_mask[0][tuple(roi_center_healthy_crop)] = 10
        save_image(extract_slices_from_volume(vis_seg_mask,
                                              roi_center_healthy_crop),
                   'test_img/healthy_crop_seg.png',
                   normalize=True,
                   nrow=3)
        cnt += 1
    print(f'Healthy crop took {cnt} iterations.')

    return batch_transformed, roi_center_healthy_crop,


def has_tumor(seg_map: torch.Tensor) -> bool:
    return (seg_map > 0).any()


def get_healthy_brain_mask(batch):
    # get non-tumor regions
    non_tumor_mask = batch['seg'] == 0

    # get brainmask from original brainmask
    fg_brain_mask = batch['brainmask'].bool()

    healthy_brain_mask = non_tumor_mask & fg_brain_mask
    return healthy_brain_mask


def get_bounding_box(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 4:
        assert mask.shape[0] == 1
        mask = mask[0]
    x_min, y_min, z_min = mask.nonzero().min(dim=0).values
    x_max, y_max, z_max = mask.nonzero().max(dim=0).values

    return torch.tensor([x_min, y_min, z_min, x_max, y_max, z_max])


def bbox_to_coords(x_min: float, y_min: float, z_min: float, x_max: float,
                   y_max: float, z_max: float) -> torch.Tensor:
    return torch.tensor([[x, y, z] for x in [x_min, x_max]
                         for y in [y_min, y_max] for z in [z_min, z_max]])


def crop_bbox(batch, roi_size, *args, **kwargs) -> dict[str, torch.Tensor]:

    img_key = kwargs['keys'][0]
    # create bounding box around tumor
    tumor_mask = batch['seg'] > 0
    bbox_tumor = get_bounding_box(tumor_mask)

    # images are in -1 to 1 range
    brainmask = get_healthy_brain_mask(batch)
    if brainmask.ndim == 4:
        brainmask = brainmask.all(dim=0, keepdim=True)
    bbox_brain = get_bounding_box(brainmask)

    # add margin of half roi size to bounding box to prevent cropping parts of the tumor
    bbox_tumor_margin = bbox_tumor.clone()

    # margin needs to be half of the roi size to prevent cropping parts of the tumor
    margin = roi_size // 2 + 1

    # add margins around the bounding box
    # sub margin from min coords
    bbox_tumor_margin[:3] -= margin
    # add margin to max coords
    bbox_tumor_margin[3:] += margin

    # clamp brain margin to ensure there is space left between the brain bbox and the tumor bbox
    margin = bbox_brain - bbox_tumor_margin

    # flip sign of max coords to get the max margin
    margin *= torch.tensor([1, 1, 1, -1, -1, -1])

    # sub one from margin to ensure there is at least one voxel between the brain and tumor bbox
    margin -= 1

    # clamp to ensure margin is not negative, it also does not need to be larger than half the roi size
    margin.clamp_(0, roi_size // 2)

    bbox_brain_margin = bbox_brain.clone()

    # add margin to min coords inside the bounding box
    bbox_brain_margin[:3] += margin[:3]
    # sub margin from max coords inside the bounding box
    bbox_brain_margin[3:] -= margin[3:]

    # handle case where tumor bounding box is too such that
    # the margin is larger than the brain bounding box
    if (margin == 0).all():
        # shrink bbox tumor bbox
        # calculate necessary offset to shrink bbox
        o = (bbox_brain - bbox_tumor_margin)
        o.clamp_min_(0)
        bbox_tumor_margin[:3] += o[:3] + 1
        bbox_tumor_margin[3:] -= o[3:] + 1
        print(
            f'Shrinking tumor bbox to fit inside brain bbox. {batch["patient_id"]}'
        )

    # add margin at the edges of the brain

    # create all possible coords inside the brain bounding box with the margin
    xx, yy, zz = torch.meshgrid([
        torch.arange(bbox_brain_margin[0], bbox_brain_margin[3] + 1),
        torch.arange(bbox_brain_margin[1], bbox_brain_margin[4] + 1),
        torch.arange(bbox_brain_margin[2], bbox_brain_margin[5] + 1),
    ],
                                indexing='ij')
    all_brain_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    # filter coords based to be outside of tumor bbox
    possible_coords = all_brain_coords[
        # x plane
        (all_brain_coords[:, 0] <= bbox_tumor_margin[0]) |
        (all_brain_coords[:, 0] >= bbox_tumor_margin[3]) |
        # y plane
        (all_brain_coords[:, 1] <= bbox_tumor_margin[1]) |
        (all_brain_coords[:, 1] >= bbox_tumor_margin[4]) |
        # z plane
        (all_brain_coords[:, 2] <= bbox_tumor_margin[2]) |
        (all_brain_coords[:, 2] >= bbox_tumor_margin[5])]

    # only take inner most 90% of the coords
    braincenter = calc_center_of_mass(brainmask)
    dists = torch.norm(possible_coords - braincenter, dim=1)
    # closest coords
    top_q = torch.quantile(dists, 0.1)
    possible_coords = possible_coords[dists <= top_q]

    # filter coords that are background (i.e. -1)
    bg_mask = (batch[img_key].isclose(-torch.ones(1)))[0][tuple(
        possible_coords.T)]
    possible_coords = possible_coords[~bg_mask]

    if possible_coords.size(0) == 0:
        print(f'No possible coords found for {batch["patient_id"]}')

    # sample one coordinate as a possible roi center
    rand_roi_center_idx = torch.randint(0, possible_coords.shape[0], (1,))
    roi_center_healthy = possible_coords[rand_roi_center_idx]

    # spatial crop the batch around the roi center
    batch_transformed = mon_transforms.SpatialCropd(
        roi_center=roi_center_healthy[0], roi_size=roi_size, *args,
        **kwargs)(batch)

    brainmask_healthy = get_healthy_brain_mask(batch_transformed)

    roi_center_healthy_crop = calc_center_of_mass(
        brainmask_healthy)[0].floor().long()

    # pad to correct size if crop exceeds image boundaries
    spatial_size = batch_transformed[img_key].shape[1:]
    if any(s != max(spatial_size) for s in spatial_size):
        batch_transformed = mon_transforms.Padd(
            padder=mon_transforms.SpatialPad(spatial_size=roi_size,
                                             mode='constant',
                                             value=-1.,
                                             method='end'),
            *args,
            **kwargs,
        )(batch_transformed)

    return batch_transformed, roi_center_healthy_crop


def RandHealthyOnlyCropd(
        vis=False,
        approach='bbox',
        *args,
        **kwargs
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:

    keys = kwargs['keys']

    def f(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        start = timeit.default_timer()

        if approach in ['iterative', 'hull']:
            roi_center_tumor = calc_center_of_mass(batch['seg'] > 0)[0]

            # get initial estimate of the center of mass of the healthy tissue
            brainmask_healthy = get_healthy_brain_mask(batch)
            roi_center_healthy = calc_center_of_mass(brainmask_healthy)[0]

            # calculate vector from tumor center to healthy center
            healthy_dir_vector = (roi_center_healthy - roi_center_tumor)
            healthy_dir_vector = healthy_dir_vector / healthy_dir_vector.norm()
            if approach == 'iterative':

                batch_transformed, roi_center_healthy_crop = crop_iter(
                    batch, roi_center_healthy, healthy_dir_vector, *args,
                    **kwargs)

            elif approach == 'hull':
                batch_transformed, roi_center_healthy_crop = crop_convex_hull(
                    batch, roi_center_tumor, roi_center_healthy,
                    healthy_dir_vector, *args, **kwargs)

        elif approach == 'bbox':
            batch_transformed, roi_center_healthy_crop = crop_bbox(
                batch, *args, **kwargs)
        else:
            raise ValueError(f'Unknown approach {approach}')
        stop = timeit.default_timer()
        if vis:
            print(f'Time for crop: {stop - start:.2f}')
            percentage_brain_crop = percentage_brain(batch_transformed[keys[0]])
            print(f'Percentage brain crop: {percentage_brain_crop:.2f}')
            batch_transformed['com'] = roi_center_healthy_crop
            save_image(extract_slices_from_volume(batch_transformed[keys[0]],
                                                  roi_center_healthy_crop),
                       'test_img/healthy_crop.png',
                       normalize=True,
                       nrow=3)
            save_image(extract_slices_from_volume(batch_transformed['seg'],
                                                  roi_center_healthy_crop),
                       'test_img/healthy_crop_seg.png',
                       normalize=True,
                       nrow=3)
        return batch_transformed

    return f


def percentage_brain(brain: torch.Tensor) -> float:
    return ((brain != 0).sum() / brain.numel()).item()


def get_simclr_transform_mri(size,
                             keys=['img', 'seg', 'brainmask'],
                             with_crop: bool = False,
                             track_meta: bool = False):
    """SimCLR transform subset that can actually be applied to MRI data"""
    crop_transform = (RandomResizedCropd(
        size=size, keys=keys, allow_missing_keys=False,
        track_meta=track_meta),) if with_crop else ()

    data_transforms = mon_transforms.Compose([
        *crop_transform,
        mon_transforms.RandGaussianSmoothd(keys=[
            'img',
        ],
                                           sigma_x=(0.1, 2.0),
                                           sigma_y=(0.1, 2.0),
                                           sigma_z=(0.1, 2.0),
                                           prob=0.5),
        mon_transforms.RandFlipd(keys=keys, spatial_axis=2, prob=0.5)
    ])
    # set deterministic
    data_transforms.set_random_state(seed=0)
    return data_transforms


def get_mri_aug(
    size: int,
    p: float,
    keys: Iterator = ('img', 'seg', 'brainmask'),
    track_meta: bool = False,
) -> Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:

    simclr_transform = get_simclr_transform_mri(size,
                                                keys=keys,
                                                with_crop=False,
                                                track_meta=track_meta)

    # intensity based augmentations
    intensity_transforms = mon_transforms.Compose([
        mon_transforms.RandAdjustContrastd(
            keys='img',
            prob=p,
            gamma=(.5, 1.5)  # default: (0.5,4.5)
        ),
        mon_transforms.RandBiasFieldd(
            keys='img',
            degree=3,
            coeff_range=(0.0, 0.1),
            prob=p,
        ),
        mon_transforms.RandGaussianNoised(
            keys='img',
            prob=p,
            mean=0.0,
            std=0.1,
        ),
    ])

    # spatial based augmentations
    spatial_transforms = mon_transforms.Compose([
        mon_transforms.Rand3DElasticd(keys=keys,
                                      sigma_range=(0.1, 2.0),
                                      magnitude_range=(0.1, 2.0),
                                      prob=p,
                                      mode='nearest'),
        mon_transforms.RandRotated(keys=keys,
                                   range_x=(0, 2 * np.pi),
                                   range_y=(0, 2 * np.pi),
                                   range_z=(0, 2 * np.pi),
                                   prob=p,
                                   mode='nearest'),
        mon_transforms.RandZoomd(keys=keys,
                                 prob=p,
                                 min_zoom=0.9,
                                 max_zoom=1.1,
                                 keep_size=True,
                                 mode='nearest')
    ])
    aug_transform = mon_transforms.Compose([
        intensity_transforms,
        spatial_transforms,
        simclr_transform,
    ] + ([mon_transforms.ToTensord(keys=keys, track_meta=False
                                  )] if not track_meta else []))

    # set deterministic
    aug_transform.set_random_state(seed=0)

    return aug_transform


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
