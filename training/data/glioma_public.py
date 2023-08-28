import logging
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import monai.transforms as mon_transforms
import numpy as np
import torch
from monai.transforms import LoadImaged, Padd, ScaleIntensity, SpatialPad, ToTensord
from torch.utils.data import Dataset
from torchvision.utils import save_image

from training.data.csv import read_csv_labels
from training.data.mri import (MriCrop, calc_center_of_mass,
                               extract_slices_from_volume)
from training.data.transforms import (CenterOfMassTumorCropd,
                                      RandHealthyOnlyCropd,
                                      TwoStageCenterOfMassTumorCropd,
                                      get_mri_aug)
from training.mode.train import TrainMode

nib_logger = logging.getLogger('nibabel')


class PublicGliomaDataset(Dataset):
    num_classes: int = 3
    SEG_LABEL_NAME = 'seg'
    BRAINMASK_NAME = 'brainmask'
    cls_to_name = {
        0: 'Astrocytoma',
        1: 'Glioblastoma',
        2: 'Oligodendroglioma',
    }
    name_to_cls_tum = {
        'Astro': 0,  # astrocytoma
        'GBM': 1,  # glioblastoma
        'Oligo': 2  # oligodendroglioma
    }

    subset_names_dict: dict[str, dict[str, tuple[str, ...]]] = {
        'study': {
            'train': (
                'brats_2021_train',
                'brats_2021_valid',
                'erasmus',
                'lumiere',
                'rembrandt',
                'ucsf_glioma',
                'upenn_gbm',
            ),
            'val': (
                'brats_2021_train',
                'brats_2021_valid',
                'erasmus',
                'lumiere',
                'rembrandt',
                'ucsf_glioma',
                'upenn_gbm',
            ),
            'test': ('tcga', 'tum_glioma')
        },
        'mixed': {
            'train':
                ('brats_2021_train', 'brats_2021_valid', 'erasmus', 'lumiere',
                 'rembrandt', 'ucsf_glioma', 'upenn_gbm', 'tcga', 'tum_glioma'),
            'val':
                ('brats_2021_train', 'brats_2021_valid', 'erasmus', 'lumiere',
                 'rembrandt', 'ucsf_glioma', 'upenn_gbm', 'tcga', 'tum_glioma'),
            'test':
                ('brats_2021_train', 'brats_2021_valid', 'erasmus', 'lumiere',
                 'rembrandt', 'ucsf_glioma', 'upenn_gbm', 'tcga', 'tum_glioma'),
        }
    }

    def __init__(
        self,
        data_dir: Path,
        img_size: int,
        mri_crop: MriCrop,
        train_mode: TrainMode,
        mri_sequences: tuple[str, ...],
        split: str,
        split_ratio: float,
        use_healthy: bool,
        with_data_aug: bool,
        split_mode: str,
        data_aug_prob: float,
        manipulate_cls: tuple[int, int] = tuple([]),
        file_types: tuple[str, str] = ('nii', 'nii.gz'),
        filter_class_labels: bool = False,
        view_transform: Optional[Callable] = None,
        aug_encoder: bool = False,
    ):
        print('\n')

        # define constants
        self._no_preop_fp = Path(
            __file__).resolve().parents[2] / 'no_preop_subjects.txt'

        self.manipulate_cls = tuple(int(x) for x in manipulate_cls)
        if len(self.manipulate_cls) == 2:
            self.num_classes = 2
        self.data_dir = Path(data_dir)

        # train: everything else (upenn, brats, rembrandt, erasmus, ucsf)
        # test: tcga, tum_glioma
        assert split in ('train', 'val',
                         'test'), 'split must be train, val or test'

        self.split_mode = split_mode
        self.subset_names = self.subset_names_dict[self.split_mode][split]

        self.split = split
        self.file_types = file_types
        self._mri_sequences = mri_sequences
        self.mri_modes = [
            *mri_sequences, self.SEG_LABEL_NAME, self.BRAINMASK_NAME
        ]

        self.subject_dirs = self._glob_subject_dirs(self.data_dir)
        # load labels
        self.cls_labels = self._load_cls_labels(self.data_dir)

        is_study = self.split_mode == 'study'
        is_not_test = split != 'test'  # split is train or val

        is_mixed = self.split_mode == 'mixed'
        if (is_study and is_not_test) or is_mixed:
            all_indices = np.arange(len(self.subject_dirs))
            # shuffle indices

            g = np.random.default_rng(0)
            g.shuffle(all_indices)

        # perform train val split
        if is_study:
            if is_not_test:
                # generate indices for train/val split
                # split indices
                split_idx = int(len(all_indices) * split_ratio)
                if split == 'train':
                    idc = all_indices[:split_idx]
                else:
                    idc = all_indices[split_idx:]
                # debug
                # print(sorted(idc))

        elif is_mixed:
            split_ratios = {
                'train': (0., 0.8),
                'val': (0.8, 0.9),
                'test': (0.9, 1.0)
            }

            split_idx_start = int(len(all_indices) * split_ratios[split][0])
            split_idx_end = int(len(all_indices) * split_ratios[split][1])
            idc = all_indices[split_idx_start:split_idx_end]

            # debug
            # print(sorted(idc))

        else:
            raise ValueError(f'Unknown split mode {self.split_mode}')

        if (is_study and is_not_test) or is_mixed:
            # select subjects for this split
            self.subject_dirs = [self.subject_dirs[i] for i in idc]

        self.cls_labels = self._fill_cls_labels()

        # if classification, remove subjects without label
        if train_mode.is_manipulate() or filter_class_labels:
            print(f'len before: {len(self.subject_dirs)}')
            self._filter_cls_labels()
            print(f'len after: {len(self.subject_dirs)}')
            print(
                'set of remaining class labels', {
                    self.cls_labels[self._make_patient_id(subj)]
                    for subj in self.subject_dirs
                })
            # print histogram of class labels
            self._init_cls_hist()
            print('samples per class', self.samples_per_cls)
            assert len(self.subject_dirs) != 0, 'no subjects left'
        # calc study distribution
        studies_in_set = [s.parts[-2] for s in self.subject_dirs]
        studies_in_set_counter = Counter(studies_in_set)
        # get relative distribution of studies in set
        studies_in_set_dist = {
            k: v / len(studies_in_set)
            for k, v in studies_in_set_counter.items()
        }

        print(
            f'studies in {split} set', {
                k: f'{studies_in_set_dist[k]:.2f}'
                for k in sorted(studies_in_set_dist.keys())
            })

        self.orig_img_size = {96: 240, 24: 64}[img_size]
        self.img_size = img_size
        self.mri_crop = mri_crop

        self.load_keys = [*self._mri_sequences, self.SEG_LABEL_NAME]

        self.load_transform = mon_transforms.Compose([
            LoadImaged(keys=self.load_keys,
                       image_only=True,
                       ensure_channel_first=True,
                       simple_keys=True,
                       dtype=np.float32),
            Padd(keys=self.load_keys,
                 padder=SpatialPad(spatial_size=self.orig_img_size,
                                   mode='constant',
                                   value=0.,
                                   method='symmetric'))
        ])
        # normalizes to [-1,1]
        self.normalize_transform = ScaleIntensity(minv=-1.,
                                                  maxv=1.,
                                                  channel_wise=False)

        self._aug_encoder = aug_encoder
        self._aug_encoder &= split == 'train'

        transform_keys = ('img', 'seg', 'brainmask')

        # define cropping operations
        # for cropping healthy regions
        self.healthy_crop_transform = RandHealthyOnlyCropd(
            keys=transform_keys,
            roi_size=self.img_size) if use_healthy else None

        if split == 'train':
            self.tumor_crop_transform = TwoStageCenterOfMassTumorCropd(
                keys=transform_keys,
                roi_size_com=int(1.25 * self.img_size),
                roi_size_rand=self.img_size)

        elif split in ['val', 'test']:
            self.tumor_crop_transform = CenterOfMassTumorCropd(
                keys=transform_keys, roi_size=self.img_size)

        # define augmentation operations
        with_data_aug |= self._aug_encoder
        with_data_aug &= split == 'train'

        self.aug_transform = get_mri_aug(
            self.orig_img_size,
            keys=transform_keys,
            p=data_aug_prob,
            track_meta=False) if with_data_aug else None
        self.view_transform = view_transform

        self.strip_meta_data = ToTensord(keys=transform_keys, track_meta=False)

    def _fill_cls_labels(self):
        return {
            self._make_patient_id(s):
            self.cls_labels.get(self._make_patient_id(s), -2)
            for s in self.subject_dirs
        }

    def _init_cls_hist(self):
        self.samples_per_cls = [
            sum([l == i
                 for l in self.cls_labels.values()])
            for i in range(self.num_classes)
        ]
        self.cls_hist = [
            c / sum(self.samples_per_cls) for c in self.samples_per_cls
        ]

    def _filter_cls_labels(self):

        def remove_label(patient_id: str) -> bool:
            cur_label = self.cls_labels.get(patient_id, -2)

            # if cur_label==-2:
            #     print(f"no label for {patient_id}")

            has_label = cur_label >= 0
            if has_label and len(self.manipulate_cls) == 2:
                is_used_label = cur_label in self.manipulate_cls
                return has_label and is_used_label
            return has_label

        # remove subjects without label
        print(f'removing subjects without label')
        self.subject_dirs = [
            s for s in self.subject_dirs
            if remove_label(self._make_patient_id(s))
        ]
        self.cls_labels = {
            c: v for c, v in self.cls_labels.items() if remove_label(c)
        }
        if len(self.manipulate_cls) == 2:
            self.cls_label_map = dict(zip(self.manipulate_cls, (0, 1)))
            self.inv_cls_label_map = dict(zip((0, 1), self.manipulate_cls))
            print(f'cls label mapping: {self.cls_label_map}')
            # map labels to (0,1)
            self.cls_labels = {
                c: self.manipulate_cls.index(v)
                for c, v in self.cls_labels.items()
            }
            print(f'sanity check: new labels: {set(self.cls_labels.values())}')

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        subject_dir = self.subject_dirs[index]
        # patient id is last two entries of subject_dir
        patient_id = self._make_patient_id(subject_dir)
        # patient_id = 'brats_2021_train/BraTS2021_00606'

        subject_dir /= 'preop'
        if not subject_dir.exists():
            # replace patient with random other patient
            random_subject = subject_dir
            while random_subject == subject_dir:
                random_subject = random.choice(self.subject_dirs)
                print(
                    f'no preop dir for {patient_id}, replacing with {random_subject}'
                )
                # store subject without preop dir to a file to avoid using it again
                with open(self._no_preop_fp, 'a') as f:
                    f.write(f'{patient_id}\n')

                self.subject_dirs[index] = random_subject
            return self.__getitem__(index)

        subject_fns = {
            k: self.get_fn_for_seq(subject_dir, k) for k in self.load_keys
        }

        nib_logger.setLevel(logging.ERROR)  # prevent nibabel warnings
        subject_data = self.load_transform(subject_fns)
        nib_logger.setLevel(logging.INFO)

        # normalize image
        for seq, img in (
            (seq, subject_data[seq]) for seq in self._mri_sequences):
            zero_mask = img == 0
            img[~zero_mask] = self.normalize_transform(img[~zero_mask])
            img[zero_mask] = -1
            subject_data[seq] = img

        img = torch.cat(
            [subject_data.pop(mode) for mode in self._mri_sequences], dim=0)
        subject_data['img'] = img
        subject_data['index'] = torch.tensor(index)
        subject_data['patient_id'] = patient_id

        # create brainmask one from segmentation
        subject_data[self.BRAINMASK_NAME] = subject_data['img'] != -1

        # augment
        subject_data_aug = self.aug_transform(
            subject_data) if self.aug_transform is not None else subject_data

        # visualize augmented imaged
        is_zeroth_worker = (torch.utils.data.get_worker_info() is None or
                            torch.utils.data.get_worker_info().id == 0)
        if is_zeroth_worker and self.split == 'train':
            self._vis_transformed(subject_data, subject_data_aug)

        if not self._aug_encoder:
            # replace original data with augmented data ( in classifcation mode)
            subject_data = subject_data_aug

        # ensure data is still in [-1,1] range
        self._clamp_intensities(subject_data)

        if self._aug_encoder:
            self._clamp_intensities(subject_data_aug)

        # coin flip to decide if we crop healthy or tumor
        # AND variable to store if we cropped healthy or tumor
        is_healthy = self.healthy_crop_transform is not None and random.random(
        ) > 0.5
        # is_healthy = True
        on_aug = False
        try:
            subject_data = self._crop_healthy_or_tumor(subject_data, is_healthy)
            if self._aug_encoder:
                subject_data_aug = self._crop_healthy_or_tumor(
                    subject_data_aug, is_healthy)
                on_aug = True
        except RuntimeError as e:
            print(f'error cropping {patient_id} ({on_aug = }): {e}')

            raise e

        subject_data['is_healthy'] = torch.tensor(is_healthy)

        # recalculate center of mass after transform
        self._update_com(subject_data, is_healthy)
        if self._aug_encoder:

            self._update_com(subject_data_aug, is_healthy)

        # update subject data with augmented data with postfix if we keep original data
        if self._aug_encoder:
            # only keep augmented img for now
            subject_data.update({
                k + '_aug': v for k, v in subject_data_aug.items() if k == 'img'
            })

        if self.healthy_crop_transform is None:
            # use original class label when only classifying tumor types
            subject_data['cls_labels'] = torch.tensor(
                self.cls_labels.get(patient_id, -1))
        else:
            # use new class label when classifying healthy vs tumor
            subject_data['cls_labels'] = torch.tensor(0 if is_healthy else 1)
            subject_data['og_cls_labels'] = torch.tensor(
                self.cls_labels.get(patient_id, -1))

        subject_data['labels'] = subject_data['cls_labels']

        if self.view_transform:
            # create views of the img
            subject_data_transformed = self.view_transform(subject_data)
            for i_t, sdt in enumerate(subject_data_transformed):
                # restore segmentation label
                subject_data_transformed[i_t][self.SEG_LABEL_NAME] = sdt[
                    self.SEG_LABEL_NAME].round().int()
                # recalculate center of mass after transform
                subject_data_transformed[i_t]['com'] = calc_center_of_mass(
                    sdt[self.SEG_LABEL_NAME] > 0)[0]

            # clamp image to -1 and 1 after transformation
            for i_t, sdt in enumerate(subject_data_transformed):
                subject_data_transformed[i_t]['img'] = sdt['img'].clamp(-1, 1)

            subject_data = subject_data_transformed
            subject_data = [self.strip_meta_data(sd) for sd in subject_data]

        else:
            # clamp image to -1 and 1
            subject_data['img'] = subject_data['img'].clamp(-1, 1)
            subject_data = self.strip_meta_data(subject_data)
            important_keys = [
                'img', 'seg', 'cls_labels', 'labels', 'index', 'patient_id',
                'com', 'is_healthy'
            ]  # "patient_id",
            subject_data = {
                k: v for k, v in subject_data.items() if k in important_keys
            }

        if is_zeroth_worker and self.split == 'train' and index == 0:
            # visualized cropped image
            if isinstance(subject_data, dict):
                self._vis_batch(subject_data)
            elif isinstance(subject_data, list):
                self._vis_views(subject_data)

        # debug vis
        if self._aug_encoder and is_zeroth_worker:
            slices = extract_slices_from_volume(subject_data['img'][None])
            save_image(slices, 'test_img/img.png', normalize=True)
            slices = extract_slices_from_volume(subject_data['img_aug'][None])
            save_image(slices, 'test_img/img_aug.png', normalize=True)

        return subject_data

    def _clamp_intensities(self, subject_data):
        subject_data['img'] = subject_data['img'].clamp(-1, 1)

    def _crop_healthy_or_tumor(
            self, subject_data: dict[str, torch.Tensor],
            do_healthy_crop: bool) -> dict[str, torch.Tensor]:
        if do_healthy_crop:
            subject_data = self.healthy_crop_transform(subject_data)
        else:
            subject_data = self.tumor_crop_transform(subject_data)
        return subject_data

    def _update_com(self, subject_data, is_healthy):
        if is_healthy:
            com_mask = (subject_data[self.BRAINMASK_NAME]).any(dim=0,
                                                               keepdim=True)
        else:
            com_mask = subject_data[self.SEG_LABEL_NAME] > 0

        subject_data['com'] = calc_center_of_mass(com_mask)[0]

    def _vis_views(self, subject_data: list[dict[str, torch.Tensor]]):
        slices = [
            extract_slices_from_volume(s['img'], s['com']) for s in subject_data
        ]
        slices_seg = [
            extract_slices_from_volume(s[self.SEG_LABEL_NAME], s['com'])
            for s in subject_data
        ]
        for i_view, slice_ in enumerate(slices):
            save_image(slice_,
                       f'test_img/batch_view_{i_view}.png',
                       normalize=True,
                       nrow=3,
                       value_range=(-1, 1))

        for i_view, slice_ in enumerate(slices_seg):
            save_image(slice_.float() / slice_.max() * 2 - 1,
                       f'test_img/batch_seg_view_{i_view}.png',
                       normalize=True,
                       nrow=3,
                       value_range=(-1, 1))

    def _vis_batch(self, subject_data: dict[str, torch.Tensor]):
        slices = extract_slices_from_volume(subject_data['img'],
                                            subject_data['com'])
        save_image(slices,
                   'test_img/batch.png',
                   normalize=True,
                   nrow=3,
                   value_range=(-1, 1))
        slices_seg = extract_slices_from_volume(subject_data['seg'],
                                                subject_data['com']) / 2 - 1

        save_image(slices_seg,
                   'test_img/batch_seg.png',
                   normalize=True,
                   nrow=3,
                   value_range=(-1, 1))

        slices_brainmask = extract_slices_from_volume(
            subject_data[self.BRAINMASK_NAME], subject_data['com']).float()
        slices_brainmask = slices_brainmask * 2 - 1
        save_image(slices_brainmask,
                   'test_img/batch_brainmask.png',
                   normalize=True,
                   nrow=3,
                   value_range=(-1, 1))
        slices_alpha = 0.5 * slices + 0.5 * slices_seg.repeat(4, 1, 1, 1)
        save_image(slices_alpha,
                   'test_img/og_img_seg.png',
                   normalize=True,
                   nrow=3,
                   value_range=(-1, 1))

    def get_fn_for_seq(self, subject_dir: Path, mode: str) -> Path:
        fn = '_'.join([
            f'sub-{subject_dir.parent.name}', f'ses-{subject_dir.name}',
            f'space-sri', mode
        ]) + '.nii.gz'
        return subject_dir / fn

    def _vis_transformed(self, subject_data, subject_data_transformed, idx=0):
        if 'com' not in subject_data:
            subject_data['com'] = calc_center_of_mass(
                subject_data[self.SEG_LABEL_NAME])[0]
        if 'com' not in subject_data_transformed:
            subject_data_transformed['com'] = calc_center_of_mass(
                subject_data_transformed[self.SEG_LABEL_NAME])[0]

        slices = extract_slices_from_volume(subject_data['img'],
                                            subject_data['com'])
        save_image(slices,
                   'test_img/og.png',
                   normalize=True,
                   nrow=3,
                   value_range=(-1, 1))

        slices_transformed = extract_slices_from_volume(
            subject_data_transformed['img'], subject_data_transformed['com'])
        save_image(slices_transformed,
                   f'test_img/transform_{idx}.png',
                   normalize=True,
                   nrow=3,
                   value_range=(-1, 1))
        save_image((slices_transformed - slices).abs(),
                   f'test_img/diff_{idx}.png',
                   normalize=True,
                   nrow=3,
                   value_range=(-1, 1))
        # visualize seg mask after augmentation
        seg_slices = extract_slices_from_volume(
            subject_data_transformed[self.SEG_LABEL_NAME],
            subject_data_transformed['com'])
        save_image(seg_slices,
                   f'test_img/transform_seg_{idx}.png',
                   normalize=True,
                   nrow=3)

    @property
    def n_seq(self) -> int:
        """Number of MRI sequences."""
        return len(self._mri_sequences)

    def _make_patient_id(self, subject_dir: Path) -> str:
        patient_id = '/'.join(subject_dir.parts[-2:])
        if 'tum_glioma' in patient_id:
            patient_id = patient_id.replace('tum_glioma', 'glioma_epic')
        return patient_id

    def _load_cls_labels(self, data_dir: Path) -> dict[str, int]:
        labels: dict[str, int] = {}

        public_labels = self._load_labels_public(data_dir)
        labels.update(public_labels)

        tum_labels = self._load_labels_tum(data_dir)
        labels.update(tum_labels)

        return labels

    def _load_labels_tum(self, data_dir: Path) -> dict[str, int]:
        labels = read_csv_labels(
            csv_fn=data_dir / 'tum_glioma_PhenoData.csv',
            label_str='Diag_WHO2021',
            patient_id_fn=lambda row: f"glioma_epic/{row['Pseudonym']}")
        # map string labels to int

        # new tumor types that are not in public dataset:
        # diffuse hemispheric glioma (DHG),
        # diffuse midline glioma (DMLG),
        # Ganglioglioma1 ( GNMT),
        # ??? (hgapa),
        # Glioma,
        # pilocytic astrocytoma (PA),
        # Pleomorphic Xanthoastrocytoma (PXA)
        # labels_hist = Counter(labels.values())
        labels = {k: self.name_to_cls_tum.get(v, -1) for k, v in labels.items()}
        return labels

    def _load_labels_public(self, data_dir: Path) -> dict[str, int]:
        labels = read_csv_labels(
            csv_fn=data_dir / 'phenoData.csv',
            label_str='WHO2021_Int',
            patient_id_fn=lambda row: f"{row['Dataset']}/{row['Patient']}")
        labels = {
            subject: int(label) if label else -1
            for subject, label in labels.items()
        }
        return labels

    def _glob_subject_dirs(self, data_dir: Path) -> list[Path]:
        subject_dirs = []
        for subset_name in self.subset_names:
            subset_dir = data_dir / subset_name
            new_subjects = [
                subject for subject in subset_dir.glob('*') if subject.is_dir()
            ]
            if len(new_subjects) == 0:
                print(f'no subjects in {subset_dir}')
            subject_dirs.extend(new_subjects)
        subject_dirs: list[Path] = sorted(subject_dirs)
        # remove broken subject dirs
        if 'tum_glioma' in self.subset_names:
            broken_subjects = self._load_broken_subjects()

            subject_dirs = [
                s for s in subject_dirs
                if self._make_patient_id(s) not in broken_subjects
            ]
        # remove subjects without preop
        no_preop_subjects = self._load_no_preop_subjects()
        if len(no_preop_subjects) > 0:
            subject_dirs = [
                s for s in subject_dirs
                if self._make_patient_id(s) not in no_preop_subjects
            ]

        return subject_dirs

    def _load_no_preop_subjects(self) -> list[str]:
        # make absolute path
        if not self._no_preop_fp.exists():
            # first time running, no file exists
            return []

        with open(self._no_preop_fp) as f:
            no_preop_subjects = [l.strip() for l in f.readlines()]
        return no_preop_subjects

    def _load_broken_subjects(self) -> list[str]:
        fp = 'datasets/broken_subjects.txt'
        # make absolute path
        fp = Path(__file__).resolve().parents[2] / fp

        with open(fp) as f:
            broken_subjects = [l.strip() for l in f.readlines()]
        return broken_subjects

    def sample_weights(self):
        """Sample weights for each subject."""
        cls_weights = 1 / np.array(self.samples_per_cls)
        print(f'cls_weights: {cls_weights}')
        subject_dir_to_cls_label = lambda s: self.cls_labels[
            self._make_patient_id(s)]
        sample_weights = [
            cls_weights[subject_dir_to_cls_label(s)] for s in self.subject_dirs
        ]
        return sample_weights
