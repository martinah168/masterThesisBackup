import logging
import shutil
from pathlib import Path

import numpy as np
import torch.utils.data
import tqdm
from monai.transforms import (Compose, LoadImaged, Padd, Resized, SaveImaged,
                              SpatialPad)
from torchvision.utils import save_image

from training.data.mri import extract_slices_from_volume

nib_logger = logging.getLogger('nibabel')


def get_fn_for_seq(subject_dir: Path, mode: str) -> Path:
    fn = '_'.join([f'sub-{subject_dir.name}', f'ses-preop', f'space-sri', mode
                  ]) + '.nii.gz'
    return subject_dir / 'preop' / fn


def PrintTransform():

    def f(x):
        print(x)
        return x

    return f


class LoadTransformWriteDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: Path, target_size: int, visualize: bool,
                 do_save_images: bool) -> None:
        super().__init__()

        self.target_size = target_size
        self.visualize = visualize
        self.do_save_images = do_save_images
        self.mri_sequences = ('t1', 't1c', 't2', 'flair')
        self.brainmask = 'brainmask'
        self.seg_name = 'seg'
        self.data_dir = Path(data_dir)

        # self.target_dir = Path(f'test_glioma_public_{self.target_size}')
        self.target_dir = Path(str(self.data_dir) + f'_{target_size}')

        self.target_dir.mkdir(exist_ok=True)
        self._copy_csv()

        self.subset_names = ('brats_2021_train', 'brats_2021_valid', 'erasmus',
                             'lumiere', 'rembrandt', 'ucsf_glioma', 'upenn_gbm',
                             'tcga', 'tum_glioma')
        self.subject_dirs = self._glob_subject_names()

        self.img_transform = self._base_transform(np.float32)
        self.label_transform = self._base_transform(np.uint8)

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, index):

        subject_dir = self.subject_dirs[index]

        fns = [get_fn_for_seq(subject_dir, seq) for seq in self.mri_sequences]

        # load, resize, and save is included in the transform
        for fn in fns:
            try:
                nib_logger.setLevel(logging.ERROR)  # prevent nibabel warnings
                out = self.img_transform({'image': fn})
                self._vis_out(out)
            except RuntimeError:
                # save fn to text file
                self.save_as_broken(fn)
                # continue

        # load and transform labels
        try:
            seg_fn = get_fn_for_seq(subject_dir, self.seg_name)
            out = self.label_transform({'image': seg_fn})
            self._vis_out(out)
        except RuntimeError:
            self.save_as_broken(seg_fn)

        # load and transform brainmask
        if (brainmask_fn := get_fn_for_seq(subject_dir,
                                           self.brainmask)).exists():
            try:
                out = self.label_transform({'image': brainmask_fn})
                self._vis_out(out)
            except RuntimeError:
                self.save_as_broken(brainmask_fn)

        else:
            if self.visualize:
                print(f'no brainmask for {subject_dir}')
        return out

    def save_as_broken(self, fn: Path):
        print(f'could not load {fn}')
        subject = '/'.join(fn.parts[-4:-2])
        with open('datasets/broken_subjects.txt', 'a') as f:
            f.write(f'{subject}\n')
        with open('datasets/broken_files.txt', 'a') as f:
            f.write(f'{fn}\n')

    def _copy_csv(self):
        csv_fns = self.data_dir.glob('*.csv')
        for csv_fn in csv_fns:
            shutil.copy(csv_fn, self.target_dir)

    def _vis_out(self, out):
        if not self.visualize:
            return
        img_slice = extract_slices_from_volume(out['image'])

        save_image(img_slice.float(), 'test_img/low_res.png', normalize=True)
        print('', end='')

    def _base_transform(self, dtype: np.dtype):
        compose = [
            # PrintTransform(),
            LoadImaged(keys='image',
                       image_only=True,
                       ensure_channel_first=True,
                       simple_keys=True,
                       dtype=dtype),
            Resized(keys='image',
                    spatial_size=self.target_size,
                    size_mode='longest',
                    mode='area' if dtype == np.float32 else 'nearest'),
        ]
        if self.visualize:
            compose.append(
                Padd(
                    keys='image',
                    padder=SpatialPad(spatial_size=self.target_size,
                                      mode='minimum',
                                      method='symmetric'),
                ))
        if self.do_save_images:
            compose.append(
                SaveImaged(keys='image',
                           output_dir=self.target_dir,
                           data_root_dir=self.data_dir,
                           output_ext='.nii.gz',
                           output_dtype=dtype,
                           output_postfix='',
                           resample=False,
                           squeeze_end_dims=True,
                           writer='NibabelWriter',
                           separate_folder=False,
                           print_log=False))
        return Compose(compose)

    def _glob_subject_names(self):
        subject_dirs = []
        for subset_name in self.subset_names:
            subset_dir = self.data_dir / subset_name
            new_subjects = [
                subject for subject in subset_dir.glob('*') if subject.is_dir()
            ]
            if len(new_subjects) == 0:
                print(f'no subjects in {subset_dir}')
            subject_dirs.extend(new_subjects)
        subject_dirs = sorted(subject_dirs)
        print(f'found {len(subject_dirs)} subjects')
        subject_dirs = [
            subject_dir for subject_dir in subject_dirs
            if (subject_dir / 'preop').exists()
        ]
        print(f'found {len(subject_dirs)} subjects with preop')
        return subject_dirs


def main():
    data_dir = Path('~/datasets/glioma_public').expanduser()
    num_workers = 16
    target_size = 64
    do_save_images = False
    assert not do_save_images, 'do not save images anymore'
    dataset = LoadTransformWriteDataset(data_dir=data_dir,
                                        target_size=target_size,
                                        visualize=False,
                                        do_save_images=do_save_images)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         persistent_workers=num_workers > 0,
                                         prefetch_factor=128)

    for batch in tqdm.tqdm(loader, total=len(loader)):
        pass


if __name__ == '__main__':
    main()
