import os
import shutil
from io import BytesIO
from multiprocessing import Process, Queue

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from training.data.transforms import d2c_crop


class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{self.original_resolution}-{str(index).zfill(self.zfill)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img


class FFHQlmdb(Dataset):
    def __init__(
        self,
        path=os.path.expanduser("datasets/ffhq256.lmdb"),
        image_size=256,
        original_resolution=256,
        split=None,
        as_tensor: bool = True,
        do_augment: bool = True,
        do_normalize: bool = True,
        **kwargs,
    ):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == "train":
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == "test":
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {"img": img, "index": index}


class CelebAlmdb(Dataset):
    """
    also supports for d2c crop.
    """

    def __init__(
        self,
        path,
        image_size,
        original_resolution=128,
        split=None,
        as_tensor: bool = True,
        do_augment: bool = True,
        do_normalize: bool = True,
        crop_d2c: bool = False,
        **kwargs,
    ):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)
        self.crop_d2c = crop_d2c

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

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

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {"img": img, "index": index}


def convert(x, format, quality=100):
    # to prevent locking!
    torch.set_num_threads(1)

    buffer = BytesIO()
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    x = x.to(torch.uint8)
    x = x.numpy()
    img = Image.fromarray(x)
    img.save(buffer, format=format, quality=quality)
    val = buffer.getvalue()
    return val


class _WriterWroker(Process):
    def __init__(self, path, format, quality, zfill, q):
        super().__init__()
        if os.path.exists(path):
            shutil.rmtree(path)

        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.q = q
        self.i = 0

    def run(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with lmdb.open(self.path, map_size=1024**4, readahead=False) as env:
            while True:
                job = self.q.get()
                if job is None:
                    break
                with env.begin(write=True) as txn:
                    for x in job:
                        key = f"{str(self.i).zfill(self.zfill)}".encode("utf-8")
                        x = convert(x, self.format, self.quality)
                        txn.put(key, x)
                        self.i += 1

            with env.begin(write=True) as txn:
                txn.put("length".encode("utf-8"), str(self.i).encode("utf-8"))


class LMDBImageWriter:
    def __init__(self, path, format="webp", quality=100, zfill=7) -> None:
        self.path = path
        self.format = format
        self.quality = quality
        self.zfill = zfill
        self.queue = None
        self.worker = None

    def __enter__(self):
        self.queue = Queue(maxsize=3)
        self.worker = _WriterWroker(self.path, self.format, self.quality, self.zfill, self.queue)
        self.worker.start()

    def put_images(self, tensor):
        """
        Args:
            tensor: (n, c, h, w) [0-1] tensor
        """
        self.queue.put(tensor.cpu())
        # with self.env.begin(write=True) as txn:
        #     for x in tensor:
        #         key = f"{str(self.i).zfill(self.zfill)}".encode("utf-8")
        #         x = convert(x, self.format, self.quality)
        #         txn.put(key, x)
        #         self.i += 1

    def __exit__(self, *args, **kwargs):
        self.queue.put(None)
        self.queue.close()
        self.worker.join()


class LMDBImageReader(Dataset):
    def __init__(self, path, zfill: int = 7):
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{str(index).zfill(self.zfill)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img
