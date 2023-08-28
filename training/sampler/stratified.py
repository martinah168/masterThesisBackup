from typing import Iterator

import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler


class UniformStratifiedSampler(Sampler[int]):
    labels: Tensor

    def __init__(self, batch_size: int, labels: list[int], *args,
                 **kwargs) -> None:
        super().__init__(data_source=labels, *args, **kwargs)

        self.labels = torch.tensor(labels)
        self.strata: torch.IntTensor = torch.unique(self.labels)

        self.strata_samples: list[list[int]] = []
        for stratum in self.strata:
            # get indices of samples in the stratum
            stratum_samples = torch.where(self.labels == stratum)[0]

            # ensure that the permutation is not the same as the previous one by adding the last index
            self.strata_samples.append(stratum_samples.tolist())

        self.n_per_stratum = self._get_n_per_stratum(batch_size)

    def __iter__(self):
        # create iters for each stratum
        strata_samples_iter = [iter(s) for s in self.strata_samples]

        while True:
            # sample from each stratum
            for stratum in self.strata:
                # sample from the stratum
                for _ in range(self.n_per_stratum[stratum.item()]):
                    try:
                        # get next in line
                        stratum_sample = next(strata_samples_iter[stratum])
                        yield stratum_sample
                    except StopIteration:
                        strata_samples_iter[
                            stratum] = self._init_stratum_samples_iter(stratum)

    def _init_stratum_samples_iter(self, stratum: int) -> Iterator[int]:
        # get current samples
        cur_samples = self.strata_samples[stratum]

        # permute samples
        cur_samples = [cur_samples[i] for i in torch.randperm(len(cur_samples))]
        return iter(cur_samples)

    def _get_n_per_stratum(self, batch_size: int) -> dict[int, int]:
        return {
            int(s): max(1, batch_size // len(self.strata)) for s in self.strata
        }

    def __len__(self):
        return len(self.labels)


class WeightedStratifiedSampler(UniformStratifiedSampler):

    def __init__(self, batch_size: int, labels: list[int], *args,
                 **kwargs) -> None:
        super().__init__(batch_size, labels, *args, **kwargs)

    def _get_n_per_stratum(self, batch_size: int) -> dict[int, int]:
        max_p = 0.5
        rem_p = (1 - max_p) / (len(self.strata) - 1)
        max_n_samples = -1
        max_stratum = -1

        # determine the stratum with the most examples and the number of examples
        for stratum in self.strata:
            if (n_strata_samples := len(
                    self.strata_samples[stratum])) > max_n_samples:
                max_n_samples = n_strata_samples
                max_stratum = stratum

        out = {
            int(s): int((max_p if s == max_stratum else rem_p) * batch_size)
            for s in self.strata
        }

        return out
