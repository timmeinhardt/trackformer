# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset

from .mot17_sequence import MOT17Sequence
from .mot20_sequence import MOT20Sequence
from .mots20_sequence import MOTS20Sequence


class MOT17Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, dets: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT17Sequence dataset
        """
        train_sequences = [
            'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
            'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = [
            'MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
            'MOT17-08', 'MOT17-12', 'MOT17-14']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError("MOT17 split not available.")

        self._data = []
        for seq in sequences:
            if dets == 'ALL':
                self._data.append(MOT17Sequence(seq_name=seq, dets='DPM', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='FRCNN', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='SDP', **kwargs))
            else:
                self._data.append(MOT17Sequence(seq_name=seq, dets=dets, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class MOT20Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05',]
        test_sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08',]

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT20-{split}"]
        else:
            raise NotImplementedError("MOT20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOT20Sequence(seq_name=seq, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]


class MOTS20Wrapper(MOT17Wrapper):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOTS20Sequence dataset
        """
        train_sequences = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
        test_sequences = ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOTS20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOTS20-{split}"]
        else:
            raise NotImplementedError("MOTS20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOTS20Sequence(seq_name=seq, **kwargs))
