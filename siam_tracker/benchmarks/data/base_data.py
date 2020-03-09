""" Base class for benchmarks """
import numpy as np
from typing import Union, List


class Sequence(object):

    def __init__(self,
                 name: str = '',
                 frames: List[str] = None,
                 gt_rects: Union[List, np.ndarray] = None,
                 attrs: List[str] = None,
                 zip_path: str = None, **kwargs):
        """ Video sequence class.
        Args:
            name (str): sequence name
            frames (List[str]): sequence path
            gt_rects (List[List[float]]): ground-truth bounding box, in order of (x1, y1, w, h)
            attrs (List[str]): attribute list
            zip_path (str): zip file path. If it is not None, the zip frames will be pre-loaded. It's useful
                            when testing on the cluster where the data should be loaded through network.
        """
        self.name = name
        self.frames = frames
        self.gt_rects = gt_rects
        self.attrs = attrs

        self.zip_content = None
        if zip_path is not None:
            # read zip content
            fid = open(zip_path, 'rb')
            try:
                self.zip_content = fid.read()
            finally:
                fid.close()

        num_frames = len(self.frames)
        for k, v in kwargs.items():
            if v is not None:
                assert len(v) == num_frames, "The tag should be same size with frames. ({} vs {}) [{}/{}]".format(
                    len(v), num_frames, name, k
                )
            setattr(self, k, v)

    def __len__(self):
        return len(self.frames)


class Dataset(object):

    zero_based_index = True  # OTB uses Matlab's format (1-based indexing)

    def __init__(self, name, zip_mode=False):
        self._name = name  # dataset name
        self._seqs = []  # sequence list
        self._seq_name_to_id = None
        self.zip_mode = zip_mode

        if self.zip_mode:
            print("Use zip file of {} dataset".format(self._name))
            print("Slow initialization speed is expected.")

    def get(self, seq_name: str) -> Sequence:
        if self._seq_name_to_id is None:
            self._seq_name_to_id = {seq.name: idx for idx, seq in enumerate(self._seqs)}
        return self._seq_name_to_id[seq_name]

    @property
    def name(self) -> str:
        return self._name

    @property
    def seqs(self) -> List[Sequence]:
        return self._seqs

    @property
    def seq_names(self) -> List[str]:
        return [seq.name for seq in self._seqs]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, item) -> Sequence:
        if isinstance(item, str):
            return self.get(item)
        elif isinstance(item, int):
            return self._seqs[item]
        else:
            raise ValueError("unknown type {}".format(type(item)))
