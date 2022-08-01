


from typing import Iterator 

import torchdata.datapipes as dp

from .base import RecDataSet
from ..features import SparseField, DenseField
from ...dict2obj import Config



class Criteo(RecDataSet):

    cfg = Config(
        # dataset
        sparse = [SparseField(name=f"C{idx}", na_value='-1', dtype=str) for idx in range(1, 27)],
        dense = [DenseField(name=f"I{idx}", na_value=0., dtype=float) for idx in range(1, 14)],
        label = [SparseField(name='label', na_value=None, dtype=int)]
    )
    cfg.fields = cfg.label + cfg.dense + cfg.sparse

    open_kw = Config(mode='rt', delimiter=',', skip_lines=1)

    def __init__(
        self, root: str, split: str = 'train', **kwargs
    ) -> None:
        super().__init__()

        self.root = root
        self.split = split
        self.open_kw.update(**kwargs) # override some basic kwargs for loading

        self.summary()

    def file_filter(self, filename: str):
        return self.split in filename

    def row_processer(self, row):
        return {
            field.name: field(val) for val, field in zip(row, self.cfg.fields)
        }

    def __iter__(self) -> Iterator:
        datapipe = dp.iter.FileLister(self.root)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = datapipe.map(self.row_processer)
        yield from datapipe


    
