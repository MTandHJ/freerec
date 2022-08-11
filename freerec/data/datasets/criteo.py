


from typing import Iterator 

import torchdata.datapipes as dp

from .base import RecDataSet
from ..fields import SparseField, DenseField
from ..tags import FEATURE, TARGET
from ...dict2obj import Config



class Criteo(RecDataSet):

    _cfg = Config(
        # dataset
        sparse = [SparseField(name=f"C{idx}", na_value='-1', dtype=str, tags=FEATURE) for idx in range(1, 27)],
        dense = [DenseField(name=f"I{idx}", na_value=0., dtype=float, tags=FEATURE) for idx in range(1, 14)],
        target = [SparseField(name='Label', na_value=None, dtype=int, transformer='none', tags=TARGET)]
    )
    _cfg.fields = _cfg.target + _cfg.dense + _cfg.sparse

    _active = False

    open_kw = Config(mode='rt', delimiter=',', skip_lines=1)

    def __init__(self, root: str, split: str = 'train', **open_kw) -> None:
        super().__init__(root, split)
        self.open_kw.update(**open_kw)
        self._compile(refer='train')

    def file_filter(self, filename: str):
        return self.split in filename

    def row_processer(self, row):
        return {
            field.name: field.caster(val) for val, field in zip(row, self.fields)
        }

    def __iter__(self) -> Iterator:
        datapipe = dp.iter.FileLister(self.root)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = datapipe.map(self.row_processer)
        yield from datapipe


    
