

from typing import Iterator 

import torchdata.datapipes as dp

from .base import RecDataSet
from ..fields import SparseField, DenseField, LabelField, Binarizer
from ...dict2obj import Config


class MovieLens1M(RecDataSet):
    """
    MovieLens1M: (user, item, rating, timestamp)
        https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets
    """

    _cfg = Config(
        sparse = [SparseField(name=name, na_value=0, dtype=int) for name in ('User', 'Item')],
        dense = [DenseField(name="Timestamp", na_value=0., dtype=float, transformer='none')],
        target = [LabelField(name='Rating', na_value=None, dtype=int, transformer='none')]
    )

    _cfg.fields = _cfg.sparse + _cfg.target + _cfg.dense

    _active = False

    open_kw = Config(mode='rt', delimiter='\t', skip_lines=0)

    def __init__(self, root: str, split: str = 'train', **open_kw) -> None:
        super().__init__(root, split)
        self.open_kw.update(**open_kw)
        self._compile(refer='train')

    def file_filter(self, filename: str):
        return self.split in filename

    def row_processer(self, row):
        return {
            field.name: field.caster(val) for val, field in zip(row, self.cfg.fields)
        }

    def __iter__(self) -> Iterator:
        datapipe = dp.iter.FileLister(self.root)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = datapipe.map(self.row_processer)
        yield from datapipe


    
