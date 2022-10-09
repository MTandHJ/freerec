

import torchdata.datapipes as dp
import random

from .base import RecDataSet
from ..fields import SparseField, DenseField
from ..tags import USER, ITEM, ID, FEATURE, TARGET
from ...dict2obj import Config


__all__ = ['MovieLens1M_m2']


class MovieLens1M_m2(RecDataSet):
    """ MovieLens1M: (user, item, rating, timestamp)
    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

    Attritbutes:
    ---

    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
        - dense: DenseField
            Timestamp
        - target: SparseField
            Rating
    open_kw: Config
        - mode: 'rt'
        - delimiter: '\\t'
        - skip_lines: 0

    """

    URL = "https://zenodo.org/record/7175950/files/MovieLens1M_m2.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=0, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=0, dtype=int, tags=[ITEM, ID]),
        ],
        dense = [DenseField(name="Timestamp", na_value=0., dtype=float, transformer='none', tags=FEATURE)],
        target = [DenseField(name='Rating', na_value=None, dtype=int, transformer='none', tags=TARGET)]
    )

    _cfg.fields = _cfg.sparse + _cfg.target + _cfg.dense

    open_kw = Config(mode='rt', delimiter='\t', skip_lines=0)

    def file_filter(self, filename: str):
        if self.mode == 'train':
            return 'train' in filename
        else:
            return 'test' in filename

    def raw2data(self) -> dp.iter.IterableWrapper:
        datapipe = dp.iter.FileLister(self.path)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = datapipe.map(self.row_processer)
        data = list(datapipe)
        if self.mode == 'train':
            random.shuffle(data)
        datapipe = dp.iter.IterableWrapper(data)
        return datapipe

    
