

import torchdata.datapipes as dp
import random

from .base import RecDataSet
from ..fields import SparseField, DenseField
from ..tags import USER, ITEM, ID, FEATURE, TARGET
from ...dict2obj import Config


__all__ = ['AmazonBooks_m1', 'AmazonCDs_m1']


class _Row2Pairer(dp.iter.IterDataPipe):

    def __init__(self, datapipe: dp.iter.IterDataPipe) -> None:
        super().__init__()
        self.source = datapipe

    def __iter__(self):
        for row in self.source:
            user = row[0]
            for item in row[1:]:
                yield user, item, 1


class AmazonBooks_m1(RecDataSet):
    """Amazon-Books dataset.
    |    Dataset     | #Users | #Items | #Interactions |  #Train   |  #Test  | Density |
    |:--------------:|:------:|:------:|:-------------:|:---------:|:-------:|:-------:|
    | AmazonBooks_m1 | 52,643 | 91,599 |   2,984,108   | 2,380,730 | 603,378 | 0.00062 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets/Amazon#AmazonBooks_m1) for details.

    Attributes:
    ---

    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
        - target: SparseField
            Rating
    open_kw: Config
        - mode: 'rt'
        - delimiter: ' '
        - skip_lines: 0

    """

    URL = "https://zenodo.org/record/7184851/files/amazon-book.zip"

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=0, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=0, dtype=int, tags=[ITEM, ID]),
        ],
        target = [DenseField(name='Rating', na_value=None, dtype=int, transformer='none', tags=TARGET)]
    )

    _cfg.fields = _cfg.sparse + _cfg.target

    open_kw = Config(mode='rt', delimiter=' ', skip_lines=0)


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
        datapipe = _Row2Pairer(datapipe)
        datapipe = datapipe.map(self.row_processer)
        data = list(datapipe)
        if self.mode == 'train':
            random.shuffle(data)
        datapipe = dp.iter.IterableWrapper(data)
        return datapipe


class AmazonCDs_m1(RecDataSet):
    """Amazon-CDs dataset.
    |   Dataset    | #Users | #Items | #Interactions | #Train  |  #Test  | Density |
    |:------------:|:------:|:------:|:-------------:|:-------:|:-------:|:-------:|
    | AmazonCDs_m1 | 43,169 | 35,648 |    777,426    | 604,475 | 172,951 | 0.00051 |

    See [here](https://github.com/openbenchmark/BARS/tree/master/candidate_matching/datasets) for details.

    Attributes:
    ---

    _cfg: Config
        - sparse: SparseField
            UserID + ItemID
        - target: SparseField
            Rating
    open_kw: Config
        - mode: 'rt'
        - delimiter: ' '
        - skip_lines: 0

    """

    URL = None

    _cfg = Config(
        sparse = [
            SparseField(name='UserID', na_value=0, dtype=int, tags=[USER, ID]),
            SparseField(name='ItemID', na_value=0, dtype=int, tags=[ITEM, ID]),
        ],
        target = [DenseField(name='Rating', na_value=None, dtype=int, transformer='none', tags=TARGET)]
    )

    _cfg.fields = _cfg.sparse + _cfg.target

    open_kw = Config(mode='rt', delimiter=' ', skip_lines=0)


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
        datapipe = _Row2Pairer(datapipe)
        datapipe = datapipe.map(self.row_processer)
        data = list(datapipe)
        if self.mode == 'train':
            random.shuffle(data)
        datapipe = dp.iter.IterableWrapper(data)
        return datapipe