

import torchdata.datapipes as dp

from .base import RecDataSet
from ..fields import SparseField, DenseField
from ..tags import FEATURE, TARGET, ITEM
from ...dict2obj import Config


__all__ = ['Criteo_x1']


class Criteo_x1(RecDataSet):
    """A Kaggle challenge for Criteo (7:2:1) display advertising.
    |     Dataset      |   Total    |   #Train   | #Validation |   #Test   |
    | :--------------: | :--------: | :--------: | :---------: | :-------: |
    |    Criteo_x1     | 45,840,617 | 33,003,326 |  8,250,124  | 4,587,167 |

    The Criteo dataset is a widely-used benchmark dataset for CTR prediction, 
    which contains about one week of click-through data for display advertising. 
    It has 13 numerical feature fields and 26 categorical feature fields.
    See [here](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/datasets/Criteo) for details.

    Attributes:
    ---

    _cfg: Config
        - sparse: SparseField
        - dense: DenseField
            Note that all dense features will be normalized by MinMaxScaler in default.
        - target: SparseField
            Label
    open_kw: Config
        - mode: 'rt'
        - delimiter: ','
        - skip_lines: 1
    """

    URL = "https://zenodo.org/record/5700987/files/Criteo_x1.zip"

    _cfg = Config(
        # dataset
        sparse = [SparseField(name=f"C{idx}", na_value='-1', dtype=str, tags=[ITEM, FEATURE]) for idx in range(1, 27)],
        dense = [DenseField(name=f"I{idx}", na_value=0., dtype=float, tags=[ITEM, FEATURE]) for idx in range(1, 14)],
        target = [SparseField(name='Label', na_value=None, dtype=int, transformer='none', tags=TARGET)]
    )
    _cfg.fields = _cfg.target + _cfg.dense + _cfg.sparse

    open_kw = Config(mode='rt', delimiter=',', skip_lines=1)


    def file_filter(self, filename: str):
        return self.mode in filename

    def raw2data(self) -> dp.iter.IterableWrapper:
        datapipe = dp.iter.FileLister(self.path)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = datapipe.map(self.row_processer)
        # data = list(datapipe)
        # if self.mode == 'train':
        #     random.shuffle(data)
        # datapipe = dp.iter.IterableWrapper(data)
        return datapipe
