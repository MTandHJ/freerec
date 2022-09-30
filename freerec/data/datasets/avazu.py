

import torchdata.datapipes as dp

from .base import RecDataSet
from ..fields import SparseField
from ..tags import FEATURE, TARGET, ITEM
from ...dict2obj import Config


__all__ = ['Avazu_x1']


class Avazu_x1(RecDataSet):
    """A Kaggle challenge dataset for Avazu CTR prediction.
    |     Dataset      |   Total    |   #Train   | #Validation |   #Test   |
    | :--------------: | :--------: | :--------: | :---------: | :-------: |
    | Avazu_x1 (7:1:2) | 40,428,967 | 28,300,276 |  4,042,897  | 8,085,794 |

    This dataset contains about 10 days of labeled click-through data on mobile advertisements. 
    It has 22 feature fields including user features and advertisement attributes. 
    See [here](https://github.com/openbenchmark/BARS/tree/master/ctr_prediction/datasets/Avazu) for details.

    Attributes:
    ---

    _cfg: Config
        - target: SparseField
            Label
        - features: SparseField
            Hour + C1 - C21, in total of 22 features. 
    open_kw: Config
        - mode: 'rt'
        - delimiter: ','
        - skip_lines: 1
    """

    _cfg = Config(
        target = [SparseField('Label', na_value=None, dtype=int, transformer='none', tags=TARGET)],
        features = [SparseField(f"Feat{k}", na_value=-1, dtype=int, tags=[ITEM, FEATURE]) for k in range(1, 23)]
    )

    _cfg.fields = _cfg.target + _cfg.features

    open_kw = Config(mode='rt', delimiter=',', skip_lines=1)

    def __init__(self, root: str, **open_kw) -> None:
        super().__init__(root)
        self.open_kw.update(**open_kw)
        self.compile()

    def file_filter(self, filename: str):
        return self.mode in filename

    def raw2data(self) -> dp.iter.IterableWrapper:
        datapipe = dp.iter.FileLister(self.root)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = datapipe.map(self.row_processer)
        # data = list(datapipe)
        # if self.mode == 'train':
        #     random.shuffle(data)
        # datapipe = dp.iter.IterableWrapper(data)
        return datapipe


