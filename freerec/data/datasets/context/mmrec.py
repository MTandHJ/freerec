

import torchdata.datapipes as dp
from .base import ContextAwareRecSet
from ...tags import USER, ITEM, ID, TIMESTAMP
from ...fields import SparseField, DenseField, Field
from ....utils import infoLogger
from ....dict2obj import Config


__all__ = [
    'DataSetFromMMRec',
]


class DataSetFromMMRec(ContextAwareRecSet):
    r"""
    Dataset from [MMRec](https://github.com/enoche/MMRec).
    """

    VALID_IS_TEST = False

    _cfg = Config(
        fields = [
            (SparseField, Config(name='UserID', na_value=None, dtype=int, tags=[USER, ID])),
            (SparseField, Config(name='ItemID', na_value=None, dtype=int, tags=[ITEM, ID])),
            (DenseField, Config(name='Timestamp', na_value=None, dtype=float, tags=[TIMESTAMP], transformer='none'))
        ],
    )

    open_kw = Config(mode='rt', delimiter='\t', skip_lines=1)

    def check(self):
        assert isinstance(self.fields[TIMESTAMP], Field), "DataSetFromMMRec must have `TIMESTAMP' field."

    def file_filter(self, filename: str):
        return self.mode in filename

    def raw2data(self) -> dp.iter.IterableWrapper:
        datapipe = dp.iter.FileLister(self.path)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = datapipe.map(self.row_processer)
        return datapipe

    def summary(self):
        super().summary()
        from prettytable import PrettyTable
        User, Item = self.fields[USER, ID], self.fields[ITEM, ID]

        table = PrettyTable(['#Users', '#Items', 'Avg.Len', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        table.add_row([
            User.count, Item.count, self.train().meanlen + 2,
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
            (self.trainsize + self.validsize + self.testsize) / (User.count * Item.count)
        ])

        infoLogger(table)