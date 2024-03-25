

import torchdata.datapipes as dp
from .base import ContextAwareRecSet
from ...tags import USER, ITEM, ID
from ...fields import SparseField
from ....utils import infoLogger
from ....dict2obj import Config


__all__ = [
    'DataSetFromMMSSL',
    'Tiktok_from_MMSSL',
    'Allrecipes_from_MMSSL',
]


class DataSetFromMMSSL(ContextAwareRecSet):
    r"""
    Dataset used in [MMSSL](https://github.com/HKUDS/MMSSL).
    """

    VALID_IS_TEST = False

    _cfg = Config(
        fields = [
            (SparseField, Config(name='UserID', na_value=None, dtype=int, tags=[USER, ID])),
            (SparseField, Config(name='ItemID', na_value=None, dtype=int, tags=[ITEM, ID])),
        ],
    )

    open_kw = Config(mode='rt', delimiter='\t', skip_lines=1)

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


class Tiktok_from_MMSSL(DataSetFromMMSSL):
    r"""
    Tiktok dataset used in [MMSSL](https://github.com/HKUDS/MMSSL).

    +--------+--------+------------------+---------------+--------+--------+-------+-----------------------+
    | #Users | #Items |     Avg.Len      | #Interactions | #Train | #Valid | #Test |        Density        |
    +--------+--------+------------------+---------------+--------+--------+-------+-----------------------+
    |  9308  |  6710  | 8.40363519036352 |     68722     | 59541  |  3051  |  6130 | 0.0011003146500902705 |
    +--------+--------+------------------+---------------+--------+--------+-------+-----------------------+

    """
    URL = "https://zenodo.org/records/10868610/files/Tiktok_from_MMSSL.zip"


class Allrecipes_from_MMSSL(DataSetFromMMSSL):
    r"""
    Allrecipes dataset used in [MMSSL](https://github.com/HKUDS/MMSSL).

    +--------+--------+-------------------+---------------+--------+--------+-------+-----------------------+
    | #Users | #Items |      Avg.Len      | #Interactions | #Train | #Valid | #Test |        Density        |
    +--------+--------+-------------------+---------------+--------+--------+-------+-----------------------+
    | 19805  | 10068  | 4.975107296137339 |     73494     | 58922  |  6751  |  7821 | 0.0003685817531420022 |
    +--------+--------+-------------------+---------------+--------+--------+-------+-----------------------+

    """
    URL = "https://zenodo.org/records/10868610/files/Allrecipes_from_MMSSL.zip"