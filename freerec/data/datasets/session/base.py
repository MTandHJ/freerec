

from typing import Tuple

import numpy as np
import torchdata.datapipes as dp

from ..base import RecDataSet
from ...tags import TIMESTAMP, SESSION, ITEM, ID
from ...fields import Field, SparseField, DenseField
from ....utils import infoLogger
from ....dict2obj import Config


__all__ = ['SessionBasedRecSet', 'SessionItemTimeTriplet']


class SessionBasedRecSet(RecDataSet):
    DATATYPE =  "Session"

    def check(self):
        assert isinstance(self.fields[TIMESTAMP], Field), "SessionRecSet must have `TIMESTAMP' field."

    def to_seqs(self, master: Tuple = (SESSION, ID), keepid: bool = False):
        return super().to_seqs(master, keepid)

    def to_roll_seqs(self, master: Tuple = (SESSION, ID), minlen: int = 2):
        return super().to_roll_seqs(master, minlen)


class SessionItemTimeTriplet(SessionBasedRecSet):

    VALID_IS_TEST = False

    _cfg = Config(
        fields = [
            (SparseField, Config(name='SessionID', na_value=None, dtype=int, tags=[SESSION, ID])),
            (SparseField, Config(name='ItemID', na_value=None, dtype=int, tags=[ITEM, ID])),
            (DenseField, Config(name='Timestamp', na_value=None, dtype=float, tags=[TIMESTAMP], transformer='none'))
        ]
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
        Session, Item = self.fields[SESSION, ID], self.fields[ITEM, ID]

        table = PrettyTable(['#Sessions', '#Items', 'Avg.Len', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        trainlens =  list(filter(lambda x: x > 0, [len(items) for items in self.train().to_seqs()]))
        validlens =  list(filter(lambda x: x > 0, [len(items) for items in self.valid().to_seqs()]))
        testlens =  list(filter(lambda x: x > 0, [len(items) for items in self.test().to_seqs()]))
        table.add_row([
            Session.count, Item.count,
            np.mean(
                trainlens + validlens + testlens
            ).item(),
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
            (self.trainsize + self.validsize + self.testsize) / (Session.count * Item.count)
        ])

        infoLogger(table)