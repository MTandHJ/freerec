

import numpy as np
import torchdata.datapipes as dp

from ..base import RecDataSet
from ...tags import TIMESTAMP, USER, ITEM, ID
from ...fields import Field, SparseField, DenseField
from ....utils import infoLogger
from ....dict2obj import Config


__all__ = ['SequentialRecSet']


class SequentialRecSet(RecDataSet):
    DATATYPE =  "Sequential"

    def check(self):
        assert isinstance(self.fields[TIMESTAMP], Field), "SequentialRecSet must have `TIMESTAMP' field."

    # def raw2pickle(self):
    #     """Convert raw data into (ordered) pickle format."""
    #     infoLogger(f"[{self.__class__.__name__}] >>> Convert raw data ({self.mode}) to chunks in pickle format")
    #     userIndex = self.fields.index(USER, ID)
    #     timeIndex = self.fields.index(TIMESTAMP)
    #     data = sorted(self.raw2data(), key=lambda row: (row[userIndex], row[timeIndex]))
    #     datapipe = dp.iter.IterableWrapper(data)

    #     count = 0
    #     for chunk in datapipe.batch(batch_size=self.DEFAULT_CHUNK_SIZE).column_():
    #         for j, field in enumerate(self.fields):
    #             chunk[j] = field.transform(chunk[j])
    #         self.write_pickle(chunk, count)
    #         count += 1
    #     infoLogger(f"[{self.__class__.__name__}] >>> {count} chunks done")


class UserItemTimeTriplet(SequentialRecSet):

    VALID_IS_TEST = False

    _cfg = Config(
        fields = [
            (SparseField, Config(name='UserID', na_value=None, dtype=int, tags=[USER, ID])),
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
        User, Item = self.fields[USER, ID], self.fields[ITEM, ID]

        table = PrettyTable(['#Users', '#Items', 'Avg.Len', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        table.add_row([
            User.count, Item.count, self.train().meanlen + 2,
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
            (self.trainsize + self.validsize + self.testsize) / (User.count * Item.count)
        ])

        infoLogger(table)