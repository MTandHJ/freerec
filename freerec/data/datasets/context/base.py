

from typing import Optional
import os
import pandas as pd
import torchdata.datapipes as dp

from ..base import RecDataSet
from ...tags import TIMESTAMP, USER, ITEM, ID
from ...fields import Field, SparseField, DenseField
from ....utils import infoLogger
from ....dict2obj import Config


__all__ = []


class ContextAwareRecSet(RecDataSet):
    DATATYPE =  "Context"

    def check(self):
        assert isinstance(self.fields[TIMESTAMP], Field), "ContextAwareRecSet must have `TIMESTAMP' field."

    def load_meta(self, root: SparseField, meta_infos: Config):
        try:
            meta_df = pd.read_csv(
                os.path.join(self.path, meta_infos.meta_file),
                sep=meta_infos.sep
            )
            fields =  root.bind(
                meta_df,
                rootCol=meta_infos.rootCol,
                mapper=meta_infos.mapper
            )
            self.fields = self.fields + fields
            infoLogger(
                f"[{self.__class__.__name__}] >>> Load meta data for {root}"
            )
        except FileNotFoundError:
            pass


class TripletWithMeta(ContextAwareRecSet):

    VALID_IS_TEST = False

    _cfg = Config(
        fields = [
            (SparseField, Config(name='UserID', na_value=None, dtype=int, tags=[USER, ID])),
            (SparseField, Config(name='ItemID', na_value=None, dtype=int, tags=[ITEM, ID])),
            (DenseField, Config(name='Timestamp', na_value=None, dtype=float, tags=[TIMESTAMP], transformer='none'))
        ],
        userMeta = Config(meta_file="user.txt", sep='\t', rootCol=0, mapper=dict()),
        itemMeta = Config(meta_file="item.txt", sep='\t', rootCol=0, mapper=dict()),
    )

    open_kw = Config(mode='rt', delimiter='\t', skip_lines=1)

    def __init__(self, root: str, filename: Optional[str] = None, download: bool = True) -> None:
        super().__init__(root, filename, download)

        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        self.load_meta(User, self._cfg.userMeta)
        self.load_meta(Item, self._cfg.itemMeta)


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