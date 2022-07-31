


from typing import Iterable, Iterator, Any, Hashable, Union

import torch
import torchdata.datapipes as dp
import pandas as pd
from collections import defaultdict
from itertools import chain

from ..features import SparseField, DenseField
from ...dict2obj import Config



__all__ = ['RecDataSet', 'Encoder']

def _add(field: defaultdict, val: Hashable, key: str = 'nums'):
    field[key][val] += 1

def _min(field: dict, val: Union[int, float], key: str = 'min'):
    field[key] = min(val, field[key])

def _max(field: dict, val: Union[int, float], key: str = 'max'):
    field[key] = max(val, field[key])


STATISTICS = {
    "nums": _add, # sparse
    "min": _min, # dense
    "max": _max # dense
}


class RecDataSet(dp.iter.IterDataPipe):

    cfg = Config()

    def __str__(self) -> str:
        head = super().__str__()
        cfg = str(self.cfg)
        return "DataSet: " + head + "\n" + cfg

    def summary(self):
        data = dict(datasize=0)
        data.update({field: dict(nums=defaultdict(int)) for field in self.cfg.sparse.keys()})
        data.update({field: dict(min=float('inf'), max=float('-inf')) for field in self.cfg.dense.keys()})
        data.update({field: dict(nums=defaultdict(int)) for field in self.cfg.label.keys()})

        for row in self:
            data['datasize'] += 1
            for field, val in row.items():
                for key in data[field]:
                    STATISTICS[key](data[field], val, key)

        for fields in self.cfg.values():
            for field in fields.values(): 
                item = data[field.name]
                if isinstance(field, SparseField):
                    field.fit(list(sorted(item['nums'].keys())))
                elif isinstance(field, DenseField):
                    field.fit(lower=item['min'], upper=item['max'])
        self.cfg.datasize = data['datasize']


@dp.functional_datapipe("sharder")
class Sharder(dp.iter.IterDataPipe):

    def __init__(self, datapipe: dp.iter.IterDataPipe) -> None:
        super().__init__()

        self.source = datapipe

    def __iter__(self) -> Iterator:
        worker_infos = torch.utils.data.get_worker_info()
        if worker_infos:
            id_, nums = worker_infos.id, worker_infos.num_workers
            for idx, item in enumerate(self.source):
                if idx % nums == id_:
                    yield item
        else:
            yield from self.source


@dp.functional_datapipe("encoder")
class Encoder(dp.iter.IterDataPipe):

    def __init__(
        self, datapipe: RecDataSet, batch_size: int, 
        shuffle: bool = True, buffer_size: int = 10000,
        fields: Iterable = None
    ) -> None:
        super().__init__()

        assert buffer_size > batch_size, "buffer_size should be greater than batch_size ..."

        self.source = datapipe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = (buffer_size // batch_size) * batch_size

        self.cfg = self.source.cfg
        if fields is None:
            self.fields = list(self.cfg.fields.keys())
        else:
            self.fields = fields

    def fields_filter(self, row: dict):
        return {field: row[field] for field in self.fields}

    def batch_processor(self, batch: pd.DataFrame) -> pd.DataFrame:
        return  {field: self.cfg.fields[field].transform(batch[field]) for field in self.fields}

    def __iter__(self) -> Iterator:
        datapipe = self.source
        datapipe = datapipe.map(self.fields_filter)
        if self.shuffle:
            datapipe = datapipe.shuffle(buffer_size=self.shuffle)
        datapipe = datapipe.sharder()
        datapipe = datapipe.batch(batch_size=self.batch_size)
        datapipe = datapipe.collate()
        datapipe = datapipe.map(self.batch_processor)
        yield from datapipe
        


