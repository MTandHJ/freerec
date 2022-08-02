


from typing import Dict, Iterable, Iterator, Hashable, Tuple, Union, List

import torch
import torchdata.datapipes as dp
from collections import defaultdict

from ..fields import Field, SparseField, DenseField
from ...dict2obj import Config



__all__ = ['RecDataSet', 'Encoder']

def _count(field: defaultdict, val: Hashable, key: str = 'nums'):
    field[key][val] += 1

def _min(field: dict, val: Union[int, float], key: str = 'min'):
    field[key] = min(val, field[key])

def _max(field: dict, val: Union[int, float], key: str = 'max'):
    field[key] = max(val, field[key])


STATISTICS = {
    "nums": _count, # sparse
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
        data.update({field.name: dict(nums=defaultdict(int)) for field in SparseField.filter(self.cfg.fields)})
        data.update({field.name: dict(min=float('inf'), max=float('-inf')) for field in DenseField.filter(self.cfg.fields)})

        for row in self:
            data['datasize'] += 1
            for field_name, val in row.items():
                for key in data[field_name]:
                    STATISTICS[key](data[field_name], val, key)

        self.cfg.datasize = data['datasize']
        for field in SparseField.filter(self.cfg.fields):
            item = data[field.name]
            field.fit(list(sorted(item['nums'].keys())))
        for field in DenseField.filter(self.cfg.fields):
            item = data[field.name]
            field.fit(lower=item['min'], upper=item['max'])


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

        self.fields: List[Field] = self.source.cfg.fields
        if fields:
            self.fields = [field for field in self.fields if field.name in fields]
        self.features = [field for field in self.fields if field.is_feature]
        self.target = [field for field in self.fields if not field.is_feature][0] # TODO: multi-targets ?

    def fields_filter(self, row: dict) -> Dict:
        return {field.name: row[field.name] for field in self.fields}

    def batch_processor(self, batch: List) -> Tuple[Dict, torch.Tensor]:
        features = {field.name: field.transform(batch[field.name]) for field in self.features}
        targets = self.target.transform(batch[self.target.name])
        return features, targets

    def __iter__(self) -> Iterator:
        datapipe = self.source
        datapipe = datapipe.map(self.fields_filter)
        if self.shuffle:
            datapipe = datapipe.shuffle(buffer_size=self.buffer_size)
        datapipe = datapipe.sharder()
        datapipe = datapipe.batch(batch_size=self.batch_size)
        datapipe = datapipe.collate()
        datapipe = datapipe.map(self.batch_processor)
        yield from datapipe
        


