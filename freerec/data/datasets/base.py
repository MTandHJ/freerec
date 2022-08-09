


from typing import Dict, Iterable, Iterator, Optional, Tuple, List

import torch
import torchdata.datapipes as dp

from ..fields import Field
from ...utils import warnLogger


__all__ = ['RecDataSet', 'Encoder']


class RecDataSet(dp.iter.IterDataPipe):
    """ RecDataSet provides a template for specific datasets.
    All datasets inherit RecDataSet should define class variables:
        _cfg: including fields of each column,
        _active: True if the type of dataset has compiled ...
    before instantiation.
    Generally speaking, the dataset will be splited into 
        trainset,
        validset,
        testset.
    Because these three datasets share the same _cfg, compiling any one of them
    will overwrite it ! So you should compile the trainset rather than other datasets by
        trainset.compile() !
    """

    def __new__(cls, *args, **kwargs):
        for attr in ('_cfg', '_active'):
            if not hasattr(cls, attr):
                raise AttributeError("_cfg, _active should be defined before instantiation ...")
        return super().__new__(cls)

    def __init__(self, root: str, split: str = 'train') -> None:
        """
        root: data file
        split: train|valid|test
        """
        super().__init__()
        self.root = root
        self.split = split

    @property
    def cfg(self):
        return self._cfg

    @property
    def active(self):
        return self._active

    @classmethod
    def compile(cls, datapipe: dp.iter.IterDataPipe):
        if cls._active:
            warnLogger(
                f"Dataset {cls.__name__} has been activated !!! Skip it ..."
            )
        cls._active = True
        datapipe = datapipe.batch(batch_size=1000).collate()
        cls._cfg.datasize = 0
        for batch in datapipe:
            for field in cls._cfg.fields:
                field.partial_fit(batch[field.name])
                batchsize = len(batch[field.name])
            cls._cfg.datasize += batchsize

    def _compile(self, refer: str = 'train'):
        if not self.active:
            split, self.split = self.split, refer
            self.compile(self) # compile according to the trainset !
            self.split = split

    def __str__(self) -> str:
        head = super().__str__()
        cfg = str(self.cfg)
        return "DataSet: " + head + "\n" + cfg



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
        


