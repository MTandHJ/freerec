

from dataclasses import replace
from typing import Dict, Iterable, Iterator, Optional, Tuple, List, Union

import torch
import torchdata.datapipes as dp
import numpy as np
import pandas as pd
import feather
import random
import os

from ..fields import Field, SparseField
from ..tags import Tag, FEATURE, TARGET, USER, ITEM, ID
from ...utils import timemeter, warnLogger, getLogger


__all__ = ['RecDataSet', 'Postprocessor']


_DEFAULT_CHUNK_SIZE = 51200


class BaseSet(dp.iter.IterDataPipe):

    def __init__(self) -> None:
        super().__init__()

        self.__mode = 'train'
        
    @property
    def mode(self):
        return self.__mode

    def train(self):
        self.__mode = 'train'

    def valid(self):
        self.__mode = 'valid'

    def test(self):
        self.__mode = 'test'

class RecDataSet(BaseSet):
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
        for attr in ('_cfg',):
            if not hasattr(cls, attr):
                raise AttributeError("_cfg, should be defined before instantiation ...")
        assert hasattr(cls._cfg, 'fields'), "Fields sorted by column should be given in _cfg ..."
        return super().__new__(cls)

    def __init__(self, root: str) -> None:
        """
        root: data file
        """
        super().__init__()
        self.root = root

    @property
    def cfg(self):
        return self._cfg

    @property
    def fields(self):
        return self._cfg.fields

    def check_feather(self):
        path = os.path.join(self.root, f"{self.__class__.__name__}2feather", self.mode)
        if os.path.exists(path):
            return any(True for _ in os.scandir(path))
        else:
            os.makedirs(path)
            return False

    def write_feather(self, dataframe: pd.DataFrame, count: int):
        file_ = os.path.join(self.root, f"{self.__class__.__name__}2feather", self.mode, f"chunk{count}.feather")
        feather.write_dataframe(dataframe, file_)

    def read_feather(self, file_: str):
        return feather.read_dataframe(file_)

    def raw2data(self) -> dp.iter.IterableWrapper:
        raise NotImplementedError

    def feather2data(self):
        datapipe = dp.iter.FileLister(os.path.join(self.root, f"{self.__class__.__name__}2feather", self.mode))
        if self.mode == 'train':
            datapipe.shuffle()
        for file_ in datapipe:
            yield self.read_feather(file_)

    def raw2feather(self):
        getLogger().info(f"[{self.__class__.__name__}] >>> Convert raw data ({self.mode}) to chunks in feather format")
        datapipe = self.raw2data()
        columns = [field.name for field in self.fields]
        count = 0
        for batch in datapipe.batch(batch_size=_DEFAULT_CHUNK_SIZE):
            df = pd.DataFrame(batch, columns=columns)
            for field in self.fields:
                df[field.name] = field.transform(df[field.name].values[:, None])
            self.write_feather(df, count)
            count += 1
        getLogger().info(f"[{self.__class__.__name__}] >>> {count} chunks done")

    def row_processer(self, row):
        return [field.caster(val) for val, field in zip(row, self.fields)]

    @timemeter("DataSet/compile")
    def compile(self):
        # prepare transformer
        columns = [field.name for field in self._cfg.fields]
        self.train()
        datapipe = self.raw2data().batch(batch_size=_DEFAULT_CHUNK_SIZE)
        for batch in datapipe:
            df = pd.DataFrame(batch, columns=columns)
            for field in self._cfg.fields:
                field.partial_fit(df[field.name].values[:, None])

        self.valid()
        datapipe = self.raw2data().batch(batch_size=_DEFAULT_CHUNK_SIZE)
        for batch in datapipe:
            df = pd.DataFrame(batch, columns=columns)
            for field in self._cfg.fields:
                field.partial_fit(df[field.name].values[:, None])

        # raw2feather
        self.train()
        if not self.check_feather():
            self.raw2feather()
        self.valid()
        if not self.check_feather():
            self.raw2feather()
        self.test()
        if not self.check_feather():
            self.raw2feather()
        self.train()

        getLogger().info(str(self))

    def __iter__(self) -> Iterator:
        yield from self.feather2data()

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.cfg.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg


class Postprocessor(BaseSet):

    def __init__(self, datapipe: Union[RecDataSet, 'Postprocessor'], training_only: bool = False) -> None:
        super().__init__()
        self.source = datapipe
        self.training_only = training_only
        self.fields: List[Field] = self.source.fields.copy()

    def train(self):
        super().train()
        self.source.train()

    def valid(self):
        super().valid()
        self.source.valid()

    def test(self):
        super().test()
        self.source.test()

    def process(self) -> Iterator:
        raise NotImplementedError()

    def __iter__(self) -> Iterator:
        if self.training_only and self.mode != 'train':
            yield from self.source
        else:
            yield from self.process()

@dp.functional_datapipe("shard_")
class Sharder(Postprocessor):

    def process(self) -> Iterator:
        worker_infos = torch.utils.data.get_worker_info()
        if worker_infos:
            id_, nums = worker_infos.id, worker_infos.num_workers
            for idx, item in enumerate(self.source):
                if idx % nums == id_:
                    yield item
        else:
            yield from self.source


@dp.functional_datapipe("pin_")
class PinMemory(Postprocessor):

    def __init__(
        self, datapipe: RecDataSet,
        buffer_size: Optional[int] = None, shuffle: bool = True,
        training_only: bool = False
    ) -> None:
        super().__init__(datapipe, training_only)

        self.buffer_size = buffer_size if buffer_size else float('inf')
        self.shuffle = shuffle

    def process(self) -> Iterator:
        _buffer = None
        for df in self.source:
            _buffer = pd.concat((_buffer, df))
            if len(_buffer) >= self.buffer_size:
                if self.mode == 'train' and self.shuffle:
                    yield _buffer.sample(frac=1)
                else:
                    yield _buffer
                _buffer = None

        if _buffer is not None and not _buffer.empty:
            if self.mode == 'train' and self.shuffle:
                yield _buffer.sample(frac=1)
            else:
                yield _buffer


@dp.functional_datapipe("subfield_")
class SubFielder(Postprocessor):
    """ Select subfields """

    def __init__(
        self, datapipe: Postprocessor,
        fields: Optional[Iterable[str]] = None,
        training_only: bool = False
    ) -> None:
        super().__init__(datapipe, training_only)
        if fields:
            self.fields = [field for field in self.fields if field.name in fields]
        self.columns = [field.name for field in self.fields]

    def process(self) -> Iterator:
        for df in self.source:
            yield df[self.columns]


@dp.functional_datapipe("chunk_")
class Chunker(Postprocessor):
    """A special batcher for dataframe only"""

    def __init__(
        self, datapipe: Postprocessor, 
        batch_size: int, drop_last: bool = False,
        training_only: bool = False
    ) -> None:
        super().__init__(datapipe, training_only)

        self.batch_size = batch_size
        self.drop_last = drop_last

    def process(self) -> Iterator:
        _buffer = None
        for df in self.source:
            _buffer = pd.concat((_buffer, df))
            for _ in range(self.batch_size, len(_buffer) + 1, self.batch_size):
                yield _buffer.head(self.batch_size)
                _buffer = _buffer[self.batch_size:]

        if not self.drop_last and not _buffer.empty:
            yield _buffer


@dp.functional_datapipe("dict_")
class Frame2Dict(Postprocessor):
    """Convert dataframe into Dict[str, List] """
    def process(self) -> Iterator:
        for df in self.source:
            yield {name: df[name].values.tolist() for name in df.columns}


@dp.functional_datapipe("list_")
class Frame2List(Postprocessor):
    """Convert dataframe into List[List] """
    def process(self) -> Iterator:
        for df in self.source:
            yield [df[field.name].values.tolist() for field in self.fields]


@dp.functional_datapipe('tensor_')
class ToTensor(Postprocessor):
    """Convert Dict[str, List] into Dict[str, torch.Tensor]"""
    def at_least_2d(self, val: torch.Tensor):
        return val.unsqueeze(1) if val.ndim == 1 else val

    def proess(self) -> Iterator:
        for batch in self.source:
            yield {field.name: self.at_least_2d(torch.tensor(batch[field.name], dtype=field.dtype)) for field in self.fields}


@dp.functional_datapipe("sample_negative_")
class NegativeSamper(Postprocessor):

    def __init__(
        self, datapipe: Postprocessor, 
        num_negatives: int = 1, pool_size: int = 99,
        training_only: bool = False
    ) -> None:
        super().__init__(datapipe, training_only)
        """
        num_negatives: for training, sampling from pool
        pool_size: for evaluation
        """

        self.num_negatives = num_negatives
        self.pool_size = pool_size
        self._User: SparseField = next(filter(lambda field: field.match([USER, ID]), self.fields))
        self._Item: SparseField = next(filter(lambda field: field.match([ITEM, ID]), self.fields))
        self.parseItems()

    @timemeter("NegativeSampler/parseItems")
    def parseItems(self):
        self.train()
        self.posItems = [set() for _ in range(self._User.count)]
        self.allItems = set(range(self._Item.count))
        self.negItems = []
        for df in self.source:
            df = df[[self._User.name, self._Item.name]]
            for idx, items in df.groupby(self._User.name).agg(set).iterrows():
                self.posItems[idx] |= set(*items)
        for items in self.posItems:
            negItems = list(self.allItems - items)
            k = self.pool_size if self.pool_size <= len(negItems) else len(negItems)
            self.negItems.append(random.sample(negItems, k = k))

    def process(self) -> Iterator:
        if self.mode == 'train':
            for df in self.source:
                negs = np.stack(
                    df.agg(
                        lambda row: random.sample(self.negItems[int(row[self._User.name])], k=self.num_negatives),
                        axis=1
                    ),
                    axis=0
                )
                df[self._Item.name] = np.concatenate((df[self._Item.name].values[:, None], negs), axis=1).tolist()
                yield df
        else:
            for df in self.source:
                negs = np.stack(
                    df.agg(
                        lambda row: self.negItems[int(row[self._User.name])],
                        axis=1
                    ),
                    axis=0
                )
                df[self._Item.name] = np.concatenate((df[self._Item.name].values[:, None], negs), axis=1).tolist()
                yield df
            

@dp.functional_datapipe('group_')
class Grouper(Postprocessor):
    """Group batch into several groups
    For example, RS generally requires
        for users, items, targets in datapipe: ...
    Note that we assume the last group is TARGET, which should be returned in List form.
    So the final returns are in the form of:
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], ..., List[torch.Tensor]
    """
    def __init__(
        self, datapipe: Postprocessor, 
        groups: Iterable[Union[Tag, Iterable[Tag]]] = (USER, ITEM, TARGET),
        training_only: bool = False
    ) -> None:
        super().__init__(datapipe, training_only)
        assert groups[-1] == TARGET, " ... "
        self.groups = [[field for field in self.fields if field.match(tags)] for tags in groups]
        self.target = self.groups.pop()[0]

    def __iter__(self) -> List:
        for batch in self.source:
            yield [*[{field.name: batch[field.name] for field in group} for group in self.groups], batch[self.target.name]]

