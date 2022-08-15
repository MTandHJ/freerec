


from typing import Dict, Iterable, Iterator, Optional, Tuple, List, Union

import torch
import torchdata.datapipes as dp
import numpy as np
import pandas as pd
import random

from ..fields import Field, SparseField
from ..tags import Tag, FEATURE, TARGET, USER, ITEM, ID
from ...utils import timemeter, warnLogger, getLogger


__all__ = ['RecDataSet', 'Postprocessor']


_DEFAULT_BATCH_SIZE = 10000


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

    def row_processer(self, row):
        return [field.caster(val) for val, field in zip(row, self.fields)]

    @timemeter("DataSet/compile")
    def compile(self):
        self.train()
        self._cfg.datasize = 0
        columns = [field.name for field in self._cfg.fields]
        datapipe = self.batch(batch_size=_DEFAULT_BATCH_SIZE)
        for batch in datapipe:
            df = pd.DataFrame(batch, columns=columns)
            for field in self._cfg.fields:
                field.partial_fit(df[field.name].values[:, None])
            self._cfg.datasize += len(df)

        self.valid()
        datapipe = self.batch(batch_size=_DEFAULT_BATCH_SIZE)
        for batch in datapipe:
            df = pd.DataFrame(batch, columns=columns)
            for field in self._cfg.fields:
                field.partial_fit(df[field.name].values[:, None])
            self._cfg.datasize += len(df)

        getLogger().info(str(self))


    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.cfg.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg



class Postprocessor(BaseSet):

    def __init__(self, datapipe: Union[RecDataSet, 'Postprocessor']) -> None:
        super().__init__()
        self.source = datapipe
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


@dp.functional_datapipe("shard_")
class Sharder(Postprocessor):

    def __iter__(self) -> Iterator:
        worker_infos = torch.utils.data.get_worker_info()
        if worker_infos:
            id_, nums = worker_infos.id, worker_infos.num_workers
            for idx, item in enumerate(self.source):
                if idx % nums == id_:
                    yield item
        else:
            yield from self.source


@dp.functional_datapipe("dataframe_")
class FrameMaker(Postprocessor):
    """Make pandas.DataFrame
    TODO: use torcharrow instead for efficient implements
    Params:
        buffer_size: int, save dataframe in memory, default: 6400 = 128 * 50;
        shuffle: bool, shuffle by row in True and in mode 'train'
    Return (__iter__):
        pandas.DataFrame
    """

    def __init__(self, datapipe: RecDataSet, buffer_size: int = 6400, shuffle: bool = True) -> None:
        super().__init__(datapipe)

        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.columns = [field.name for field in self.fields]

    def __iter__(self) -> Iterator:
        datapipe = self.source
        for batch in datapipe.batch(self.buffer_size):
            df = pd.DataFrame(batch, columns=self.columns)
            if self.mode == 'train' and self.shuffle:
                yield df.sample(frac=1)
            else:
                yield df


@dp.functional_datapipe("subfield_")
class SubFielder(Postprocessor):
    """ Select subfields """

    def __init__(self, datapipe: Postprocessor, fields: Optional[Iterable[str]] = None) -> None:
        super().__init__(datapipe)
        if fields:
            self.fields = [field for field in self.fields if field.name in fields]
        self.columns = [field.name for field in self.fields]

    def __iter__(self) -> Iterator:
        for df in self.source:
            yield df[self.columns]


@dp.functional_datapipe("encode_")
class Encoder(Postprocessor):
    """Transform int|float into required formats by column"""

    def __iter__(self) -> Iterator:
        for df in self.source:
            for field in self.fields:
                df[field.name] = field.transform(df[field.name].values[:, None])
            yield df


@dp.functional_datapipe("chunk_")
class Chunker(Postprocessor):
    """A special batcher for dataframe only"""

    def __init__(self, datapipe: Postprocessor, batch_size: int) -> None:
        super().__init__(datapipe)

        self.batch_size = batch_size
        self._buffer = pd.DataFrame(
            columns=[field.name for field in self.fields]
        )

    def __iter__(self) -> Iterator:
        for df in self.source:
            self._buffer = pd.concat((self._buffer, df))
            for _ in range(self.batch_size, len(self._buffer) + 1, self.batch_size):
                yield self._buffer.head(self.batch_size)
                self._buffer = self._buffer[self.batch_size:]
        if not self._buffer.empty:
            yield self._buffer
            self._buffer.drop(self._buffer.index, inplace=True)


@dp.functional_datapipe("dict_")
class Frame2Dict(Postprocessor):
    """Convert dataframe into Dict[str, List] """
    def __iter__(self) -> Iterator:
        for df in self.source:
            yield {name: df[name].values.tolist() for name in df.columns}


@dp.functional_datapipe("list_")
class Frame2List(Postprocessor):
    """Convert dataframe into List[List] """
    def __iter__(self) -> Iterator:
        for df in self.source:
            yield [df[field.name].values.tolist() for field in self.fields]


@dp.functional_datapipe('tensor_')
class ToTensor(Postprocessor):
    """Convert Dict[str, List] into Dict[str, torch.Tensor]"""
    def at_least_2d(self, val: torch.Tensor):
        return val.unsqueeze(1) if val.ndim == 1 else val

    def __iter__(self) -> Iterator:
        for batch in self.source:
            yield {field.name: self.at_least_2d(torch.tensor(batch[field.name], dtype=field.dtype)) for field in self.fields}


@dp.functional_datapipe("sample_negative_")
class NegativeSamper(Postprocessor):

    def __init__(self, datapipe: Postprocessor, num_negatives: int = 1, pool_size: int = 99) -> None:
        super().__init__(datapipe)
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

    def __iter__(self) -> Iterator:
        if self.mode == 'train':
            for df in self.source:
                # df[self._Item.name] = df.agg(
                #     lambda row: [row[self._Item.name]] \
                #         + random.sample(self.negItems[int(row[self._User.name])], k=self.num_negatives), 
                #     axis=1
                # )
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
                # df[self._Item.name] = df.agg(
                #     lambda row: [row[self._Item.name]] \
                #         + self.negItems[int(row[self._User.name])], 
                #     axis=1
                # ) 
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
        self, datapipe: Postprocessor, groups: Iterable[Union[Tag, Iterable[Tag]]] = (USER, ITEM, TARGET)
    ) -> None:
        super().__init__(datapipe)
        assert groups[-1] == TARGET, " ... "
        self.groups = [[field for field in self.fields if field.match(tags)] for tags in groups]
        self.target = self.groups.pop()[0]

    def __iter__(self) -> List:
        for batch in self.source:
            yield [*[{field.name: batch[field.name] for field in group} for group in self.groups], batch[self.target.name]]

