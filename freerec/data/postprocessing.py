

from typing import Callable, Iterable, Iterator, Optional, List, Union

import torch
import torchdata.datapipes as dp
import numpy as np
import pandas as pd
import random

from .datasets import BaseSet, RecDataSet
from .fields import SparseField
from .tags import Tag, FEATURE, TARGET, USER, ITEM, ID
from ..utils import timemeter, warnLogger



class Postprocessor(BaseSet):

    def __init__(self, datapipe: Union[RecDataSet, 'Postprocessor']) -> None:
        super().__init__()
        self.source = datapipe
        self.fields = self.source.fields

    def train(self):
        super().train()
        self.source.train()

    def valid(self):
        super().valid()
        self.source.valid()

    def test(self):
        super().test()
        self.source.test()


@dp.functional_datapipe("dataframe_")
class DataFrame(Postprocessor):

    def __init__(
        self, datapipe: Union[RecDataSet, Postprocessor],
        columns: Optional[Iterable] = None
    ) -> None:
        super().__init__(datapipe)
        self.columns = columns

    def __iter__(self) -> Iterator:
        for batch in self.source:
            yield pd.DataFrame(batch, columns=self.columns)



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


@dp.functional_datapipe("pin_")
class PinMemory(Postprocessor):

    def __init__(
        self, datapipe: RecDataSet,
        buffer_size: Optional[int] = None, shuffle: bool = True
    ) -> None:
        super().__init__(datapipe)

        self.buffer_size = buffer_size if buffer_size else float('inf')
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
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
        filter_: Optional[Callable] = None
    ) -> None:
        super().__init__(datapipe)
        if filter_:
            self.fields = [field for field in self.fields if filter_(field)]
        self.columns = [field.name for field in self.fields]

    def __iter__(self) -> Iterator:
        for df in self.source:
            yield df[self.columns]


@dp.functional_datapipe("chunk_")
class Chunker(Postprocessor):
    """A special batcher for dataframe only"""

    def __init__(
        self, datapipe: Postprocessor, 
        batch_size: int, drop_last: bool = False
    ) -> None:
        super().__init__(datapipe)

        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator:
        _buffer = None
        for df in self.source:
            _buffer = pd.concat((_buffer, df))
            for _ in range(self.batch_size, len(_buffer) + 1, self.batch_size):
                yield _buffer.head(self.batch_size)
                _buffer = _buffer[self.batch_size:]

        if not self.drop_last and _buffer is not None and not _buffer.empty:
            yield _buffer


@dp.functional_datapipe("dict_")
class Frame2Dict(Postprocessor):
    """Convert dataframe into Dict[str, List] """
    def __iter__(self) -> Iterator:
        for df in self.source:
            yield {name: df[name].values.tolist() for name in df.columns}


@dp.functional_datapipe("list_")
class Frame2list(Postprocessor):
    """Convert dataframe into List[List] """
    def __iter__(self) -> Iterator:
        for df in self.source:
            yield [df[name].values.tolist() for name in df.columns]


@dp.functional_datapipe("tensor_")
class ToTensor(Postprocessor):
    """Convert Dict[str, List] into Dict[str, torch.Tensor]"""
    def at_least_2d(self, val: torch.Tensor):
        return val.unsqueeze(1) if val.ndim == 1 else val

    def __iter__(self) -> Iterator:
        for batch in self.source:
            yield {field.name: self.at_least_2d(torch.tensor(batch[field.name], dtype=field.dtype)) for field in self.fields}


@dp.functional_datapipe("sample_negative_")
class NegativeSamper(Postprocessor):

    def __init__(
        self, datapipe: Postprocessor, 
        num_negatives: int = 1, pool_size: int = 99
    ) -> None:
        super().__init__(datapipe)
        """
        num_negatives: for training, sampling from pool
        pool_size: for evaluation
        """

        self.num_negatives = num_negatives
        self.pool_size = pool_size
        self.User: SparseField = self.fields.whichis(USER, ID)
        self.Item: SparseField = self.fields.whichis(ITEM, ID)
        self.parseItems()

    @timemeter("NegativeSampler/parseItems")
    def parseItems(self):
        self.train()
        self.posItems = [set() for _ in range(self.User.count)]
        self.allItems = set(range(self.Item.count))
        self.negItems = []
        for df in self.source:
            df = df[[self.User.name, self.Item.name]]
            for idx, items in df.groupby(self.User.name).agg(set).iterrows():
                self.posItems[idx] |= set(*items)
        for items in self.posItems:
            negItems = list(self.allItems - items)
            k = self.pool_size if self.pool_size <= len(negItems) else len(negItems)
            self.negItems.append(random.sample(negItems, k = k))

    def __iter__(self) -> Iterator:
        if self.mode == 'train':
            for df in self.source:
                negs = np.stack(
                    df.agg(
                        lambda row: random.sample(self.negItems[int(row[self.User.name])], k=self.num_negatives),
                        axis=1
                    ),
                    axis=0
                )
                df[self.Item.name] = np.concatenate((df[self.Item.name].values[:, None], negs), axis=1).tolist()
                yield df
        else:
            for df in self.source:
                negs = np.stack(
                    df.agg(
                        lambda row: self.negItems[int(row[self.User.name])], # TODO: negItems including positives of testset
                        axis=1
                    ),
                    axis=0
                )
                df[self.Item.name] = np.concatenate((df[self.Item.name].values[:, None], negs), axis=1).tolist()
                yield df
            

@dp.functional_datapipe("group_")
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
        groups: Iterable[Union[Tag, Iterable[Tag]]] = (USER, ITEM, TARGET)
    ) -> None:
        super().__init__(datapipe)
        self.groups = [[field for field in self.fields if field.match(tags)] for tags in groups]

    def __iter__(self) -> List:
        for batch in self.source:
            yield [{field.name: batch[field.name] for field in group} for group in self.groups]


@dp.functional_datapipe("wrap_")
class Wrapper(Postprocessor):

    def __init__(
        self, datapipe: Postprocessor,
        validpipe: Optional[Postprocessor] = None,
        testpipe: Optional[Postprocessor] = None,
    ) -> None:
        """
        Args:
            datapipe: trainpipe
        Kwargs:
            validpipe: validpipe <- trainpipe if validpipe is None
            testpipe: testpipe <- validpipe if testpipe is None
        """
        super().__init__(datapipe)
        self.validpipe = datapipe if validpipe is None else validpipe
        self.testpipe = self.validpipe if testpipe is None else testpipe
        self.train()

    def train(self):
        super().train()
        self.source.train()

    def valid(self):
        super().valid()
        self.validpipe.valid()

    def test(self):
        super().test()
        self.testpipe.test()

    def __iter__(self) -> Iterator:
        if self.mode == 'train':
            yield from self.source
        elif self.mode == 'valid':
            yield from self.validpipe
        else:
            yield from self.testpipe

# rename

@dp.functional_datapipe("batch_")
class Batcher(Postprocessor):

    def __init__(
        self, datapipe: Union[RecDataSet, Postprocessor],
        batch_size: int, drop_last: bool = False
    ) -> None:
        super().__init__(datapipe)
        self.datapipe = datapipe.batch(batch_size, drop_last)

    def __iter__(self) -> Iterator:
        yield from self.datapipe


@dp.functional_datapipe("map_")
class Mapper(Postprocessor):

    def __init__(
        self, datapipe: Union[RecDataSet, Postprocessor], fn: Callable
    ) -> None:
        super().__init__(datapipe)
        self.datapipe = datapipe.map(fn)
    
    def __iter__(self) -> Iterator:
        yield from self.datapipe
