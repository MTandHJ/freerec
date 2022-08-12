


import chunk
from dataclasses import field
from typing import Dict, Iterable, Iterator, Optional, Tuple, List, Union

import torch
import torchdata.datapipes as dp
import pandas as pd

from ..fields import Field
from ..tags import Tag, FEATURE, TARGET, USER, ITEM, ID
from ...utils import warnLogger, getLogger


__all__ = ['RecDataSet', 'Encoder']


_DEFAULT_BATCH_SIZE = 1000

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
        assert hasattr(cls._cfg, 'fields'), "Fields sorted by column should be given in _cfg ..."
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
    def fields(self):
        return self._cfg.fields

    @property
    def active(self):
        return self._active

    def row_processer(self, row):
        return [field.caster(val) for val, field in zip(row, self.fields)]

    @classmethod
    def compile(cls, datapipe: dp.iter.IterDataPipe):
        if cls._active:
            warnLogger(
                f"Dataset {cls.__name__} has been activated !!! Skip it ..."
            )
        cls._active = True
        datapipe = datapipe.batch(batch_size=_DEFAULT_BATCH_SIZE)
        cls._cfg.datasize = 0
        columns = [field.name for field in cls._cfg.fields]
        for batch in datapipe:
            df = pd.DataFrame(batch, columns=columns)
            for field in cls._cfg.fields:
                field.partial_fit(df[[field.name]])
            cls._cfg.datasize += len(df)


    def _compile(self, refer: str = 'train'):
        if not self.active:
            split, self.split = self.split, refer
            self.compile(self) # compile according to the trainset !
            self.split = split
        getLogger().info(str(self))

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.cfg.fields))
        return f"{self.__class__.__name__} >>> \n" + cfg



class Postprocessor(dp.iter.IterDataPipe):

    def __init__(self, datapipe: Union[RecDataSet, 'Postprocessor']) -> None:
        super().__init__()
        self.source = datapipe
        self.fields: List[Field] = self.source.fields


@dp.functional_datapipe("shard")
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


@dp.functional_datapipe("frame")
class FrameMaker(Postprocessor):

    def __init__(self, datapipe: RecDataSet, buffer_size: int = 2560, shuffle: bool = True) -> None:
        super().__init__(datapipe)

        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.columns = [field.name for field in self.fields]

    def __iter__(self) -> Iterator:
        datapipe = self.source
        for batch in datapipe.batch(self.buffer_size):
            df = pd.DataFrame(batch, columns=self.columns)
            if self.shuffle:
                yield df.sample(frac=1)
            else:
                yield df


@dp.functional_datapipe("subfield")
class SubFielder(Postprocessor):

    def __init__(self, datapipe: Postprocessor, fields: Optional[Iterable[str]] = None) -> None:
        super().__init__(datapipe)
        if fields:
            self.fields = [field for field in self.fields if field.name in fields]
        self.columns = [field.name for field in self.fields]

    def __iter__(self) -> Iterator:
        for df in self.source:
            yield df[self.columns]


@dp.functional_datapipe("encode")
class Encoder(Postprocessor):

    def __init__(self, datapipe: Postprocessor) -> None:
        super().__init__(datapipe)

        self.dtypes = {field.name: field.dtype for field in self.fields}

    def __iter__(self) -> Iterator:
        for df in self.source:
            for field in self.fields:
                df[field.name] = field.transform(df[[field.name]])
            yield df.astype(self.dtypes)


@dp.functional_datapipe("chunk")
class Chunker(Postprocessor):

    def __init__(self, datapipe: Postprocessor, batch_size: int) -> None:
        super().__init__(datapipe)

        self.batch_size = batch_size
        self._buffer = pd.DataFrame(
            columns=[field.name for field in self.fields]
        ).astype({field.name: field.dtype for field in self.fields})


    def __iter__(self) -> Iterator:
        for df in self.source:
            self._buffer = pd.concat((self._buffer, df))
            for _ in range(self.batch_size, len(self._buffer) + 1, self.batch_size):
                yield self._buffer.head(self.batch_size)
                self._buffer = self._buffer[self.batch_size:]
        yield self._buffer
        self._buffer.drop(self._buffer.index, inplace=True)


@dp.functional_datapipe("dict")
class Frame2Dict(Postprocessor):

    def __iter__(self) -> Iterator:
        for df in self.source:
            yield {field.name: df[field.name].values for field in self.fields}


@dp.functional_datapipe("list")
class Frame2List(Postprocessor):

    def __iter__(self) -> Iterator:
        for df in self.source:
            yield [df[field.name].values for field in self.fields]


@dp.functional_datapipe('tensor')
class ToTensor(Postprocessor):

    def __init__(self, datapipe: Union[RecDataSet, 'Postprocessor']) -> None:
        super().__init__(datapipe)
        if type(self.source) == Frame2Dict:
            self.__func = self.dict2tensor
        elif type(self.source) == Frame2List:
            self.__func = self.list2tensor
        else:
            raise TypeError("datapipe should be Frame2Dict or Frame2List ...")

    def dict2tensor(self, batch):
        return {field.name: torch.from_numpy(batch[field.name]).view(-1, 1) for field in self.fields}

    def list2tensor(self, batch):
        return [torch.from_numpy(item).view(-1, 1) for item in batch]

    def __iter__(self) -> Iterator:
        for batch in self.source:
            yield self.__func(batch)

@dp.functional_datapipe('group')
class Grouper(Postprocessor):

    def __init__(
        self, datapipe: Postprocessor, groups: Iterable[Union[Tag, Iterable[Tag]]] = (USER, ITEM, TARGET)
    ) -> None:
        super().__init__(datapipe)
        assert groups[-1] == TARGET, " ... "
        self.groups = [[field for field in self.fields if field.match(tags)] for tags in groups]
        self.target = self.groups.pop()[0]

    def __iter__(self) -> List[Dict]:
        for batch in self.source:
            yield [*[{field.name: batch[field.name] for field in group} for group in self.groups], batch[self.target.name]]


class Grapher(Postprocessor):

    def __init__(self, datapipe: RecDataSet) -> None:
        super().__init__(datapipe)

        self.User: Field = [field for field in self.fields.match([USER, ID])][0]
        self.Item: Field = [field for field in self.fields.match([ITEM, ID])][0]
        self.Rating: Field = [field for field in self.fields.match([TARGET])][0]
        self.fields = [self.User, self.Item, self.Rating]

    def compile(self):
        self.posItems = [[] for _ in range(self.User.count)]
        for row in self.source:
            users = row[self.User.name]
            items = row[self.Item.name]
            ratings = row[self.Rating.name]
