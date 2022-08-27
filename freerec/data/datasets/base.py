

from typing import Iterator

import torchdata.datapipes as dp
import pandas as pd
import feather
import os

from ...utils import timemeter, infoLogger


__all__ = ['BaseSet', 'RecDataSet']


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
        infoLogger(f"[{self.__class__.__name__}] >>> Convert raw data ({self.mode}) to chunks in feather format")
        datapipe = self.raw2data()
        columns = [field.name for field in self.fields]
        count = 0
        for batch in datapipe.batch(batch_size=_DEFAULT_CHUNK_SIZE):
            df = pd.DataFrame(batch, columns=columns)
            for field in self.fields:
                df[field.name] = field.transform(df[field.name].values[:, None])
            self.write_feather(df, count)
            count += 1
        infoLogger(f"[{self.__class__.__name__}] >>> {count} chunks done")

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

        infoLogger(str(self))

    def __iter__(self) -> Iterator:
        yield from self.feather2data()

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.cfg.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg
