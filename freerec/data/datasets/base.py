

from typing import Iterator

import torchdata.datapipes as dp
import pandas as pd
import feather, os
from freeplot.utils import import_pickle, export_pickle

from ..fields import Field, Fielder
from ...utils import timemeter, infoLogger, errorLogger, mkdirs


__all__ = ['BaseSet', 'RecDataSet']


_DEFAULT_FEATHER_FMT = "{0}2feather"
_DEFAULT_TRANSFORM_FILENAME = "transforms.pickle"
_DEFAULT_CHUNK_FMT = "chunk{0}.feather"
_DEFAULT_CHUNK_SIZE = 51200


class BaseSet(dp.iter.IterDataPipe):

    def __init__(self) -> None:
        super().__init__()

        self.__mode = 'train'

    @property
    def fields(self):
        return self.__fields

    @fields.setter
    def fields(self, vals) -> Fielder:
        self.__fields = Fielder(vals)
        
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
                errorLogger("_cfg, should be defined before instantiation ...", AttributeError)
        if not hasattr(cls._cfg, 'fields'):
            errorLogger("Fields sorted by column should be given in _cfg ...", AssertionError)
        return super().__new__(cls)

    def __init__(self, root: str) -> None:
        """
        root: data file
        """
        super().__init__()
        self.root = root
        self.fields = self._cfg.fields

        if not os.path.exists(self.root) or not any(True for _ in os.scandir(self.root)):
            errorLogger(
                f"No such root of {self.root} or this dir is empty ...",
                FileNotFoundError
            )

    @property
    def cfg(self):
        return self._cfg

    def check_transforms(self):
        file_ = os.path.join(
            self.root, 
            _DEFAULT_FEATHER_FMT.format(self.__class__.__name__), 
            _DEFAULT_TRANSFORM_FILENAME
        )
        if os.path.isfile(file_):
            return True
        else:
            mkdirs(os.path.join(
                self.root,
                _DEFAULT_FEATHER_FMT.format(self.__class__.__name__)
            ))
            return False

    def save_transforms(self):
        infoLogger(f"[{self.__class__.__name__}] >>> Save transformers ...")
        state_dict = self.fields.state_dict()
        file_ = os.path.join(
            self.root, 
            _DEFAULT_FEATHER_FMT.format(self.__class__.__name__), 
            _DEFAULT_TRANSFORM_FILENAME
        )
        export_pickle(state_dict, file_)

    def load_transforms(self):
        file_ = os.path.join(
            self.root, 
            _DEFAULT_FEATHER_FMT.format(self.__class__.__name__), 
            _DEFAULT_TRANSFORM_FILENAME
        )
        self.fields.load_state_dict(import_pickle(file_), strict=False)

    def check_feather(self):
        path = os.path.join(
            self.root, 
            _DEFAULT_FEATHER_FMT.format(self.__class__.__name__), 
            self.mode
        )
        if os.path.exists(path):
            return any(True for _ in os.scandir(path))
        else:
            os.makedirs(path)
            return False

    def write_feather(self, dataframe: pd.DataFrame, count: int):
        file_ = os.path.join(
            self.root, 
            _DEFAULT_FEATHER_FMT.format(self.__class__.__name__),
            self.mode, _DEFAULT_CHUNK_FMT.format(count)
        )
        feather.write_dataframe(dataframe, file_)

    def read_feather(self, file_: str):
        return feather.read_dataframe(file_)

    def raw2data(self) -> dp.iter.IterableWrapper:
        errorLogger(
            "raw2data method should be specified ...",
            NotImplementedError
        )

    def feather2data(self):
        datapipe = dp.iter.FileLister(
            os.path.join(
                self.root, 
                _DEFAULT_FEATHER_FMT.format(self.__class__.__name__),
                self.mode
            )
        )
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
        if self.check_transforms():
            self.load_transforms()
        else:
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

            self.save_transforms()
            
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
