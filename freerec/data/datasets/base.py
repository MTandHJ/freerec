

from typing import Iterator, List, Dict

import torchdata.datapipes as dp
import os
from freeplot.utils import import_pickle, export_pickle

from freerec.data.tags import SPARSE

from ..fields import Field, Fielder
from ..utils import collate_dict
from ...utils import timemeter, infoLogger, errorLogger, mkdirs


__all__ = ['BaseSet', 'RecDataSet']


_DEFAULT_PICKLE_FMT = "{0}2pickle"
_DEFAULT_TRANSFORM_FILENAME = "transforms.pickle"
_DEFAULT_CHUNK_FMT = "chunk{0}.pickle"


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

    _DEFAULT_CHUNK_SIZE = 51200 # chunk size

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
        """Return the config of the dataset"""
        return self._cfg

    def check_transforms(self):
        """Check if the transformations exist."""
        file_ = os.path.join(
            self.root, 
            _DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            _DEFAULT_TRANSFORM_FILENAME
        )
        if os.path.isfile(file_):
            return True
        else:
            mkdirs(os.path.join(
                self.root,
                _DEFAULT_PICKLE_FMT.format(self.__class__.__name__)
            ))
            return False

    def save_transforms(self):
        infoLogger(f"[{self.__class__.__name__}] >>> Save transformers ...")
        state_dict = self.fields.state_dict()
        file_ = os.path.join(
            self.root, 
            _DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            _DEFAULT_TRANSFORM_FILENAME
        )
        export_pickle(state_dict, file_)

    def load_transforms(self):
        file_ = os.path.join(
            self.root, 
            _DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            _DEFAULT_TRANSFORM_FILENAME
        )
        self.fields.load_state_dict(import_pickle(file_), strict=False)

    def check_pickle(self):
        """Check if the dataset has been converted into feather format."""
        path = os.path.join(
            self.root, 
            _DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            self.mode
        )
        if os.path.exists(path):
            return any(True for _ in os.scandir(path))
        else:
            os.makedirs(path)
            return False

    def write_pickle(self, data: Dict, count: int):
        """Save pickle format data."""
        file_ = os.path.join(
            self.root, 
            _DEFAULT_PICKLE_FMT.format(self.__class__.__name__),
            self.mode, _DEFAULT_CHUNK_FMT.format(count)
        )
        export_pickle(data, file_)

    def read_pickle(self, file_: str):
        """Load pickle format data."""
        return import_pickle(file_)

    def raw2data(self) -> dp.iter.IterableWrapper:
        errorLogger(
            "raw2data method should be specified ...",
            NotImplementedError
        )

    def row_processer(self, row):
        """Row processer for raw data."""
        return {field.name: field.caster(val) for val, field in zip(row, self.fields)}

    def raw2pickle(self):
        """Convert raw data into pickle format."""
        infoLogger(f"[{self.__class__.__name__}] >>> Convert raw data ({self.mode}) to chunks in pickle format")
        datapipe = self.raw2data()
        count = 0
        for chunk in datapipe.batch(batch_size=self._DEFAULT_CHUNK_SIZE).collate(collate_dict):
            for field in self.fields:
                chunk[field.name] = field.transform(chunk[field.name][:, None]).ravel()[:, None] # N x 1
            self.write_pickle(chunk, count)
            count += 1
        infoLogger(f"[{self.__class__.__name__}] >>> {count} chunks done")

    def pickle2data(self):
        """Read pickle data in chunks."""
        datapipe = dp.iter.FileLister(
            os.path.join(
                self.root, 
                _DEFAULT_PICKLE_FMT.format(self.__class__.__name__),
                self.mode
            )
        )
        if self.mode == 'train':
            datapipe.shuffle()
        for file_ in datapipe:
            yield self.read_pickle(file_)

    @timemeter("DataSet/compile")
    def compile(self):
        """Check current dataset and transformations."""

        def fit_transform(fields):
            datapipe = self.raw2data().batch(batch_size=self._DEFAULT_CHUNK_SIZE).collate(collate_dict)
            for batch in datapipe:
                for field in fields:
                    field.partial_fit(batch[field.name][:, None])

        if self.check_transforms():
            self.load_transforms()
        else:
            self.train()
            fit_transform(self.fields)

            # avoid unseen tokens not included in trainset
            self.valid()
            fit_transform(self.fields.whichis(SPARSE))
            self.test()
            fit_transform(self.fields.whichis(SPARSE))

            self.save_transforms()
            
        # raw2pickle
        self.train()
        if not self.check_pickle():
            self.raw2pickle()
        self.valid()
        if not self.check_pickle():
            self.raw2pickle()
        self.test()
        if not self.check_pickle():
            self.raw2pickle()
        self.train()

        infoLogger(str(self))

    def __iter__(self) -> Iterator:
        yield from self.pickle2data()

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.cfg.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg
