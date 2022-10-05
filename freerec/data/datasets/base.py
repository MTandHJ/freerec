

from typing import Iterator, List, Dict, Union

import torchdata.datapipes as dp
import os
from freeplot.utils import import_pickle, export_pickle
from math import ceil


from ..tags import SPARSE
from ..fields import Field, Fielder, SparseField
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

    def __len__(self):
        raise NotImplementedError()


class RecDataSet(BaseSet):
    """ RecDataSet provides a template for specific datasets.

    Attributes:
    ---

    _cfg: Config[str, Field]
        Includes fields of each column.
    _DEFAULT_CHUNK_SIZE: int, defalut 51200
        Chunk size for saving.

    Notes:
    ---

    All datasets inherit RecDataSet should define the class variable of `_cfg` before instantiation.
    Generally speaking, the dataset will be splited into 
        - trainset
        - validset
        - testset

    Because these three datasets share the same _cfg, compiling any one of them will overwrite it ! 
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
        Parameters:
        ---

        root: Data path.
        """
        super().__init__()
        self.root = root
        self.fields = self._cfg.fields
        self.trainsize: int = 0
        self.validsize: int = 0
        self.testsize: int = 0

        if not os.path.exists(self.root) or not any(True for _ in os.scandir(self.root)):
            errorLogger(
                f"No such root of {self.root} or this dir is empty ...",
                FileNotFoundError
            )

    @property
    def cfg(self):
        """Return the config of the dataset."""
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
        state_dict['trainsize'] = self.trainsize
        state_dict['validsize'] = self.validsize
        state_dict['testsize'] = self.testsize
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
        state_dict = import_pickle(file_)
        self.trainsize = state_dict['trainsize']
        self.validsize = state_dict['validsize']
        self.testsize = state_dict['testsize']
        self.fields.load_state_dict(state_dict, strict=False)

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
        """Check current dataset and transformations.

        Flows:
        ---

        1. Check whether the transformation has been fitted:
            - `True`: Skip.
            - `False`: Fit the total trainset and the `SPARSE` fields in valid|testset
                to avoid unseen features. This operation will not cause information leakage.
            
        2. Convert each set into pickle format for fast loading.

        """

        def fit_transform(fields):
            datapipe = self.raw2data().batch(batch_size=self._DEFAULT_CHUNK_SIZE).collate(collate_dict)
            datasize = 0
            try:
                for batch in datapipe:
                    for field in fields:
                        field.partial_fit(batch[field.name][:, None])
                    datasize += len(batch[field.name])
            except NameError as e:
                errorLogger(e, NameError)
            return datasize

        if self.check_transforms():
            self.load_transforms()
        else:
            self.train()
            self.trainsize = fit_transform(self.fields)

            # avoid unseen tokens not included in trainset
            self.valid()
            self.validsize = fit_transform(self.fields.whichis(SPARSE))
            self.test()
            self.testsize = fit_transform(self.fields.whichis(SPARSE))

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

    @property
    def datasize(self):
        if self.mode == 'train':
            return self.trainsize
        elif self.mode == 'valid':
            return self.validsize
        else:
            return self.testsize

    def __len__(self):
        return ceil(self.datasize / self._DEFAULT_CHUNK_SIZE)

    def __iter__(self) -> Iterator:
        yield from self.pickle2data()

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.cfg.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg
