

from typing import Iterator, Optional, Tuple, Callable

import torch, os
import numpy as np
import torchdata.datapipes as dp
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected
from freeplot.utils import import_pickle, export_pickle

from ..tags import SPARSE, USER, ITEM, ID
from ..fields import Field, BufferField, SparseField, FieldList, FieldTuple
from ..utils import collate_list, download_from_url, extract_archive
from ...utils import timemeter, infoLogger, mkdirs, warnLogger
from ...dict2obj import Config


__all__ = ['BaseSet', 'RecDataSet']


DEFAULT_PICKLE_FMT = "{0}_from_pickle"
DEFAULT_TRANSFORM_FILENAME = "transforms.pickle"
DEFAULT_CHUNK_FMT = "chunk{0}.pickle"


class RecSetBuildingError(Exception): ...


class BaseSet(dp.iter.IterDataPipe):

    def __init__(self) -> None:
        super().__init__()

        self.__mode = 'train'
        
    @property
    def mode(self):
        return self.__mode

    def train(self):
        self.__mode = 'train'
        return self

    def valid(self):
        self.__mode = 'valid'
        return self

    def test(self):
        self.__mode = 'test'
        return self

    def listmap(self, func: Callable, *iterables):
        return list(map(func, *iterables))

    def __len__(self):
        raise NotImplementedError()

    @timemeter("DataSet/to_graph")
    def to_heterograph(self, *edge_types: Tuple[SparseField, Optional[str], SparseField]) -> HeteroData:
        """Convert datapipe to a heterograph.

        Parameters:
        ---

        *edge_types: (src, edge, dst)
            - src: SparseField
                Source node.
            - edge: Optional[str]
                The name of the edge. `src.name2dst.name` will be specified if `edge` is `None`.
            - dst: SparseField
                Destination node.

        Notes:
        ---

        Warning will be raised if current mode is not 'train' !

        Examples:
        ---

        >>> from freerec.data.datasets import Gowalla_m1
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> graph = basepipe.to_heterograph(
        ...    (fields[USER, ID], None, fields[ITEM, ID]), 
        ...    (fields[ITEM, ID], None, fields[USER, ID])
        ... )
        >>> graph
        HeteroData(
            UserID={ x=[29858, 0] },
            ItemID={ x=[40981, 0] },
            (UserID, UserID2ItemID, ItemID)={ edge_index=[2, 810128] },
            (ItemID, ItemID2UserID, UserID)={ edge_index=[2, 810128] }
        )
        """
        if self.mode != 'train':
            warnLogger(f"Convert the datapipe for {self.mode} to graph. Make sure that this is intentional ...")

        srcs, _, dsts = zip(*edge_types)
        edges = list(map(lambda triplet: triplet[1] if triplet[1] else f"{triplet[0].name}2{triplet[2].name}", edge_types))
        data = {node.name: [] for node in set(srcs + dsts)}
        for df in self:
            for node in data:
                data[node].append(np.ravel(df[node]))
        for key in data:
            data[key] = torch.tensor(np.concatenate(data[key], axis=0), dtype=torch.long)

        graph = HeteroData()
        for node in srcs:
            graph[node.name].x = torch.empty((node.count, 0), dtype=torch.long)
        for node in dsts:
            if node not in srcs:
                graph[node.name].x = torch.empty((node.count, 0), dtype=torch.long)
        for src, edge, dst in zip(srcs, edges, dsts):
            u, v = data[src.name], data[dst.name]
            graph[src.name, edge, dst.name].edge_index = torch.stack((u, v), dim=0) # 2 x N
        return graph.coalesce()

    def to_bigraph(
        self,
        src: SparseField,
        dst: SparseField,
        edge_type: Optional[str] = None
    ) -> HeteroData:
        """Convert datapipe to a bipartite graph.

        Parameters:
        ---

        src: SparseField
            Source node.
        dst: SparseField
            Destination node.
        edge_type: Optional[str]
            The name of the edge. `src.name2dst.name` will be specified if `edge` is `None`.

        Notes:
        ---

        Warning will be raised if current mode is not 'train' !

        Examples:
        ---

        >>> from freerec.data.datasets import Gowalla_m1
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> graph = basepipe.to_bigraph(
        ...    fields[USER, ID], fields[ITEM, ID]
        ... )
        >>> graph
        HeteroData(
            UserID={ x=[29858, 0] },
            ItemID={ x=[40981, 0] },
            (UserID, UserID2ItemID, ItemID)={ edge_index=[2, 810128] }
        )
        """
        return self.to_heterograph((src, edge_type, dst))
   
    def to_graph(self, src: SparseField, dst: SparseField) -> Data:
        """Convert datapipe to a homogeneous graph.

        Parameters:
        ---

        src: SparseField
            Source node.
        dst: SparseField
            Destination node.

        Notes:
        ---

        Warning will be raised if current mode is not 'train' !

        Examples:
        ---

        >>> from freerec.data.datasets import Gowalla_m1
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> graph = basepipe.to_bigraph(
        ...    fields[USER, ID], fields[ITEM, ID],
        ... )
        >>> graph
        Data(edge_index=[2, 1620256], x=[70839, 0], node_type=[70839], edge_type=[810128])
        """
        graph = self.to_heterograph((src, None, dst)).to_homogeneous()
        graph.edge_index = to_undirected(graph.edge_index)
        return graph

    def summary(self):
        infoLogger(str(self))


    def forward(self):
        raise NotImplementedError("_iter method should be implemented ...")

    def __iter__(self) -> Iterator[FieldList[BufferField]]:
        for chunk in self.forward():
            yield FieldList(map(lambda field: field.buffer(), chunk))



class RecDataSet(BaseSet):
    """ RecDataSet provides a template for specific datasets.

    Attributes:
    ---

    _cfg: Config[str, Field]
        Includes fields of each column.
    DEFAULT_CHUNK_SIZE: int, defalut 51200
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

    DEFAULT_CHUNK_SIZE = 51200 # chunk size
    URL: str
    VALID_IS_TEST: bool

    def __new__(cls, *args, **kwargs):
        for attr in ('_cfg', 'VALID_IS_TEST'):
            if not hasattr(cls, attr):
                raise RecSetBuildingError(f"'{attr}' should be defined before instantiation ...")
        assert 'fields' in cls._cfg, "the config of fields should be defined in '_cfg' ..."
        return super().__new__(cls)

    def __init__(self, root: str, filename: Optional[str] = None, download: bool = True) -> None:
        """
        Parameters:
        ---

        root: str
            The path storing datasets.
        filename: str, optional
            The dirname of the dataset.
            - `None`: Set the classname as the filename.
        
        download: bool
            Download the dataset from a URL.
        """
        super().__init__()
        filename = filename if filename else self.__class__.__name__
        self.path = os.path.join(root, filename)
        self.trainsize: int = 0
        self.validsize: int = 0
        self.testsize: int = 0

        fields = []
        for field_type, cfg in self._cfg['fields']:
            fields.append(field_type(**cfg))
        self.fields = fields

        if not os.path.exists(self.path) or not any(True for _ in os.scandir(self.path)):
            if download:
                extract_archive(
                    download_from_url(self.URL, root, overwrite=False),
                    self.path
                )
            else:
                FileNotFoundError(f"No such file of {self.path}, or this dir is empty ...")
        self.compile()

    @property
    def fields(self):
        return self.__fields

    @fields.setter
    def fields(self, vals) -> FieldTuple[Field]:
        self.__fields = FieldTuple(vals)

    def check_transforms(self):
        """Check if the transformations exist."""
        file_ = os.path.join(
            self.path,
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            DEFAULT_TRANSFORM_FILENAME
        )
        if os.path.isfile(file_):
            return True
        else:
            mkdirs(os.path.join(
                self.path,
                DEFAULT_PICKLE_FMT.format(self.__class__.__name__)
            ))
            return False

    def save_transforms(self):
        infoLogger(f"[{self.__class__.__name__}] >>> Save transformers ...")
        state_dict = self.fields.state_dict()
        state_dict['trainsize'] = self.trainsize
        state_dict['validsize'] = self.validsize
        state_dict['testsize'] = self.testsize
        file_ = os.path.join(
            self.path, 
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            DEFAULT_TRANSFORM_FILENAME
        )
        export_pickle(state_dict, file_)

    def load_transforms(self):
        file_ = os.path.join(
            self.path, 
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            DEFAULT_TRANSFORM_FILENAME
        )
        state_dict = import_pickle(file_)
        self.trainsize = state_dict['trainsize']
        self.validsize = state_dict['validsize']
        self.testsize = state_dict['testsize']
        self.fields.load_state_dict(state_dict, strict=False)

    def check_pickle(self):
        """Check if the dataset has been converted into feather format."""
        path = os.path.join(
            self.path,
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            self.mode
        )
        if os.path.exists(path):
            return any(True for _ in os.scandir(path))
        else:
            os.makedirs(path)
            return False

    def write_pickle(self, data, count: int):
        """Save pickle format data."""
        file_ = os.path.join(
            self.path,
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__),
            self.mode, DEFAULT_CHUNK_FMT.format(count)
        )
        export_pickle(data, file_)

    def read_pickle(self, file_: str):
        """Load pickle format data."""
        return import_pickle(file_)

    def row_processer(self, row):
        """Row processer for raw data."""
        return [field.caster(val) for val, field in zip(row, self.fields)]

    def raw2data(self) -> dp.iter.IterableWrapper:
        """Return `Tuple` by row !!!"""
        raise NotImplementedError("raw2data method should be implemented ...")

    def raw2pickle(self):
        """Convert raw data into pickle format."""
        infoLogger(f"[{self.__class__.__name__}] >>> Convert raw data ({self.mode}) to chunks in pickle format")
        datapipe = self.raw2data()
        count = 0
        for chunk in datapipe.batch(batch_size=self.DEFAULT_CHUNK_SIZE).collate(collate_list):
            for j, field in enumerate(self.fields):
                chunk[j] = field.transform(chunk[j])
            self.write_pickle(chunk, count)
            count += 1
        infoLogger(f"[{self.__class__.__name__}] >>> {count} chunks done")

    def pickle2data(self):
        """Read pickle data in chunks."""
        datapipe = dp.iter.FileLister(
            os.path.join(
                self.path,
                DEFAULT_PICKLE_FMT.format(self.__class__.__name__),
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

        def buffer(field: Field, col):
            return field.buffer(col)

        def fit_transform(*tags):
            datapipe = self.raw2data().batch(self.DEFAULT_CHUNK_SIZE).collate(collate_list)
            datasize = 0
            fields = self.fields.groupby(*tags)
            for chunk in datapipe:
                datasize += len(chunk[0])
                chunk = FieldList(map(buffer, self.fields, chunk)).groupby(*tags)
                for j, field in enumerate(fields):
                    field.partial_fit(chunk[j].data)
            return datasize

        if self.check_transforms():
            self.load_transforms()
        else:
            self.train()
            self.trainsize = fit_transform()

            # avoid unseen tokens not included in trainset
            self.valid()
            self.validsize = fit_transform(SPARSE)
            self.test()
            self.testsize = fit_transform(SPARSE)

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

        self.summary()

    @property
    def datasize(self):
        if self.mode == 'train':
            return self.trainsize
        elif self.mode == 'valid':
            return self.validsize
        else:
            return self.testsize

    def forward(self) -> Iterator:
        for chunk in self.pickle2data():
            yield [field.buffer(col) for field, col in zip(self.fields, chunk)]

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg


class _Row2Pairer(dp.iter.IterDataPipe):

    def __init__(self, datapipe: dp.iter.IterDataPipe) -> None:
        super().__init__()
        self.source = datapipe

    def __iter__(self):
        for row in self.source:
            user = row[0]
            for item in row[1:]:
                if item:
                    yield user, item


class ImplicitRecSet(RecDataSet):
    """Implicit feedback data.
    The data should be collected in the order of users; that is,
    each row represents a user's interacted items.
    """

    VALID_IS_TEST = True

    _cfg = Config(
        fields = [
            (SparseField, Config(name='UserID', na_value=None, dtype=int, tags=[USER, ID])),
            (SparseField, Config(name='ItemID', na_value=None, dtype=int, tags=[ITEM, ID]))
        ]
    )

    open_kw = Config(mode='rt', delimiter=' ', skip_lines=0)

    def file_filter(self, filename: str):
        if self.mode == 'train':
            return 'train' in filename
        else:
            return 'test' in filename

    def raw2data(self) -> dp.iter.IterableWrapper:
        datapipe = dp.iter.FileLister(self.path)
        datapipe = datapipe.filter(filter_fn=self.file_filter)
        datapipe = datapipe.open_files(mode=self.open_kw.mode)
        datapipe = datapipe.parse_csv(delimiter=self.open_kw.delimiter, skip_lines=self.open_kw.skip_lines)
        datapipe = _Row2Pairer(datapipe)
        datapipe = datapipe.map(self.row_processer)
        return datapipe

    def summary(self):
        super().summary()
        from prettytable import PrettyTable
        User, Item = self.fields[USER, ID], self.fields[ITEM, ID]

        table = PrettyTable(['#User', '#Item', '#Interactions', '#Train', '#Test', 'Density'])
        table.add_row([
            User.count, Item.count, self.trainsize + self.test().testsize,
            self.trainsize, self.testsize,
            (self.trainsize + self.testsize) / (User.count * Item.count)
        ])

        infoLogger(table)