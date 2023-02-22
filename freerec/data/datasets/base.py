

from typing import Iterator, Optional, TypeVar, Tuple

import torch, os
import numpy as np
import torchdata.datapipes as dp
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_undirected
from freeplot.utils import import_pickle, export_pickle

from ..tags import FieldTags, SPARSE, USER, ITEM, ID
from ..fields import Field, BufferField, SparseField, FieldList, FieldTuple
from ..utils import collate_list, download_from_url, extract_archive
from ...utils import timemeter, infoLogger, mkdirs, warnLogger
from ...dict2obj import Config


__all__ = ['BaseSet', 'RecDataSet']


DEFAULT_PICKLE_FMT = "{0}_from_pickle"
DEFAULT_TRANSFORM_FILENAME = "transforms.pickle"
DEFAULT_CHUNK_FMT = "chunk{0}.pickle"


T = TypeVar('T')


class RecSetBuildingError(Exception): ...


class BaseSet(dp.iter.IterDataPipe):
    """ 
    Base class for data pipes. Defines basic functionality and methods for 
    pre-processing the data for the learning models.
    """
    def __init__(self) -> None:
        super().__init__()

        self.__mode = 'train'
        
    @property
    def mode(self) -> str:
        """
        Return the mode in which the dataset is currently being used.

        Returns:
            str: The mode in which the dataset is currently being used.
        """

        return self.__mode

    def train(self: T) -> T:
        """Switch the dataset mode to 'train'."""
        self.__mode = 'train'
        return self

    def valid(self: T) -> T:
        """Switch the dataset mode to 'valid'."""
        self.__mode = 'valid'
        return self

    def test(self: T) -> T:
        """Switch the dataset mode to 'test'."""
        self.__mode = 'test'
        return self

    @property
    def fields(self) -> FieldTuple:
        """Return: A tuple containing the fields of the dataset."""
        raise NotImplementedError("Fields not defined for this dataset.")

    def __len__(self) -> int:
        """Return: The length of the dataset."""
        raise NotImplementedError()

    def summary(self):
        """Print a summary of the dataset."""
        infoLogger(str(self))

    @timemeter("DataSet/to_graph")
    def to_heterograph(self, *edge_types: Tuple[Tuple[FieldTags], Optional[str], Tuple[FieldTags]]) -> HeteroData:
        """
        Convert datapipe to a heterograph.

        Parameters:
        -----------
        *edge_types: Tuple[Tuple[str, str], Optional[str], Tuple[str, str]]
            The desired edges in the returned heterograph. Each edge is defined by a tuple (source, edge, destination):
            - source: Tuple of field tags for filtering the source node data.
            - edge: Optional[str], default None
                The name of the edge. If not provided, the name 'src.field2dst.field' will be used.
            - destination: Tuple of field tags for filtering the destination node data.

        Returns:
        --------
        HeteroData:
            The resulting heterograph, with the requested edges and nodes.

        Notes:
        ------
        A warning will be raised if the current mode is not 'train'!

        Examples:
        ---------
        >>> from freerec.data.datasets import Gowalla_m1
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> graph = basepipe.to_heterograph(
        ...    ((USER, ID), None, (ITEM, ID)),
        ...    ((ITEM, ID), None, (USER, ID))
        ... )
        >>> graph
        HeteroData(
            UserID={ x=[29858, 0] },
            ItemID={ x=[40981, 0] },
            (UserID, UserID2ItemID, ItemID)={ edge_index=[2, 810128] },
            (ItemID, ItemID2UserID, UserID)={ edge_index=[2, 810128] }
        )
        """

        # check mode and raise warning if not in 'train' mode
        if self.mode != 'train':
            warnLogger(f"Convert the datapipe for {self.mode} to graph. Make sure that this is intentional ...")

        srcs, edges, dsts = zip(*edge_types)
        srcs = [self.fields[src] for src in srcs]
        dsts = [self.fields[dst] for dst in dsts]
        nodes = set(srcs + dsts)
        edges = list(map(
            lambda src, edge, dst: edge if edge else f"{src.name}2{dst.name}",
            srcs, edges, dsts
        ))
        data = {node.name: [] for node in nodes}
        for chunk in self:
            for node in nodes:
                data[node.name].append(np.ravel(chunk[node.tags].data))
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
        src: Tuple[FieldTags] = (USER, ID),
        dst: Tuple[FieldTags] = (ITEM, ID),
        edge_type: Optional[str] = None
    ) -> HeteroData:
        """
        Convert datapipe to a bipartite graph.

        Parameters:
        ----------
        src: Tuple[FieldTags] 
            Source node.
        dst: Tuple[FieldTags] 
            Destination node.
        edge_type: str, optional 
            The name of the edge. `src.name2dst.name` will be specified if `edge_type` is `None`.

        Returns:
        --------
        HeteroData:
            The resulting heterograph, with the requested edges and nodes.

        Notes:
        ------
        A warning will be raised if the current mode is not 'train'!

        Examples:
        ---------
        >>> from freerec.data.datasets import Gowalla_m1
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> graph = basepipe.to_bigraph(
        ...    (USER, ID), (ITEM, ID)
        ... )
        >>> graph
        HeteroData(
            UserID={ x=[29858, 0] },
            ItemID={ x=[40981, 0] },
            (UserID, UserID2ItemID, ItemID)={ edge_index=[2, 810128] }
        )
        """
        return self.to_heterograph((src, edge_type, dst))
   
    def to_graph(self, src: Tuple[FieldTags], dst: Tuple[FieldTags]) -> Data:
        """
        Convert datapipe to a homogeneous graph.

        Parameters:
        ----------
        src: Tuple[FieldTags]
            Source node.
        dst: Tuple[FieldTags]
            Destination node.

        Returns:
        --------
        HeteroData:
            The resulting heterograph, with the requested edges and nodes.

        Notes:
        ------
        A warning will be raised if current mode is not 'train' !

        Examples:
        --------
        >>> from freerec.data.datasets import Gowalla_m1
        >>> from freerec.data.tags import USER, ITEM, ID
        >>> basepipe = Gowalla_m1("../data")
        >>> fields = basepipe.fields
        >>> graph = basepipe.to_graph(
        ...    fields[USER, ID], fields[ITEM, ID],
        ... )
        >>> graph
        Data(edge_index=[2, 1620256], x=[70839, 0])
        """
        graph = self.to_heterograph((src, None, dst)).to_homogeneous()
        graph.edge_index = to_undirected(graph.edge_index)
        return graph

    def pickle2data(self):
        raise NotImplementedError(
                f"{self.__class__.__name__}.pickle2data should be implemented before using ..."
        )

    def __iter__(self) -> Iterator[FieldList[BufferField]]:
        for cols in self.pickle2data():
            yield FieldList(map(
                lambda field, col: field.buffer(col),
                self.fields,
                cols
            ))


class RecDataSet(BaseSet):
    """ 
    RecDataSet provides a template for specific datasets.

    Parameters:
    -----------
    root: str
        The path storing datasets.
    filename: str, optional 
        The dirname of the dataset. If `None`, sets the classname as the filename.
    download: bool 
        Download the dataset from a URL.

    Attributes:
    -----------
    _cfg: Config[str, Field] 
        Includes fields of each column.
    DEFAULT_CHUNK_SIZE: int, default 51200 
        Chunk size for saving.
    VALID_IS_TEST: bool 
        The validset and testset are the same one sometimes.

    Notes:
    ------
    All datasets that inherit RecDataSet should define the class variable `_cfg` before instantiation.
    Generally speaking, the dataset will be split into:
        - trainset
        - validset
        - testset
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
        """Return: Tuple of Field."""
        self.__fields = FieldTuple(vals)

    def check_transforms(self) -> None:
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

    def save_transforms(self) -> None:
        """Save transformers in a pickle format."""
        infoLogger(f"[{self.__class__.__name__}] >>> Save transformers ...")
        # Get the state dictionary of the fields and add the size information.
        state_dict = self.fields.state_dict()
        state_dict['trainsize'] = self.trainsize
        state_dict['validsize'] = self.validsize
        state_dict['testsize'] = self.testsize
        file_ = os.path.join(
            self.path, 
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            DEFAULT_TRANSFORM_FILENAME
        )
        # Export the state dictionary as a pickle file.
        export_pickle(state_dict, file_)

    def load_transforms(self) -> None:
        """Load transformers from a pickle file."""
        file_ = os.path.join(
            self.path, 
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__), 
            DEFAULT_TRANSFORM_FILENAME
        )
        # Import the state dictionary from the pickle file and update the size information.
        state_dict = import_pickle(file_)
        self.trainsize = state_dict['trainsize']
        self.validsize = state_dict['validsize']
        self.testsize = state_dict['testsize']
        # Load the state dictionary into the fields.
        self.fields.load_state_dict(state_dict, strict=False)

    def check_pickle(self) -> bool:
        """Check if the dataset has been converted into pickle format."""
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

    def write_pickle(self, data, count: int) -> None:
        """
        Save pickle format data.

        Parameters:
        -----------
        data: Any
            The data to be saved in pickle format.
        count: int
            The count of the data chunks.

        Returns:
        --------
        None.
        """
        file_ = os.path.join(
            self.path,
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__),
            self.mode, DEFAULT_CHUNK_FMT.format(count)
        )
        export_pickle(data, file_)

    def read_pickle(self, file_: str):
        """
        Load pickle format data.

        Parameters:
        -----------
        file_: str 
            The file path to load the pickle format data.

        Returns:
        --------
        The loaded pickle format data.
        """
        return import_pickle(file_)

    def row_processer(self, row):
        """Process a row of raw data.

        Parameters:
        -----------
        row: Any
            A row of raw data.

        Returns:
        --------
        A processed row of data.
        """
        return [field.caster(val) for val, field in zip(row, self.fields)]

    def raw2data(self) -> dp.iter.IterableWrapper:
        """
        Process raw data.

        Returns:
        --------
        A processed data.

        Raises:
        -------
        NotImplementedError: Subclasses should implement this method.

        Notes:
        ------
        This method should be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.raw2data() method should be implemented ...")

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
        """
        Read the pickle data and return it as a generator.

        Yields:
        -------
        A chunk of the pickle data.
        """
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
        """
        Check current dataset and transformations.

        Flows:
        ------
        1. Check whether the transformation has been fitted:
            - `True`: Skip.
            - `False`: Fit the total trainset and the `SPARSE` fields in valid|testset
                to avoid unseen features. This operation will not cause information leakage.
            
        2. Convert each set into pickle format for fast loading.
        """

        def fit_transform(*tags):
            """Function to fit and transform data for pickle conversion."""
            datapipe = self.raw2data().batch(self.DEFAULT_CHUNK_SIZE).collate(collate_list)
            datasize = 0
            fields = self.fields.groupby(*tags)
            for chunk in datapipe:
                datasize += len(chunk[0])
                chunk = FieldList(map(lambda field, col: field.buffer(col), self.fields, chunk)).groupby(*tags)
                for j, field in enumerate(fields):
                    field.partial_fit(chunk[j].data)
            return datasize

        if self.check_transforms():
            self.load_transforms()
        else:
            self.train()
            self.trainsize = fit_transform()

            # avoid unseen IDs not included in trainset
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
        """Return the size of dataset according to the current mode."""
        if self.mode == 'train':
            return self.trainsize
        elif self.mode == 'valid':
            return self.validsize
        else:
            return self.testsize

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
    """
    Implicit feedback data.
    The data should be collected in the order of users; that is,
    each row represents a user's interacted items.
    """

    # Same validset and testset are the same now !
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