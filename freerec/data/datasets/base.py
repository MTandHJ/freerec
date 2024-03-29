

from typing import Iterator, Optional, TypeVar, Tuple, List

import torch, os, abc
import numpy as np
import torchdata.datapipes as dp
from functools import lru_cache
from freeplot.utils import import_pickle, export_pickle

from ..tags import FieldTags, SPARSE, USER, SESSION, ITEM, ID
from ..fields import Field, BufferField, FieldList, FieldTuple
from ..utils import download_from_url, extract_archive
from ...utils import timemeter, infoLogger, mkdirs, warnLogger


__all__ = ['BaseSet', 'RecDataSet']


DEFAULT_PICKLE_FMT = "{0}_from_pickle"
DEFAULT_TRANSFORM_FILENAME = "transforms.pickle"
DEFAULT_CHUNK_FMT = "chunk{0}.pickle"


T = TypeVar('T')


#===============================Basic Class===============================

class RecSetBuildingError(Exception): ...


class BaseSet(dp.iter.IterDataPipe, metaclass=abc.ABCMeta):
    r""" 
    Base class for data pipes. Defines basic functionality and methods for 
    pre-processing the data for the learning models.

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
    DATATYPE: str
        Dataset type.
        - `General': for general recommendation.
        - `Sequential': for sequential recommendation.
        - `Session': for session-based recommendation.
        - `Context': for context-aware recommendation.
        - `Knowledge': for knowledge-based recommendation.
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
    DATATYPE: str
    VALID_IS_TEST: bool

    def __new__(cls, *args, **kwargs):
        for attr in ('_cfg', 'DATATYPE', 'VALID_IS_TEST'):
            if not hasattr(cls, attr):
                raise RecSetBuildingError(f"'{attr}' should be defined before instantiation ...")
        assert 'fields' in cls._cfg, "the config of fields should be defined in '_cfg' ..."
        return super().__new__(cls)

    def __init__(self, root: str, filename: Optional[str] = None, download: bool = True) -> None:
        super().__init__()

        self.trainsize: int = 0
        self.validsize: int = 0
        self.testsize: int = 0

        fields = []
        for field_type, cfg in self._cfg['fields']:
            fields.append(field_type(**cfg))
        self.fields = fields
        self.__mode = 'train'

        filename = filename if filename else self.__class__.__name__
        self.path = os.path.join(root, self.DATATYPE, filename)
        if not os.path.exists(self.path) or not any(True for _ in os.scandir(self.path)):
            if download:
                extract_archive(
                    download_from_url(self.URL, root, overwrite=False),
                    self.path
                )
            else:
                raise FileNotFoundError(f"No such file of {self.path}, or this dir is empty ...")

        self.compile()
        self.check()

    def check(self):
        """Self-check program should be placed here."""
        ...
        
    @property
    def mode(self) -> str:
        r"""
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
    def fields(self):
        return self.__fields

    @fields.setter
    def fields(self, vals) -> FieldTuple[Field]:
        """Return: Tuple of Field."""
        self.__fields = FieldTuple(vals)

    @abc.abstractmethod
    def raw2data(self) -> dp.iter.IterableWrapper:
        r"""
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

    @abc.abstractmethod
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

    def summary(self):
        """Print a summary of the dataset."""
        infoLogger(str(self))

    @timemeter
    def to_heterograph(self, *edge_types: Tuple[Tuple[FieldTags], Optional[str], Tuple[FieldTags]]):
        r"""
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
        from torch_geometric.data import HeteroData

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
    ):
        r"""
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
   
    def to_graph(self, src: Tuple[FieldTags], dst: Tuple[FieldTags]):
        r"""
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
        from torch_geometric.utils import to_undirected
        graph = self.to_heterograph((src, None, dst)).to_homogeneous()
        graph.edge_index = to_undirected(graph.edge_index)
        return graph


class RecDataSet(BaseSet):
    """RecDataSet provides a template for specific datasets."""

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

    def load_transforms(self) -> None:
        """Load transformers from a pickle file."""
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
        r"""
        Save pickle format data.

        Parameters:
        -----------
        data: Any
            The data to be saved in pickle format.
        count: int
            The count of the data chunks.

        Returns:
        --------
        None
        """
        file_ = os.path.join(
            self.path,
            DEFAULT_PICKLE_FMT.format(self.__class__.__name__),
            self.mode, DEFAULT_CHUNK_FMT.format(count)
        )
        export_pickle(data, file_)

    def read_pickle(self, file_: str):
        r"""
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
        r"""
        Process a row of raw data.

        Parameters:
        -----------
        row: Any
            A row of raw data.

        Returns:
        --------
        A processed row of data.
        """
        return [field.caster(val) for val, field in zip(row, self.fields)]

    def raw2pickle(self):
        """Convert raw data into pickle format."""
        infoLogger(f"[{self.__class__.__name__}] >>> Convert raw data ({self.mode}) to chunks in pickle format")
        datapipe = self.raw2data()
        count = 0
        for chunk in datapipe.batch(batch_size=self.DEFAULT_CHUNK_SIZE).column_():
            for j, field in enumerate(self.fields):
                chunk[j] = field.transform(chunk[j])
            self.write_pickle(chunk, count)
            count += 1
        infoLogger(f"[{self.__class__.__name__}] >>> {count} chunks done")

    def pickle2data(self):
        r"""
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
            ),
            non_deterministic=False # return sorted chunks
        )
        for file_ in datapipe:
            yield self.read_pickle(file_)

    @timemeter
    def compile(self):
        r"""
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
            datapipe = self.raw2data().batch(self.DEFAULT_CHUNK_SIZE).column_()
            datasize = 0
            fields = self.fields.groupby(*tags)
            for chunk in datapipe:
                datasize += len(chunk[0])
                for field in fields:
                    index = self.fields.index(*field.tags)
                    field.partial_fit(chunk[index])
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

    def to_pairs(self, master: Tuple = (USER, ID)) -> List:
        r"""
        Return dataset in pairs.

        Parameters:
        -----------
        master: Tuple
            Tuple of tags to spefic a field, e.g., (USER, ID), (SESSION, ID)
        
        Returns:
        --------
        List
        """
        Master = self.fields[master]
        assert Master is not None, f"{Master} is not in fields ..."
        pairs = []

        for chunk in self:
            pairs.extend(list(zip(chunk[master], chunk[ITEM, ID])))
        return pairs

    def to_seqs(self, master: Tuple = (USER, ID), keepid: bool = False) -> List:
        r"""
        Return dataset in sequence.

        Parameters:
        -----------
        master: Tuple
            Tuple of tags to spefic a field, e.g., (USER, ID), (SESSION, ID)
        keepid: bool, default to False
            `True`: return list of (id, items)
            `False`: return list of items
        
        Returns:
        --------
        List
        """
        Master = self.fields[master]
        assert Master is not None, f"{Master} is not in fields ..."
        seqs = [[] for id_ in range(Master.count)]

        for chunk in self:
            list(map(
                lambda id_, item: seqs[id_].append(item),
                chunk[master], chunk[ITEM, ID]
            ))

        if keepid:
            seqs = [(id_, tuple(items)) for id_, items in enumerate(seqs)]
        else:
            seqs = [tuple(items) for items in seqs]

        return seqs

    def to_roll_seqs(
        self, master: Tuple = (USER, ID), 
        minlen: int = 2, maxlen: Optional[int] = None,
        keep_at_least_itself: bool = True
    ) -> List:
        r"""
        Rolling dataset in sequence.

        Parameters:
        -----------
        master: Tuple
            Tuple of tags to spefic a field, e.g., (USER, ID), (SESSION, ID)
        minlen: int
            Shorest sequence
        maxlen: int, optional
            Maximum length
            `None`: Roll throughout the whole sequence
        keep_at_least_itself: bool, default to `True`
            `True`: Keep the sequence with items less than `minlen`
       
        Returns:
        --------
        List
        """
        seqs = self.to_seqs(master=master, keepid=True)

        roll_seqs = []
        for id_, items in seqs:
            if maxlen is not None:
                items = items[-maxlen:]
            if len(items) <= minlen and keep_at_least_itself:
                roll_seqs.append(
                    (id_, items)
                )
                continue
            for k in range(minlen, len(items) + 1):
                roll_seqs.append(
                    (id_, items[:k])
                )

        return roll_seqs

    def seqlens(self, master: Tuple = (USER, ID)) -> List:
        seqs = self.to_seqs(master, keepid=False)
        return list(filter(lambda x: x > 0, [len(items) for items in seqs]))

    @lru_cache()
    def has_duplicates(self, master: Tuple = (USER, ID)) -> bool:
        r"""
        Check whether the dataset has repeated interactions.

        Parameters:
        -----------
        master: Tuple
            Tuple of tags to spefic a field, e.g., (USER, ID), (SESSION, ID)

        Returns:
        --------
        bool
        """
        from itertools import chain
        train_seqs = self.train().to_seqs(master, keepid=False)
        valid_seqs = self.valid().to_seqs(master, keepid=False)
        test_seqs = self.test().to_seqs(master, keepid=False)
        seqs = map(
            lambda triple: chain(*triple),
            zip(train_seqs, valid_seqs, test_seqs)
        )
        for seq in seqs:
            seq = list(seq)
            if len(seq) != len(set(seq)):
                return True
        return False

    @property
    def maxlen(self) -> int:
        return np.max(self.seqlens()).item()

    @property
    def minlen(self) -> int:
        return np.min(self.seqlens()).item()

    @property
    def meanlen(self) -> int:
        return np.mean(self.seqlens()).item()

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg
