

from typing import Any, TypeVar, Literal, Union, Optional, Iterator, Iterable, Dict, Tuple, List

import torch, os, abc
import numpy as np
import pandas as pd
import torchdata.datapipes as dp
from copy import copy
from functools import lru_cache

from ..tags import FieldTags, USER, ITEM, ID, RATING, TIMESTAMP, FEATURE, SEQUENCE
from ..fields import Field, FieldTuple
from ..utils import download_from_url, extract_archive
from ...utils import timemeter, infoLogger, warnLogger
from ...dict2obj import Config


__all__ = ['BaseSet', 'RecDataSet']


DEFAULT_PICKLE_FMT = "{0}_from_pickle"
DEFAULT_TRANSFORM_FILENAME = "transforms.pkl"
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

    field_builder = {
        USER.name: Field(USER.name, USER, ID),
        ITEM.name: Field(ITEM.name, ITEM, ID),
        RATING.name: Field(RATING.name, RATING),
        TIMESTAMP.name: Field(TIMESTAMP.name, TIMESTAMP)
    }

    open_kwargs = Config(
        trainfile='train.txt', validfile='valid.txt', testfile='test.txt',
        userfile='user.txt', itemfile='item.txt',
        delimiter='\t'
    )

    URL: str
    DATATYPE: str

    def __new__(cls, *args, **kwargs):
        for attr in ('DATATYPE',):
            if not hasattr(cls, attr):
                raise RecSetBuildingError(f"'{attr}' should be defined before instantiation ...")
        return super().__new__(cls)

    def __init__(
        self, root: str, filedir: Optional[str] = None, download: bool = True
    ) -> None:
        super().__init__()

        self.fields = []
        self.__mode: Literal['train', 'valid', 'test'] = 'train'

        filedir = filedir if filedir else self.__class__.__name__
        self.path = os.path.join(root, 'Processed', filedir)
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
    def mode(self) -> Literal['train', 'test', 'valid']:
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
    def datasize(self):
        """Return the size of dataset according to the current mode."""
        if self.mode == 'train':
            return self.trainsize
        elif self.mode == 'valid':
            return self.validsize
        else:
            return self.testsize

    @property
    def interdata(self) -> Dict[Field, Tuple]:
        return self.__interdata[self.mode]

    @property
    def userdata(self) -> Dict[Field, Tuple]:
        try:
            return self.__userdata
        except AttributeError:
            self.load_user()
            return self.__userdata

    @property
    def itemdata(self) -> Dict[Field, Tuple]:
        try:
            return self.__itemdata
        except AttributeError:
            self.load_item()
            return self.__itemdata

    @property
    def fields(self) -> FieldTuple[Field]:
        """Return: Tuple of Field."""
        return self.__fields

    @fields.setter
    def fields(self, fields: Iterable[Field]):
        self.__fields = FieldTuple(fields)
        traindata = dict()
        validdata = dict()
        testdata = dict()
        for field in self.fields:
            try:
                traindata[field] = self.train().interdata[field]
                validdata[field] = self.valid().interdata[field]
                testdata[field] = self.test().interdata[field]
            except KeyError:
                raise ValueError(
                    f"Setting a {field} not in `self.fields` ..."
                )
        self.__interdata = {
            'train': traindata, 'valid': validdata, 'test': testdata
        }

    @classmethod
    def build_fields(cls, columns: Iterable[str], *tags: FieldTags) -> List[Field]:
        fields = []
        for colname in columns:
            field = cls.field_builder.get(
                colname,
                Field(colname, FEATURE)
            ).fork(*tags)
            fields.append(field)
        return fields

    def load_inter(self):
        train_df = pd.read_csv(
            os.path.join(self.path, self.open_kwargs.trainfile),
            delimiter=self.open_kwargs.delimiter
        )
        valid_df = pd.read_csv(
            os.path.join(self.path, self.open_kwargs.validfile),
            delimiter=self.open_kwargs.delimiter
        )
        test_df = pd.read_csv(
            os.path.join(self.path, self.open_kwargs.testfile),
            delimiter=self.open_kwargs.delimiter
        )

        self.trainsize = len(train_df)
        self.validsize = len(valid_df)
        self.testsize = len(test_df)

        self.__fields = self.build_fields(train_df.columns)

        self.__interdata = {
            'train': dict(), 'valid': dict(), 'test': dict()
        }
        for field in self.fields:
            colname = field.name
            traindata = field.try_to_numeric(train_df[colname])
            validdata = field.try_to_numeric(valid_df[colname])
            testdata = field.try_to_numeric(test_df[colname])

            field.count = len(
                set(traindata) | set(validdata) | set(testdata)
            )

            self.__interdata['train'][field] = traindata
            self.__interdata['valid'][field] = validdata
            self.__interdata['test'][field] = testdata

    def load_user(self):
        user_df = pd.read_csv(
            os.path.join(self.path, self.open_kwargs.userfile),
            delimiter=self.open_kwargs.delimiter
        )
        fields = self.build_fields(user_df.columns, USER)
        self.__userdata = {}
        for field in fields:
            colname = field.name
            self.__userdata[field] = field.try_to_numeric(user_df[colname])

    def load_item(self):
        item_df = pd.read_csv(
            os.path.join(self.path, self.open_kwargs.itemfile),
            delimiter=self.open_kwargs.delimiter
        )
        fields = self.build_fields(item_df.columns, ITEM)
        self.__itemdata = {}
        for field in fields:
            colname = field.name
            data = field.try_to_numeric(item_df[colname])
            field.count = len(set(data))
            self.__itemdata[field] = data

    def summary(self):
        """Print a summary of the dataset."""
        infoLogger(str(self))

    @timemeter
    def compile(self):
        self.load_inter()
        self.summary()

    def match_all(self, *tags: FieldTags) -> 'BaseSet':
        """Return a copy of dataset with fields matching all given tags."""
        dataset = copy(self)
        dataset.fields = self.fields.match_all(*tags)
        return dataset

    def match_any(self, *tags: FieldTags) -> 'BaseSet':
        """Return a copy of dataset with fields matching any given tags."""
        dataset = copy(self)
        dataset.fields = self.fields.match_any(*tags)
        return dataset

    def __str__(self) -> str:
        cfg = '\n'.join(map(str, self.fields))
        return f"[{self.__class__.__name__}] >>> \n" + cfg

    def __iter__(self) -> Iterator[Dict[Field, Any]]:
        fields = self.interdata.keys()
        for values in zip(*self.interdata.values()):
            yield dict(zip(fields, values))


class RecDataSet(BaseSet):
    """RecDataSet provides a template for specific datasets."""

    def to_pairs(self) -> Iterator[Dict[Field, int]]:
        r"""Return (User, Item) in pairs."""
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        for user, item in zip(self.interdata[User], self.interdata[Item]):
            yield {User: user, Item: item}

    def to_seqs(self) -> Iterator[Dict[Field, int]]:
        r"""
        Return dataset in sequence.

        Parameters:
        -----------
        master: Tuple
            Tuple of tags to spefic a field, e.g., (USER, ID), (SESSION, ID)
        keepid: bool, default to False
            `True`: return list of (id, items)
            `False`: return list of items
        """
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID].fork(SEQUENCE)
        seqs = [[] for id_ in range(User.count)]

        for user, item in self.to_pairs():
            seqs[user].append(item)

        for user, seq in enumerate(seqs):
            yield {User: user, Item: tuple(seq)}

    def to_roll_seqs(
        self, minlen: int = 2, maxlen: Optional[int] = None,
        keep_at_least_itself: bool = True
    ) -> Iterator[Dict(Field, Union[int, Tuple[int]])]:
        r"""
        Rolling dataset in sequence.

        Parameters:
        -----------
        minlen: int
            Shorest sequence
        maxlen: int, optional
            Maximum length
            `None`: Roll throughout the whole sequence
        keep_at_least_itself: bool, default to `True`
            `True`: Keep the sequence with items less than `minlen`
        """
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID].fork(SEQUENCE)

        for data in self.to_seqs():
            user, seq = data[User], data[Item]
            if maxlen is not None:
                seq = seq[-maxlen:]
            if len(seq) <= minlen and keep_at_least_itself:
                yield {User: user, Item: seq}
                continue
            for k in range(minlen, len(seq) + 1):
                yield {User: user, Item: seq[:k]}

    @lru_cache()
    def has_duplicates(self) -> bool:
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
        Item = self.fields[ITEM, ID].fork(SEQUENCE)

        for d1, d2, d3 in zip(
            self.train().to_seqs(),
            self.valid().to_seqs(),
            self.test().to_seqs()
        ):
            seq = list(chain(d1[Item], d2[Item], d3[Item]))
            if len(seq) != len(set(seq)):
                return True
        return False

    def seqlens(self) -> List:
        Item = self.fields[ITEM, ID].fork(SEQUENCE)
        seqlens = [len(data[Item]) for data in self.to_seqs()]
        return list(filter(lambda x: x > 0, seqlens))

    @property
    def maxlen(self) -> int:
        return np.max(self.seqlens()).item()

    @property
    def minlen(self) -> int:
        return np.min(self.seqlens()).item()

    @property
    def meanlen(self) -> int:
        return np.mean(self.seqlens()).item()

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
            warnLogger(f"Convert the {self.mode} datapipe to graph. Make sure that this is intentional ...")

        srcs, edges, dsts = zip(*edge_types)
        srcs = [self.fields[src] for src in srcs]
        dsts = [self.fields[dst] for dst in dsts]
        nodes = set(srcs + dsts)
        edges = list(map(
            lambda src, edge, dst: edge if edge else f"{src.name}2{dst.name}",
            srcs, edges, dsts
        ))
        data = {node.name: self.interdata[node] for node in nodes}
        for key in data:
            data[key] = torch.tensor(data, dtype=torch.long)

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
   
    def to_graph(
        self, 
        src: Tuple[FieldTags] = (USER, ID), 
        dst: Tuple[FieldTags] = (ITEM, ID)
    ):
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
        Data:
            The resulting graph, with the requested edges and nodes.

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

    def to_normalized_adj(
        self,
        src: Tuple[FieldTags] = (USER, ID), 
        dst: Tuple[FieldTags] = (ITEM, ID),
        normalization: str = 'sym'
    ):
        r"""
        Convert datapipe to a normalized adjacency matrix.

        Parameters:
        ----------
        src: Tuple[FieldTags]
            Source node.
        dst: Tuple[FieldTags]
            Destination node.
        normalization: str
            `sym`: Symmetric sqrt normalization
                :math: `\mathbf{\tilde{A}} = \mathbf{D}_l^{-1/2} \mathbf{A} \mathbf{D}_r^{-1/2}`
            'left': Left-side normalization
                :math: `\mathbf{\tilde{A}} = \mathbf{D}_l^{-1} \mathbf{A}`
            'right': Right-side normalization
                :math: `\mathbf{\tilde{A}} = \mathbf{A} \mathbf{D}_r^{-1}`
    
        Returns:
        --------
        Adj: CSR Tensor
        """
        from ...graph import to_normalized, to_adjacency
        User = self.fields[src]
        Item = self.fields[dst]
        edge_index, edge_weight = to_normalized(
            self.to_graph(src, dst).edge_index, normalization=normalization
        )
        return to_adjacency(
            edge_index, edge_weight,
            num_nodes=User.count + Item.count
        )