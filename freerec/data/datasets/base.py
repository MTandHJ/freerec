

from typing import Any, TypeVar, Literal, Union, Optional, Callable, Iterator, Iterable, Dict, Tuple, List

import torch, os, abc
import numpy as np
import pandas as pd
import torchdata.datapipes as dp
from copy import copy
from functools import lru_cache

from ..tags import FieldTags, TaskTags, USER, ITEM, ID, RATING, TIMESTAMP, FEATURE, SEQUENCE
from ..fields import Field, FieldTuple
from ..utils import download_from_url, extract_archive
from ...utils import timemeter, infoLogger, warnLogger
from ...dict2obj import Config


__all__ = ['BaseSet', 'RecDataSet']


T = TypeVar('T')


def safe_mode(*modes):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if self.mode not in modes:
                fname = f"\033[0m\033[0;31;47m{func.__name__}\033[0m\033[1;31m"
                mode = f"\033[0m\033[0;31;47m{self.mode}\033[0m\033[1;31m"
                warnLogger(f"{fname} runs in {mode} mode. Make sure that this is intentional ...")
            return func(self, *args, **kwargs)
        wrapper.__name__ == func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


#===============================Basic Class===============================

class RecSetBuildingError(Exception): ...


class BaseSet(dp.iter.IterDataPipe, metaclass=abc.ABCMeta):
    r""" 
    Base class for data pipes. Defines basic functionality and methods for 
    pre-processing the data for the learning models.

    Parameters:
    -----------
    root: str
        The root path storing datasets.
    filefir: str, optional 
        The dirname of the dataset. If `None`, set the classname as the filedir.
    download: bool 
        Download the dataset from a URL.
    """

    _field_builder = {
        USER.name: Field(USER.name, USER, ID),
        ITEM.name: Field(ITEM.name, ITEM, ID),
        RATING.name: Field(RATING.name, RATING),
        TIMESTAMP.name: Field(TIMESTAMP.name, TIMESTAMP)
    }

    _open_kwargs = Config(
        trainfile='train.txt', validfile='valid.txt', testfile='test.txt',
        userfile='user.txt', itemfile='item.txt',
        sep='\t'
    )

    TASK: TaskTags
    URL: Optional[str] = None

    def __init__(
        self, 
        root: str, 
        filedir: Optional[str] = None, 
        download: bool = True,
        tasktag: Optional[TaskTags] = None
    ) -> None:
        super().__init__()

        self.fields = []
        self.__mode: Literal['train', 'valid', 'test'] = 'train'
        if tasktag is not None:
            self.TASK = tasktag

        filedir = filedir if filedir else self.__class__.__name__
        self.path = os.path.join(root, 'Processed', filedir)
        if not os.path.exists(self.path) or not any(True for _ in os.scandir(self.path)):
            if download and self.URL is not None:
                extract_archive(
                    download_from_url(self.URL, root, overwrite=False),
                    self.path
                )
            else:
                raise FileNotFoundError(
                    f"No such file of {self.path}, or this dir is empty. \n"
                    f"Please use `freerec make` to prepare the dataset with the following setting: \n {self.__class__.__doc__}"
                )

        self.compile()
        self.check()

    def check(self):
        """Self-check program should be placed here."""
        ...
        
    @property
    def mode(self) -> Literal['train', 'test', 'valid']:
        """Return the current mode."""
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
        r"""
        Get interaction data.

        Examples:
        ---------
        >>> dataset: BaseSet
        >>> dataset.train().interdata == dataset.valid().interdata
        False
        """
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
        """Return a tuple of Field."""
        return self.__fields

    @fields.setter
    def fields(self, fields: Iterable[Field]):
        r"""
        Set fields.

        Notes:
        ------
        Set fields will change the saved interaction data.
        """
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
    def build_fields(cls, columns: Iterable[str], *tags: FieldTags) -> FieldTuple[Field]:
        fields = []
        for colname in columns:
            field = cls._field_builder.get(
                colname,
                Field(colname, FEATURE)
            ).fork(*tags)
            fields.append(field)
        return FieldTuple(fields)

    def load_inter(self):
        r"""
        Load interaction data.

        Flows:
        ------
        1. Read dataframe according `_open_kwargs`.
        2. Record datasize.
        3. Build fields.
        4. Transform each field data.
        """
        train_df = pd.read_csv(
            os.path.join(self.path, self._open_kwargs.trainfile),
            sep=self._open_kwargs.sep
        )
        valid_df = pd.read_csv(
            os.path.join(self.path, self._open_kwargs.validfile),
            sep=self._open_kwargs.sep
        )
        test_df = pd.read_csv(
            os.path.join(self.path, self._open_kwargs.testfile),
            sep=self._open_kwargs.sep
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
            os.path.join(self.path, self._open_kwargs.userfile),
            sep=self._open_kwargs.sep
        )
        fields = self.build_fields(user_df.columns, USER)
        self.__userdata = {}
        for field in fields:
            colname = field.name
            data = field.try_to_numeric(user_df[colname])
            field.count = len(set(data))
            self.__userdata[field] = data

    def load_item(self):
        item_df = pd.read_csv(
            os.path.join(self.path, self._open_kwargs.itemfile),
            sep=self._open_kwargs.sep
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
        self.train()

    def match_all(self: T, *tags: FieldTags) -> T:
        r"""
        Return a copy of dataset with fields matching all given tags.

        Examples:
        ---------
        >>> dataset: BaseSet
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM|RATING:RATING|TIMESTAMP:TIMESTAMP)
        >>> dataset.match_all(ID)
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM)
        >>> dataset.match_all()
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM|RATING:RATING|TIMESTAMP:TIMESTAMP)
        """
        dataset = copy(self)
        dataset.fields = self.fields.match_all(*tags)
        return dataset

    def match_any(self: T, *tags: FieldTags) -> T:
        r"""
        Return a copy of dataset with fields matching any given tags.

        Examples:
        ---------
        >>> dataset: BaseSet
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM|RATING:RATING|TIMESTAMP:TIMESTAMP)
        >>> dataset.match_any(ID, TIMESTAMP)
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM|TIMESTAMP:TIMESTAMP)
        >>> dataset.match_any()
        RecDataSet()
        """
        dataset = copy(self)
        dataset.fields = self.fields.match_any(*tags)
        return dataset

    @staticmethod
    def listmap(func: Callable, *iterables) -> List:
        r"""
        Apply a function to multiple iterables and return a list.

        Parameters:
        -----------
        func (Callable): The function to be applied.
        *iterables: Multiple iterables to be processed.

        Returns:
        --------
        List: The results after applying the function to the iterables.
        """
        return list(map(func, *iterables))

    @classmethod
    def to_rows(cls, field_dict: Dict[Field, Iterable[T]]) -> List[Dict[Field, T]]:
        fields = field_dict.keys()
        return cls.listmap(
            lambda values: dict(zip(fields, values)),
            zip(*field_dict.values())
        )

    def __repr__(self) -> str:
        cfg = '|'.join(map(str, self.fields))
        return f"{self.__class__.__name__}({cfg})"

    def __str__(self) -> str:
        cfg = ' | '.join(map(str, self.fields))
        return f"[{self.__class__.__name__}] >>> " + cfg

    def __iter__(self) -> Iterator[Dict[Field, Any]]:
        yield from iter(
            self.to_rows(self.interdata)
        )


class RecDataSet(BaseSet):
    """RecDataSet provides a template for specific datasets."""

    def to_pairs(self) -> List[Dict[Field, int]]:
        """Return (User, Item) in pairs."""
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        users, items = self.interdata[User], self.interdata[Item]
        return self.to_rows({User: users, Item:items})

    def to_seqs(self, maxlen: Optional[int] = None) -> List[Dict[Field, Union[int, Tuple[int]]]]:
        r"""
        Return dataset in sequence.

        Parameters:
        -----------
        maxlen: int, optional
            Maximum length
            `None`: return the whole sequence
            `int`: return the recent `maxlen` items
        """
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        seqs = [[] for id_ in range(User.count)]

        self.listmap(
            lambda data: seqs[data[User]].append(data[Item]),
            self.to_pairs()
        )
        users = list(range(User.count))
        if maxlen is not None:
            seqs = [tuple(items[-maxlen:]) for items in seqs]
        else:
            seqs = [tuple(items) for items in seqs]

        return self.to_rows({User: users, Item.fork(SEQUENCE): seqs})

    def to_roll_seqs(
        self, minlen: int = 2, maxlen: Optional[int] = None,
        keep_at_least_itself: bool = True
    ) -> List[Dict[Field, Union[int, Tuple[int]]]]:
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
        ISeq = self.fields[ITEM, ID].fork(SEQUENCE)
        data = self.to_seqs(maxlen)

        roll_seqs = []
        for row in data:
            user, seq = row[User], row[ISeq]
            if len(seq) <= minlen and keep_at_least_itself:
                roll_seqs.append(
                    {User: user, ISeq: seq}
                )
                continue
            for k in range(minlen, len(seq) + 1):
                roll_seqs.append(
                    {User: user, ISeq: seq[:k]}
                )

        return roll_seqs

    def seqlens(self) -> List[int]:
        ISeq = self.fields[ITEM, ID].fork(SEQUENCE)
        seqlens = [len(row[ISeq]) for row in self.to_seqs()]
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

    @lru_cache()
    def has_duplicates(self) -> bool:
        """Check whether the dataset has repeated interactions."""
        ISeq = self.fields[ITEM, ID].fork(SEQUENCE)
        traindata = self.train().to_seqs()
        validdata = self.valid().to_seqs()
        testdata = self.test().to_seqs()

        for triplet in zip(traindata, validdata, testdata):
            seq = triplet[0][ISeq] + triplet[1][ISeq] + triplet[2][ISeq]
            if len(seq) != len(set(seq)):
                return True
        return False

    @safe_mode('train')
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
        >>> dataset: RecDataSet
        >>> graph = dataset.to_heterograph(
        ...    ((USER, ID), None, (ITEM, ID)),
        ...    ((ITEM, ID), None, (USER, ID))
        ... )
        >>> graph
        HeteroData(
            USER={ x=[22363, 0] },
            ITEM={ x=[12101, 0] },
            (USER, USER2ITEM, ITEM)={ edge_index=[2, 153776] },
            (ITEM, ITEM2USER, USER)={ edge_index=[2, 153776] }
        )
        """
        from torch_geometric.data import HeteroData

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
            data[key] = torch.tensor(data[key], dtype=torch.long)

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
        >>> dataset: RecDataSet
        >>> graph = dataset.to_bigraph(
        ...    (USER, ID), (ITEM, ID)
        ... )
        >>> graph
        HeteroData(
            USER={ x=[22363, 0] },
            ITEM={ x=[12101, 0] },
            (USER, USER2ITEM, ITEM)={ edge_index=[2, 153776] }
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
        >>> dataset: RecDataSet
        >>> graph = basepipe.to_graph(
        ...    fields[USER, ID], fields[ITEM, ID],
        ... )
        >>> graph
        Data(edge_index=[2, 307552], x=[34464, 0], node_type=[34464], edge_type=[153776])
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

    @safe_mode('valid', 'test')
    def ordered_user_ids_source(self):
        r"""
        To ordered User ID source.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.valid().to_ordered_user_ids()
        >>> len(datapipe) == dataset.fields[USER, ID].count
        True
        """
        from ..postprocessing.source import OrderedSource
        User = self.fields[USER, ID]
        source = self.to_rows({User: list(range(User.count))})
        return OrderedSource(self, source)

    @safe_mode('train')
    def choiced_user_ids_source(self):
        r"""
        To random choiced User ID source.
        The datasize equals the current dataset's datasize.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().to_choiced_user_ids()
        >>> len(datapipe) == dataset.trainsize
        True
        """
        from ..postprocessing.source import RandomChoicedSource
        User = self.fields[USER, ID]
        source = self.to_rows({User: list(range(User.count))})
        return RandomChoicedSource(self, source)

    @safe_mode('train')
    def shuffled_pairs_source(self):
        r"""
        To random shuffled (User, Item) pairs source.
        The datasize equals the current dataset's datasize.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().to_shuffled_pairs()
        >>> len(datapipe) == dataset.trainsize
        True
        >>> list(datapipe)[0].keys()
        dict_keys([Field(USER:ID,USER), Field(ITEM:ID,ITEM)])
        """
        from ..postprocessing.source import RandomShuffledSource
        return RandomShuffledSource(self, self.to_pairs())

    @safe_mode('train')
    def shuffled_seqs_source(self, maxlen: Optional[int] = None):
        r"""
        To random shuffled (User, ISeq) source.

        Parameters:
        -----------
        maxlen: int, optional
            Maximum length
            `None`: return the whole sequence
            `int`: return the recent `maxlen` items

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().to_shuffled_seqs()
        >>> len(datapipe) == dataset[USER, ID].count
        True
        >>> list(datapipe)[0].keys()
        dict_keys([Field(USER:ID,USER), Field(ITEM:ID,ITEM,SEQUENCE)])
        """
        from ..postprocessing.source import RandomShuffledSource
        return RandomShuffledSource(self, self.to_seqs(maxlen))

    @safe_mode('train')
    def shuffled_roll_seqs_source(
        self, minlen: int = 2, maxlen: Optional[int] = None,
        keep_at_least_itself: bool = True
    ):
        r"""
        To random shuffled (User, ISeq) rolling source.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().to_shuffled_roll_seqs()
        >>> list(datapipe)[0].keys()
        dict_keys([Field(USER:ID,USER), Field(ITEM:ID,ITEM,SEQUENCE)])
        """
        from ..postprocessing.source import RandomShuffledSource
        return RandomShuffledSource(
            self, self.to_roll_seqs(minlen, maxlen, keep_at_least_itself)
        )

    def summary(self):
        super().summary()
        from prettytable import PrettyTable
        User, Item = self.fields[USER, ID], self.fields[ITEM, ID]

        table = PrettyTable(['#Users', '#Items', 'Avg.Len', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        table.add_row([
            User.count, Item.count, self.train().meanlen + 2,
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
            (self.trainsize + self.validsize + self.testsize) / (User.count * Item.count)
        ])

        infoLogger(table)


class MatchingRecDataSet(RecDataSet):
    TASK = TaskTags.MATCHING


class NextItemRecDataSet(RecDataSet):
    TASK = TaskTags.NEXTITEM