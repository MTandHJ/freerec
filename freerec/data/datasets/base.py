

from typing import Any, TypeVar, Literal, Union, Optional, Callable, Iterator, Iterable, Dict, Tuple, List

import torch, os, abc, glob, yaml
import numpy as np
import polars as pl
import torchdata.datapipes as dp
from copy import copy
from functools import lru_cache

from ..tags import (
    FieldTags, TaskTags, 
    USER, ITEM, LABEL, ID, RATING, TIMESTAMP, FEATURE, SEQUENCE
)
from ..fields import Field, FieldTuple
from ..utils import download_from_url, extract_archive, is_empty_dir, check_sha1
from ...utils import (
    timemeter, infoLogger, warnLogger, 
    mkdirs, import_pickle, export_pickle, import_yaml
)


__all__ = [
    'BaseSet', 'RecDataSet', 
    'MatchingRecDataSet', 'NextItemRecDataSet', 'PredictionRecDataSet'
]


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
    tasktag: Tasktag
        Tasktag for subsequent sampling.
    cfg: config
        - `None`: search the config file in default path
        - `str`: the path to config file
        - `Dict`: config
    """

    DEFAULT_FIELD_BUILDER = {
        USER.name: Field(USER.name, USER, ID),
        ITEM.name: Field(ITEM.name, ITEM, ID),
        LABEL.name: Field(LABEL.name, LABEL),
        RATING.name: Field(RATING.name, RATING),
        TIMESTAMP.name: Field(TIMESTAMP.name, TIMESTAMP)
    }

    DEFAULT_FIELD_CONFIG = {
        'tags': tuple(),
        'dtype': None,
        'fill_null_strategy': 'zero',
        'normalizer': None,
    }

    DEFAULT_CSV_FILE = "{mode}.txt"
    DEFAULT_CSV_SEPARATOR = "\t"
    DEFAULT_SCHEMA_FILE = "schema.pkl"
    DEFAULT_PARQUET_DIR = "parquet/{mode}"
    DEFAULT_PARQUET_FILE = "p{chunk}.parquet"
    DEFAULT_CHUNK_SIZE = 256 * 512
    DEFAULT_CONFIG_FILE = "config.yaml"

    TASK: TaskTags
    URL: Optional[str] = None

    def __init__(
        self, 
        root: str, 
        filedir: Optional[str] = None, 
        *,
        download: bool = True,
        tasktag: Optional[TaskTags] = None,
        cfg: Union[None, str, Dict] = None
    ) -> None:
        super().__init__()

        self.fields = []
        self.__mode: Literal['train', 'valid', 'test'] = 'train'
        if tasktag is not None:
            self.TASK = tasktag

        filedir = filedir if filedir else self.__class__.__name__
        self.path = os.path.join(root, 'Processed', filedir)
        if is_empty_dir(self.path):
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

        self.set_config(cfg)
        self.compile()
        self.check()

    def set_config(self, cfg: Union[None, str, Dict] = None):
        """Set config for fields."""
        if cfg is None:
            cfg = os.path.join(self.path, self.DEFAULT_CONFIG_FILE)
            if os.path.exists(cfg):
                cfg = import_yaml(cfg)
            else:
                cfg = dict()
        elif isinstance(cfg, str):
            cfg = import_yaml(cfg)
        
        self.cfg: Dict[str, Dict] = {
            field_name.upper(): self.DEFAULT_FIELD_CONFIG | field_cfg
            for field_name, field_cfg in cfg.items()
        }

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
    def fields(self) -> FieldTuple[Field]:
        """Return a tuple of Field."""
        return self.__fields

    @fields.setter
    def fields(self, fields: Iterable[Field]):
        """Set fields."""
        self.__fields = FieldTuple(fields)

    def build_fields(self, columns: Iterable[str], *tags: FieldTags) -> FieldTuple[Field]:
        fields = []
        for colname in columns:
            field_cfg = self.cfg.get(colname, self.DEFAULT_FIELD_CONFIG).copy()
            private_tags = [FieldTags(tag) for tag in field_cfg.pop('tags')]
            field = self.DEFAULT_FIELD_BUILDER.get(
                colname,
                Field(colname, FEATURE)
            ).fork(*tags, *private_tags)
            field.set_normalizer(**field_cfg)
            fields.append(field)
        return FieldTuple(fields)

    def parquet(
        self, fields: Iterable[Field]
    ) -> Iterator[pl.DataFrame]:
        path = os.path.join(
            self.path, 
            self.DEFAULT_PARQUET_DIR.format(mode=self.mode),
            self.DEFAULT_PARQUET_FILE
        )
        columns = [field.name for field in fields]
        num_chunks = len(glob.glob(path.format(chunk='*')))
        for k in range(num_chunks):
            parquet_file = path.format(chunk=k)
            yield pl.read_parquet(
                parquet_file, columns=columns
            )

    def load_inter(self):
        r"""
        Load interaction data.

        Flows:
        ------
        1. Traversing and fitting train|valid|test sets.
        2. Normalizing and splitting train|valid|test sets into parquet chunks.
        """
        schema_file = os.path.join(self.path, self.DEFAULT_SCHEMA_FILE)
        sha1_hash = check_sha1(
            yaml.dump(self.cfg).encode()
        )
        try:
            if os.path.exists(schema_file):
                infoLogger(f"[DataSet] >>> Load Schema from {schema_file} ...")
                schema = import_pickle(schema_file)
                if sha1_hash != schema.get('sha1_hash', ''):
                    infoLogger(f"[DataSet] >>> Schema's sha1 hash value is not matched ...")
                    raise RecSetBuildingError
            else:
                raise RecSetBuildingError
        except RecSetBuildingError:
            schema = {
                'fields': None,
                'sha1_hash': sha1_hash,
                'trainsize': 0,
                'validsize': 0,
                'testsize': 0,
            }
            # fitting fields over csv files
            for mode in ('train', 'valid', 'test'):
                infoLogger(f"[DataSet] >>> Fitting fields over `{mode}` set ...")
                df = pl.read_csv(
                    os.path.join(
                        self.path,
                        self.DEFAULT_CSV_FILE.format(mode=mode)
                    ),
                    separator=self.DEFAULT_CSV_SEPARATOR
                )
                schema[mode + 'size'] = df.height
                if schema['fields'] is None:
                    schema['fields'] = self.build_fields(
                        df.columns
                    )

                for field in schema['fields']:
                    field.fit(
                        df.select(pl.col(field.name)),
                        partial=True
                    )

            # transforming and splitting csv files into parquet formats
            for mode in ('train', 'valid', 'test'):
                infoLogger(f"[DataSet] >>> Normalizing fields over `{mode}` set ...")
                path = os.path.join(
                    self.path, 
                    self.DEFAULT_PARQUET_DIR.format(mode=mode)
                )
                mkdirs(path)

                df = pl.read_csv(
                    os.path.join(
                        self.path,
                        self.DEFAULT_CSV_FILE.format(mode=mode)
                    ),
                    separator=self.DEFAULT_CSV_SEPARATOR
                )

                for k, chunk in enumerate(df.iter_slices(self.DEFAULT_CHUNK_SIZE)):
                    chunk = chunk.with_columns(
                        field.normalize(
                            chunk.select(pl.col(field.name))
                        )
                        for field in schema['fields']
                    )
                    chunk.write_parquet(
                        os.path.join(
                            path,
                            self.DEFAULT_PARQUET_FILE.format(chunk=k)
                        )
                    )
            export_pickle(schema, schema_file)
        finally:
            # .fork() for consistent hash value
            self.fields = [field.fork() for field in schema['fields']]
            self.trainsize = schema['trainsize']
            self.validsize = schema['validsize']
            self.testsize = schema['testsize']

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

    def match_not(self: T, *tags: FieldTags) -> T:
        r"""
        Return a copy of dataset with fields matching any given tags.

        Examples:
        ---------
        >>> dataset: BaseSet
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM|RATING:RATING|TIMESTAMP:TIMESTAMP)
        >>> dataset.match_not(TIMESTAMP)
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM)
        >>> dataset.match_not()
        RecDataSet()
        """
        dataset = copy(self)
        dataset.fields = self.fields.match_not(*tags)
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
    def to_rows(cls, coldata: Dict[Field, Iterable[T]]) -> List[Dict[Field, T]]:
        fields = coldata.keys()
        return cls.listmap(
            lambda values: dict(zip(fields, values)),
            zip(*coldata.values())
        )

    def __repr__(self) -> str:
        cfg = '|'.join(map(str, self.fields))
        return f"{self.__class__.__name__}({cfg})"

    def __str__(self) -> str:
        cfg = ' | '.join(map(str, self.fields))
        return f"[{self.__class__.__name__}] >>> " + cfg

    def __getitem__(self, fields: Union[Field, Iterable[Field]]) -> Optional[Dict[Field, List]]:
        r"""
        Obtain column data according to `fields`.

        Notes:
        ------
        It is expensive if the dataset is very large.
        """
        if isinstance(fields, Field):
            fields = (fields,)
        if len(fields) == 0:
            return None
        else:
            data = pl.concat(
                self.parquet(fields),
                how='vertical'
            )
            return {field: data[field.name].to_list() for field in fields}

    def __iter__(self) -> Iterator[Dict[Field, Any]]:
        for df in self.parquet(self.fields):
            for row in df.iter_rows():
                yield dict(zip(self.fields, row))


class RecDataSet(BaseSet):
    """RecDataSet provides a template for specific datasets."""

    def to_pairs(self) -> List[Dict[Field, int]]:
        """Return (User, Item) in pairs."""
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        return self.to_rows(self[User, Item])

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
        return seqlens

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
        r"""
        Check whether the dataset has repeated interactions.
        This will be used in evaluation if `seen` items should be removed.

        Notes:
        ------
        Return `False` for some CTR datasets lacking of (User, Item) fields.
        """
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        if User is not None and Item is not None:
            ISeq = Item.fork(SEQUENCE)
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
        data = self[nodes]
        for key in data:
            data[key] = torch.tensor(data[key], dtype=torch.long)

        graph = HeteroData()
        for node in srcs:
            graph[node.name].x = torch.empty((node.count, 0), dtype=torch.long)
        for node in dsts:
            if node not in srcs:
                graph[node.name].x = torch.empty((node.count, 0), dtype=torch.long)
        for src, edge, dst in zip(srcs, edges, dsts):
            u, v = data[src], data[dst]
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

    @safe_mode('valid', 'test')
    def ordered_inter_source(self):
        r"""
        To ordered [Label, Feature1, Feature2, ...] source.

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.valid().ordered_inter_source()
        >>> list(datapipe)[0].keys()
        dict_keys([Field(LABEL:LABEL), Field(USER:ID,USER), Field(ITEM:ID,ITEM), ...])
        """
        from ..postprocessing.source import PipedSource
        return PipedSource(
            self, self
        )

    @safe_mode('train')
    def shuffled_inter_source(
        self, buffer_size: Optional[int] = None
    ):
        r"""
        To shuffled [Label, Feature1, Feature2, ...] source.

        Parameters:
        -----------
        buffer_size: int, optional
            - `None`: use the default chunk size instead

        Examples:
        ---------
        >>> dataset: RecDataSet
        >>> datapipe = dataset.train().shuffled_inter_source()
        >>> list(datapipe)[0].keys()
        dict_keys([Field(LABEL:LABEL), Field(USER:ID,USER), Field(ITEM:ID,ITEM), ...])
        """
        buffer_size = self.DEFAULT_CHUNK_SIZE if buffer_size is None else buffer_size
        from ..postprocessing.source import PipedSource
        return PipedSource(
            self, self.shuffle(buffer_size=buffer_size)
        )


class MatchingRecDataSet(RecDataSet):
    TASK = TaskTags.MATCHING

    def summary(self):
        super().summary()
        from prettytable import PrettyTable
        User, Item = self.fields[USER, ID], self.fields[ITEM, ID]

        table = PrettyTable(['#Users', '#Items', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        table.add_row([
            User.count, Item.count,
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
            (self.trainsize + self.validsize + self.testsize) / (User.count * Item.count)
        ])

        infoLogger(table)


class NextItemRecDataSet(RecDataSet):
    TASK = TaskTags.NEXTITEM

    def summary(self):
        super().summary()
        from prettytable import PrettyTable
        User, Item = self.fields[USER, ID], self.fields[ITEM, ID]

        table = PrettyTable(['#Users', '#Items', 'Avg.Len', '#Interactions', '#Train', '#Valid', '#Test', 'Density'])
        table.add_row([
            User.count, Item.count, self.train().meanlen + self.valid().meanlen + self.test().meanlen,
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
            (self.trainsize + self.validsize + self.testsize) / (User.count * Item.count)
        ])

        infoLogger(table)


class PredictionRecDataSet(RecDataSet):
    TASK = TaskTags.PREDICTION

    def summary(self):
        super().summary()
        from prettytable import PrettyTable

        table = PrettyTable(['#Interactions', '#Train', '#Valid', '#Test'])
        table.add_row([
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
        ])

        infoLogger(table)