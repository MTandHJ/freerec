


from typing import Any, TypeVar, Literal, Union, Optional, Callable, Iterator, Iterable, Dict, Tuple, List

import torch, os, abc, glob, yaml, random
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
    r"""Decorator that warns when a method runs outside the expected mode(s).

    Parameters
    ----------
    *modes : str
        Allowed dataset modes (e.g., ``'train'``, ``'valid'``, ``'test'``).

    Returns
    -------
    callable
        Decorated function that emits a warning when invoked in an
        unexpected mode.
    """
    def decorator(func):
        r"""Wrap *func* with a mode check."""
        def wrapper(self, *args, **kwargs):
            r"""Call the wrapped function after an optional mode warning."""
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

class RecSetBuildingError(Exception):
    r"""Raised when a :class:`~BaseSet` schema cannot be loaded or built."""
    ...


class BaseSet(dp.iter.IterDataPipe, metaclass=abc.ABCMeta):
    r"""Base class for recommendation dataset pipes.

    Provides common functionality for loading, splitting, normalizing,
    and iterating over recommendation datasets stored in CSV/chunk
    format on disk.

    Parameters
    ----------
    root : str
        Root directory for dataset storage.
    filedir : str or None, optional
        Directory name of the dataset under ``root/Processed/``. If
        ``None``, the class name is used.
    download : bool, optional
        Whether to download the dataset from :pyattr:`URL` when the
        local path is missing.
    tasktag : :class:`~TaskTags` or None, optional
        Task tag that determines downstream sampling behavior.
    cfg : None, str, or dict, optional
        Field configuration. ``None`` searches for a default config
        file; a str is interpreted as a YAML path; a dict is used
        directly.
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
    DEFAULT_CHUNK_DIR = "chunks/{mode}"
    DEFAULT_CHUNK_FILE = "p{chunk}.pkl"
    DEFAULT_CHUNK_SIZE = 256 * 512
    DEFAULT_CONFIG_FILE = "config.yaml"
    STREAMING: bool = True # if `False`, iter(dataset) will shuffle the saved chunks during training.

    TASK: TaskTags
    URL: Optional[str] = None

    def __init__(
        self,
        root: str,
        filedir: Optional[str] = None,
        *,
        download: bool = True,
        tasktag: Optional[TaskTags] = None,
        cfg: Union[None, str, Dict] = None,
    ) -> None:
        r"""Initialize the dataset, download if needed, and compile."""
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

        self.rng = random.Random()
        self.set_seed(0)

        self.set_config(cfg)
        self.compile()
        self.check()

    def set_config(self, cfg: Union[None, str, Dict] = None):
        r"""Load and apply field configuration.

        Parameters
        ----------
        cfg : None, str, or dict, optional
            ``None`` looks for a default config file; a str is a YAML
            path; a dict is used directly.
        """
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
        r"""Run self-check logic. Override in subclasses if needed."""
        ...

    @property
    def mode(self) -> Literal['train', 'test', 'valid']:
        r"""Return the current dataset mode."""
        return self.__mode

    def train(self: T) -> T:
        r"""Switch the dataset mode to ``'train'``."""
        self.__mode = 'train'
        return self

    def valid(self: T) -> T:
        r"""Switch the dataset mode to ``'valid'``."""
        self.__mode = 'valid'
        return self

    def test(self: T) -> T:
        r"""Switch the dataset mode to ``'test'``."""
        self.__mode = 'test'
        return self

    @property
    def datasize(self):
        r"""Return the number of interactions for the current mode."""
        if self.mode == 'train':
            return self.trainsize
        elif self.mode == 'valid':
            return self.validsize
        else:
            return self.testsize

    @property
    def fields(self) -> FieldTuple[Field]:
        r"""Return the fields as a :class:`~FieldTuple`."""
        return self.__fields

    @fields.setter
    def fields(self, fields: Iterable[Field]):
        r"""Set the dataset fields."""
        self.__fields = FieldTuple(fields)

    def build_fields(self, columns: Iterable[str], *tags: FieldTags) -> FieldTuple[Field]:
        r"""Build :class:`~Field` objects from column names and optional tags.

        Parameters
        ----------
        columns : iterable of str
            Column names from the CSV header.
        *tags : :class:`~FieldTags`
            Additional tags to attach to every field.

        Returns
        -------
        :class:`~FieldTuple`
            Constructed field tuple.
        """
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

    def set_seed(self, seed: int):
        r"""Set the random seed for chunk shuffling.

        Parameters
        ----------
        seed : int
            Random seed value.
        """
        self.rng.seed(seed)

    def read_chunk(
        self,
        fields: Iterable[Field],
        streaming: bool = True
    ) -> Iterator[pl.DataFrame]:
        r"""Yield chunks of data for the current mode.

        Parameters
        ----------
        fields : iterable of :class:`~Field`
            Fields to include in each yielded chunk.
        streaming : bool, optional
            If ``True``, chunks are yielded in order. If ``False``,
            chunks are shuffled before yielding.

        Yields
        ------
        dict
            Mapping from :class:`~Field` to column data for one chunk.
        """
        path = os.path.join(
            self.path,
            self.DEFAULT_CHUNK_DIR.format(mode=self.mode),
            self.DEFAULT_CHUNK_FILE
        )
        num_chunks = len(glob.glob(path.format(chunk='*')))
        chunks = list(range(num_chunks))
        if not streaming:
            self.rng.shuffle(chunks)
        for k in chunks:
            chunk_file = path.format(chunk=k)
            data = import_pickle(chunk_file)
            yield {field: data[field.name] for field in fields}

    def load_inter(self):
        r"""Load interaction data, fitting fields and creating chunks.

        On first call (or when the config hash changes), this method
        traverses and fits fields over train/valid/test CSV files, then
        normalizes and persists them as chunked pickle files.
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

            # transforming and splitting csv files into chunk formats
            for mode in ('train', 'valid', 'test'):
                infoLogger(f"[DataSet] >>> Normalizing fields over `{mode}` set ...")
                path = os.path.join(
                    self.path,
                    self.DEFAULT_CHUNK_DIR.format(mode=mode)
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
                    export_pickle(
                        chunk.to_dict(as_series=False),
                        os.path.join(
                            path,
                            self.DEFAULT_CHUNK_FILE.format(chunk=k)
                        )
                    )
            export_pickle(schema, schema_file)
        # .fork() for consistent hash value
        self.fields = [field.fork() for field in schema['fields']]
        self.trainsize = schema['trainsize']
        self.validsize = schema['validsize']
        self.testsize = schema['testsize']

    def summary(self):
        r"""Print a summary of the dataset."""
        infoLogger(str(self))

    @timemeter
    def compile(self):
        r"""Load interactions, print summary, and switch to train mode."""
        self.load_inter()
        self.summary()
        self.train()

    def match_all(self: T, *tags: FieldTags) -> T:
        r"""Return a shallow copy keeping only fields that match all *tags*.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags that every retained field must possess.

        Returns
        -------
        :class:`~BaseSet`
            Copied dataset with filtered fields.

        Examples
        --------
        >>> dataset.match_all(ID)
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM)
        """
        dataset = copy(self)
        dataset.fields = self.fields.match_all(*tags)
        return dataset

    def match_any(self: T, *tags: FieldTags) -> T:
        r"""Return a shallow copy keeping fields that match any of *tags*.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags; a field is retained if it has at least one.

        Returns
        -------
        :class:`~BaseSet`
            Copied dataset with filtered fields.

        Examples
        --------
        >>> dataset.match_any(ID, TIMESTAMP)
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM|TIMESTAMP:TIMESTAMP)
        """
        dataset = copy(self)
        dataset.fields = self.fields.match_any(*tags)
        return dataset

    def match_not(self: T, *tags: FieldTags) -> T:
        r"""Return a shallow copy excluding fields that have any of *tags*.

        Parameters
        ----------
        *tags : :class:`~FieldTags`
            Tags to exclude.

        Returns
        -------
        :class:`~BaseSet`
            Copied dataset with filtered fields.

        Examples
        --------
        >>> dataset.match_not(TIMESTAMP)
        RecDataSet(USER:ID,USER|ITEM:ID,ITEM)
        """
        dataset = copy(self)
        dataset.fields = self.fields.match_not(*tags)
        return dataset

    @staticmethod
    def listmap(func: Callable, *iterables) -> List:
        r"""Apply *func* to *iterables* and collect results into a list.

        Parameters
        ----------
        func : callable
            Function to apply element-wise.
        *iterables
            One or more iterables whose elements are passed to *func*.

        Returns
        -------
        list
            Collected results.
        """
        return list(map(func, *iterables))

    @classmethod
    def to_rows(cls, coldata: Dict[Field, Iterable[T]]) -> List[Dict[Field, T]]:
        r"""Transpose column-oriented data into row-oriented dicts.

        Parameters
        ----------
        coldata : dict
            Mapping from :class:`~Field` to iterable of values.

        Returns
        -------
        list of dict
            Each dict maps the same fields to a single value per row.
        """
        fields = coldata.keys()
        return cls.listmap(
            lambda values: dict(zip(fields, values)),
            zip(*coldata.values())
        )

    def __repr__(self) -> str:
        r"""Return a compact string representation."""
        cfg = '|'.join(map(str, self.fields))
        return f"{self.__class__.__name__}({cfg})"

    def __str__(self) -> str:
        r"""Return a human-readable string representation."""
        cfg = ' | '.join(map(str, self.fields))
        return f"[{self.__class__.__name__}] >>> " + cfg

    def __getitem__(self, fields: Union[Field, Iterable[Field]]) -> Optional[Dict[Field, List]]:
        r"""Retrieve full column data for the given fields.

        Parameters
        ----------
        fields : :class:`~Field` or iterable of :class:`~Field`
            Field(s) to retrieve.

        Returns
        -------
        dict or None
            Mapping from each field to its full column data, or ``None``
            if *fields* is empty.

        Notes
        -----
        This is expensive for large datasets because all chunks are
        loaded into memory.
        """
        if isinstance(fields, Field):
            fields = (fields,)
        if len(fields) == 0:
            return None
        else:
            data = {field: [] for field in fields}
            for chunk in self.read_chunk(fields, streaming=True):
                for field in fields:
                    data[field].extend(chunk[field])
            return data

    def __iter__(self) -> Iterator[Dict[Field, Any]]:
        r"""Iterate over the dataset row by row for the current mode."""
        for chunk in self.read_chunk(
            self.fields,
            streaming=self.STREAMING or (self.mode != 'train')
        ):
            yield from self.to_rows(chunk)


class RecDataSet(BaseSet):
    r"""Recommendation dataset that adds sequence and graph utilities."""

    def to_pairs(self) -> List[Dict[Field, int]]:
        r"""Return all (User, Item) interactions as row dicts.

        Returns
        -------
        list of dict
            Each dict maps the User and Item fields to integer IDs.
        """
        User = self.fields[USER, ID]
        Item = self.fields[ITEM, ID]
        return self.to_rows(self[User, Item])

    def to_seqs(self, maxlen: Optional[int] = None) -> List[Dict[Field, Union[int, Tuple[int]]]]:
        r"""Group interactions into per-user item sequences.

        Parameters
        ----------
        maxlen : int or None, optional
            If not ``None``, truncate each sequence to the most recent
            *maxlen* items.

        Returns
        -------
        list of dict
            Each dict maps the User field to a user ID and the Item
            (sequence) field to a tuple of item IDs.
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
        r"""Generate rolling (expanding) sub-sequences per user.

        Parameters
        ----------
        minlen : int, optional
            Minimum sub-sequence length to emit.
        maxlen : int or None, optional
            Maximum length of the base sequence before rolling. If
            ``None``, use the full sequence.
        keep_at_least_itself : bool, optional
            If ``True``, keep sequences shorter than *minlen* as-is.

        Returns
        -------
        list of dict
            Row dicts with User and Item (sequence) fields.
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
        r"""Return the sequence length for every user.

        Returns
        -------
        list of int
            Per-user sequence lengths.
        """
        ISeq = self.fields[ITEM, ID].fork(SEQUENCE)
        seqlens = [len(row[ISeq]) for row in self.to_seqs()]
        return seqlens

    @property
    def maxlen(self) -> int:
        r"""Return the maximum sequence length across all users."""
        return np.max(self.seqlens()).item()

    @property
    def minlen(self) -> int:
        r"""Return the minimum sequence length across all users."""
        return np.min(self.seqlens()).item()

    @property
    def meanlen(self) -> int:
        r"""Return the mean sequence length across all users."""
        return np.mean(self.seqlens()).item()

    @lru_cache()
    def has_duplicates(self) -> bool:
        r"""Check whether any user has duplicate items across splits.

        Returns
        -------
        bool
            ``True`` if at least one user has a repeated item across
            the train, valid, and test sequences.

        Notes
        -----
        Returns ``False`` for CTR datasets that lack User or Item
        fields.
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
        r"""Convert the dataset to a heterogeneous graph.

        Parameters
        ----------
        *edge_types : tuple
            Each element is ``(source_tags, edge_name, dest_tags)`` where
            *source_tags* and *dest_tags* are tuples of
            :class:`~FieldTags` and *edge_name* is an optional str
            (``None`` auto-generates the name).

        Returns
        -------
        :class:`torch_geometric.data.HeteroData`
            The constructed heterogeneous graph.

        Notes
        -----
        A warning is raised if the current mode is not ``'train'``.

        Examples
        --------
        >>> graph = dataset.to_heterograph(
        ...     ((USER, ID), None, (ITEM, ID)),
        ...     ((ITEM, ID), None, (USER, ID)),
        ... )
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
        r"""Convert the dataset to a bipartite graph.

        Parameters
        ----------
        src : tuple of :class:`~FieldTags`, optional
            Source node tags.
        dst : tuple of :class:`~FieldTags`, optional
            Destination node tags.
        edge_type : str or None, optional
            Edge name. If ``None``, derived from field names.

        Returns
        -------
        :class:`torch_geometric.data.HeteroData`
            The bipartite graph.

        Notes
        -----
        A warning is raised if the current mode is not ``'train'``.

        Examples
        --------
        >>> graph = dataset.to_bigraph((USER, ID), (ITEM, ID))
        """
        return self.to_heterograph((src, edge_type, dst))

    def to_graph(
        self,
        src: Tuple[FieldTags] = (USER, ID),
        dst: Tuple[FieldTags] = (ITEM, ID)
    ):
        r"""Convert the dataset to an undirected homogeneous graph.

        Parameters
        ----------
        src : tuple of :class:`~FieldTags`, optional
            Source node tags.
        dst : tuple of :class:`~FieldTags`, optional
            Destination node tags.

        Returns
        -------
        :class:`torch_geometric.data.Data`
            The homogeneous graph with undirected edges.

        Notes
        -----
        A warning is raised if the current mode is not ``'train'``.

        Examples
        --------
        >>> graph = dataset.to_graph((USER, ID), (ITEM, ID))
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
        r"""Convert the dataset to a normalized adjacency matrix.

        Parameters
        ----------
        src : tuple of :class:`~FieldTags`, optional
            Source node tags.
        dst : tuple of :class:`~FieldTags`, optional
            Destination node tags.
        normalization : str, optional
            Normalization type:

            - ``'sym'`` -- Symmetric: :math:`D_l^{-1/2} A D_r^{-1/2}`
            - ``'left'`` -- Left: :math:`D_l^{-1} A`
            - ``'right'`` -- Right: :math:`A D_r^{-1}`

        Returns
        -------
        :class:`torch.Tensor`
            Normalized adjacency matrix in CSR format.
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
        r"""Create an ordered source over all user IDs.

        Returns
        -------
        :class:`~OrderedSource`
            Data source yielding user IDs in order.

        Examples
        --------
        >>> source = dataset.valid().ordered_user_ids_source()
        """
        from ..postprocessing.source import OrderedSource
        User = self.fields[USER, ID]
        source = self.to_rows({User: list(range(User.count))})
        return OrderedSource(self, source)

    @safe_mode('train')
    def choiced_user_ids_source(self):
        r"""Create a randomly sampled source of user IDs.

        Returns
        -------
        :class:`~RandomChoicedSource`
            Data source yielding randomly chosen user IDs with size
            equal to the current dataset's interaction count.

        Examples
        --------
        >>> source = dataset.train().choiced_user_ids_source()
        """
        from ..postprocessing.source import RandomChoicedSource
        User = self.fields[USER, ID]
        source = self.to_rows({User: list(range(User.count))})
        return RandomChoicedSource(self, source)

    @safe_mode('train')
    def shuffled_pairs_source(self):
        r"""Create a shuffled source of (User, Item) pairs.

        Returns
        -------
        :class:`~RandomShuffledSource`
            Data source yielding shuffled interaction pairs.

        Examples
        --------
        >>> source = dataset.train().shuffled_pairs_source()
        >>> list(source)[0].keys()
        dict_keys([Field(USER:ID,USER), Field(ITEM:ID,ITEM)])
        """
        from ..postprocessing.source import RandomShuffledSource
        return RandomShuffledSource(self, self.to_pairs())

    @safe_mode('train')
    def shuffled_seqs_source(self, maxlen: Optional[int] = None):
        r"""Create a shuffled source of (User, ItemSequence) data.

        Parameters
        ----------
        maxlen : int or None, optional
            Maximum sequence length. ``None`` keeps the full sequence.

        Returns
        -------
        :class:`~RandomShuffledSource`
            Data source yielding shuffled user sequences.

        Examples
        --------
        >>> source = dataset.train().shuffled_seqs_source()
        >>> list(source)[0].keys()
        dict_keys([Field(USER:ID,USER), Field(ITEM:ID,ITEM,SEQUENCE)])
        """
        from ..postprocessing.source import RandomShuffledSource
        return RandomShuffledSource(self, self.to_seqs(maxlen))

    @safe_mode('train')
    def shuffled_roll_seqs_source(
        self, minlen: int = 2, maxlen: Optional[int] = None,
        keep_at_least_itself: bool = True
    ):
        r"""Create a shuffled source of rolling (User, ItemSequence) data.

        Parameters
        ----------
        minlen : int, optional
            Minimum sub-sequence length.
        maxlen : int or None, optional
            Maximum base sequence length before rolling.
        keep_at_least_itself : bool, optional
            If ``True``, keep sequences shorter than *minlen*.

        Returns
        -------
        :class:`~RandomShuffledSource`
            Data source yielding shuffled rolling sequences.

        Examples
        --------
        >>> source = dataset.train().shuffled_roll_seqs_source()
        >>> list(source)[0].keys()
        dict_keys([Field(USER:ID,USER), Field(ITEM:ID,ITEM,SEQUENCE)])
        """
        from ..postprocessing.source import RandomShuffledSource
        return RandomShuffledSource(
            self, self.to_roll_seqs(minlen, maxlen, keep_at_least_itself)
        )

    @safe_mode('valid', 'test')
    def ordered_inter_source(self):
        r"""Create an ordered source over all interactions.

        Returns
        -------
        :class:`~PipedSource`
            Data source yielding interactions in original order.

        Examples
        --------
        >>> source = dataset.valid().ordered_inter_source()
        """
        from ..postprocessing.source import PipedSource
        return PipedSource(
            self, self
        )

    @safe_mode('train')
    def shuffled_inter_source(
        self, buffer_size: Optional[int] = None
    ):
        r"""Create a shuffled source over all interactions.

        Parameters
        ----------
        buffer_size : int or None, optional
            Shuffle buffer size. If ``None``, the default chunk size is
            used.

        Returns
        -------
        :class:`~PipedSource`
            Data source yielding shuffled interactions.

        Examples
        --------
        >>> source = dataset.train().shuffled_inter_source()
        """
        buffer_size = self.DEFAULT_CHUNK_SIZE if buffer_size is None else buffer_size
        from ..postprocessing.source import PipedSource
        return PipedSource(
            self, self.shuffle(buffer_size=buffer_size)
        )


class MatchingRecDataSet(RecDataSet):
    r"""Recommendation dataset for item matching (retrieval) tasks."""
    TASK = TaskTags.MATCHING

    def summary(self):
        r"""Print a summary table with user/item/interaction statistics."""
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
    r"""Recommendation dataset for next-item prediction tasks."""
    TASK = TaskTags.NEXTITEM

    def summary(self):
        r"""Print a summary table including average sequence length."""
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
    r"""Recommendation dataset for rating/click prediction tasks."""
    TASK = TaskTags.PREDICTION
    STREAMING = False

    def summary(self):
        r"""Print a summary table of interaction counts per split."""
        super().summary()
        from prettytable import PrettyTable

        table = PrettyTable(['#Interactions', '#Train', '#Valid', '#Test'])
        table.add_row([
            self.trainsize + self.validsize + self.testsize,
            self.trainsize, self.validsize, self.testsize,
        ])

        infoLogger(table)
