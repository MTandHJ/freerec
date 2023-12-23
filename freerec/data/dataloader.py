

import torch
# import torchdata.datapipes as dp
# from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

from .datasets import GeneralRecSet, SequentialRecSet, SessionBasedRecSet
from .postprocessing.source import OrderedIDs, OrderedSource
from .tags import USER, SESSION, ITEM, TIMESTAMP, ID, UNSEEN, SEEN, POSITIVE


__all__ = [
    'DataLoader',
    'load_gen_validpipe', 'load_gen_testpipe',
    'load_seq_lpad_validpipe', 'load_seq_lpad_testpipe',
    'load_seq_rpad_validpipe', 'load_seq_rpad_testpipe',
    'load_sess_lpad_validpipe', 'load_sess_lpad_testpipe',
    'load_sess_rpad_validpipe', 'load_sess_rpad_testpipe',
]


def _collate_fn(batch):
    return batch[0]

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, datapipe, num_workers: int = 0, **kwargs) -> None:
        super().__init__(
            dataset=datapipe, num_workers=num_workers, 
            batch_size=1, collate_fn=_collate_fn,
            **kwargs
        )


def _load_gen_datapipe(
    dataset: GeneralRecSet, mode: str,
    batch_size: int, ranking: str,
    num_negs_for_sample_based_ranking: int = 100
):
    r"""
    Load datapipe for general recommendation.

    Parameters:
    -----------
    dataset: SequentialRecSet
    mode: str, 'valid' or 'test'
    batch_size: int
    ranking: str
        `full`: datapipe for full ranking
        `pool`: datapipe for sampled-based ranking
    num_negs_for_sample_based_ranking: int
        The number of negatives for sample-based ranking.

    Raises:
    -------
    AssertionError: invalid `mode` received
    NotImplementedError: `ranking' is out of 'full' or 'pool'
    """
    
    assert mode in ('valid', 'test'), "`mode' should be 'valid' or 'test' ..."

    User = dataset.fields[USER, ID]
    Item = dataset.fields[ITEM, ID]

    if ranking == 'full':
        datapipe = OrderedIDs(
            field=User
        ).sharding_filter().__getattr__(f"gen_{mode}_yielding_")(
            dataset # return (user, unseen, seen)
        ).batch(batch_size).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif ranking == 'pool':
        datapipe = OrderedSource(
            source=getattr(dataset, mode)().to_pairs()
        ).sharding_filter().__getattr__(f"gen_{mode}_sampling_")(
            dataset, num_negs_for_sample_based_ranking # return (user, pool)
        ).batch(batch_size).column_().tensor_()
    else:
        raise NotImplementedError(f"{ranking} ranking is not supported ...")
    
    return datapipe

def load_gen_validpipe(
    dataset: SequentialRecSet,
    batch_size: int = 512, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_gen_datapipe(
        dataset, mode='valid', 
        batch_size=batch_size, ranking=ranking,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def load_gen_testpipe(
    dataset: SequentialRecSet,
    batch_size: int = 512, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_gen_datapipe(
        dataset, mode='test', 
        batch_size=batch_size, ranking=ranking,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )


def _load_seq_datapipe(
    dataset: SequentialRecSet, mode: str,
    maxlen: int, batch_size: int, ranking: str,
    padding_way, NUM_PADS: int, padding_value: int,
    num_negs_for_sample_based_ranking: int = 100
):
    r"""
    Load datapipe for sequential recommendation.

    Parameters:
    -----------
    dataset: SequentialRecSet
    mode: str, 'valid' or 'test'
    maxlen: int
    batch_size: int
    ranking: str
        `full`: datapipe for full ranking
        `pool`: datapipe for sampled-based ranking
    padding_way: str
        `left`: sequence will be padded from left
    NUM_PADS: int
    padding_value: int
    num_negs_for_sample_based_ranking: int
        The number of negatives for sample-based ranking.

    Raises:
    -------
    AssertionError: invalid `mode` or `padding_way` received
    NotImplementedError: `ranking' is out of 'full' or 'pool'
    """
    
    assert mode in ('valid', 'test'), "`mode' should be 'valid' or 'test' ..."
    assert padding_way in ('left', 'right'), "`padding_way` should be 'left' or 'right' ..."

    padding_way = 'lpad_' if padding_way == 'left' else 'rpad_'
    User = dataset.fields[USER, ID]
    Item = dataset.fields[ITEM, ID]

    if ranking == 'full':
        datapipe = OrderedIDs(
            field=User
        ).sharding_filter().__getattr__(f"seq_{mode}_yielding_")(
            dataset # yielding (user, seq, unseen, seen)
        ).lprune_(
            indices=[1], maxlen=maxlen,
        ).add_(
            indices=[1], offset=NUM_PADS
        ).__getattr__(padding_way)(
            indices=[1], maxlen=maxlen, padding_value=padding_value
        ).batch(batch_size).column_().tensor_().field_(
            User.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif ranking == 'pool':
        datapipe = OrderedIDs(
            field=User
        ).sharding_filter().__getattr__(f"seq_{mode}_sampling_")(
            dataset, num_negs_for_sample_based_ranking # yielding (user, seq, (target + (100) negatives))
        ).lprune_(
            indices=[1], maxlen=maxlen,
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).__getattr__(padding_way)(
            indices=[1], maxlen=maxlen, padding_value=0
        ).batch(batch_size).column_().tensor_()
    else:
        raise NotImplementedError(f"{ranking} ranking is not supported ...")
    
    return datapipe

def load_seq_lpad_validpipe(
    dataset: SequentialRecSet,
    maxlen: int, NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 128, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_seq_datapipe(
        dataset, mode='valid', 
        maxlen=maxlen, batch_size=batch_size, ranking=ranking, 
        padding_way='left', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def load_seq_rpad_validpipe(
    dataset: SequentialRecSet,
    maxlen: int, NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 128, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_seq_datapipe(
        dataset, mode='valid', 
        maxlen=maxlen, batch_size=batch_size, ranking=ranking, 
        padding_way='right', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def load_seq_lpad_testpipe(
    dataset: SequentialRecSet,
    maxlen: int, NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 128, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_seq_datapipe(
        dataset, mode='test', 
        maxlen=maxlen, batch_size=batch_size, ranking=ranking, 
        padding_way='left', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def load_seq_rpad_testpipe(
    dataset: SequentialRecSet,
    maxlen: int, NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 128, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_seq_datapipe(
        dataset, mode='test', 
        maxlen=maxlen, batch_size=batch_size, ranking=ranking, 
        padding_way='right', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def _load_sess_datapipe(
    dataset: SessionBasedRecSet, mode: str,
    batch_size: int, ranking: str,
    padding_way, NUM_PADS: int, padding_value: int,
    num_negs_for_sample_based_ranking: int = 100
):
    r"""
    Load datapipe for session-based recommendation.

    Parameters:
    -----------
    dataset: SessionBasedRecSet
    mode: str, 'valid' or 'test'
    batch_size: int
    ranking: str
        `full`: datapipe for full ranking
        `pool`: datapipe for sampled-based ranking
    padding_way: str
        `left`: sequence will be padded from left
    NUM_PADS: int
    padding_value: int
    num_negs_for_sample_based_ranking: int
        The number of negatives for sample-based ranking

    Raises:
    -------
    AssertionError: invalid `mode` or `padding_way` received
    NotImplementedError: `ranking' is out of 'full' or 'pool'
    """
    
    assert mode in ('valid', 'test'), "`mode' should be 'valid' or 'test' ..."
    assert padding_way in ('left', 'right'), "`padding_way` should be 'left' or 'right' ..."

    padding_way = 'lpad_' if padding_way == 'left' else 'rpad_'
    Session = dataset.fields[SESSION, ID]
    Item = dataset.fields[ITEM, ID]

    if ranking == 'full':
        datapipe = OrderedSource(
            source=getattr(dataset, mode)().to_roll_seqs(minlen=2)
        ).sharding_filter().__getattr__(f"sess_{mode}_yielding_")(
            dataset # yielding (sesses, seq, unseen, seen)
        ).add_(
            indices=[1], offset=NUM_PADS
        ).batch(batch_size).column_().__getattr__(f"{padding_way}col_")(
            indices=[1], maxlen=None, padding_value=padding_value
        ).tensor_().field_(
            Session.buffer(), Item.buffer(tags=POSITIVE), Item.buffer(tags=UNSEEN), Item.buffer(tags=SEEN)
        )
    elif ranking == 'pool':
        datapipe = OrderedSource(
            source=getattr(dataset, mode)().to_roll_seqs(minlen=2)
        ).sharding_filter().__getattr__(f"sess_{mode}_sampling_")(
            dataset, num_negs_for_sample_based_ranking # yielding (sesses, seq, (target + (100) negatives))
        ).add_(
            indices=[1, 2], offset=NUM_PADS
        ).batch(batch_size).column_().__getattr__(f"{padding_way}col_")(
            indices=[1], maxlen=None, padding_value=padding_value
        ).tensor_()
    else:
        raise NotImplementedError(f"{ranking} ranking is not supported ...")
    
    return datapipe

def load_sess_lpad_validpipe(
    dataset: SessionBasedRecSet,
    NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 256, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_sess_datapipe(
        dataset, mode='valid', 
        batch_size=batch_size, ranking=ranking, 
        padding_way='left', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def load_sess_rpad_validpipe(
    dataset: SessionBasedRecSet,
    NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 256, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_sess_datapipe(
        dataset, mode='valid', 
        batch_size=batch_size, ranking=ranking, 
        padding_way='right', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def load_sess_lpad_testpipe(
    dataset: SessionBasedRecSet,
    NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 256, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_sess_datapipe(
        dataset, mode='test', 
        batch_size=batch_size, ranking=ranking, 
        padding_way='left', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )

def load_sess_rpad_testpipe(
    dataset: SessionBasedRecSet,
    NUM_PADS: int, padding_value: int = 0,
    batch_size: int = 256, ranking: str = 'full',
    num_negs_for_sample_based_ranking: int = 100
):
    return _load_sess_datapipe(
        dataset, mode='test', 
        batch_size=batch_size, ranking=ranking, 
        padding_way='right', NUM_PADS=NUM_PADS, padding_value=padding_value,
        num_negs_for_sample_based_ranking=num_negs_for_sample_based_ranking
    )