

"""
Refer to
    https://zhuanlan.zhihu.com/p/67287992
for formal definitions.
The implementations below are fully due to torchmetrics.
"""

from multiprocessing import reduction
from typing import Optional, Union, List 

import torch
import torchmetrics
from freerec.utils import errorLogger


__all__ = [
    'mean_abs_error', 'mean_squared_error', 'root_mse',
    'precision', 'recall', 'f1_score', 'auroc', 'hit_rate',
    'normalized_dcg', 'mean_reciprocal_rank', 'mean_average_precision'
]

def _reduce(reduction='mean'):
    def decorator(func):
        def wrapper(
            preds: Union[List[torch.Tensor], torch.Tensor], 
            targets: Union[List[torch.Tensor], torch.Tensor], 
            reduction: str = reduction, **kwargs
        ):
            if isinstance(preds, List):
                results = torch.tensor([func(pred, target, **kwargs) for pred, target in zip(preds, targets)])
            else:
                results = func(preds, targets, **kwargs)
            if reduction == 'none':
                return results
            elif reduction == 'mean':
                return results.mean()
            elif reduction == 'sum':
                return results.sum()
            else:
                errorLogger(f"reduction should be 'none'|'mean'|'sum' but {reduction} is received ...", ValueError)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# quality of predictions ========================================================================

@_reduce('mean')
def mean_abs_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MAE.

    .. math::

        \phi(x_i, y_i) = \frac{1}{d} \|x_i - y_i\|_1.

    Parameters:
    ---

    preds: (N, d) or (N, d)
        N query predictions or a single query.
    targets: (d,) or (n, d)
        Targets in accordance with preds.
    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---
    
        torch.Tensor, (1,) or (n,)
    
    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_abs_error(preds, targets, reduction='none')
    tensor([0.5000, 0.5250])
    >>> mean_abs_error(preds, targets)
    tensor(0.5125)

    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).abs().mean(-1)

@_reduce('mean')
def mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE.

    .. math::

        \phi(x_i, y_i) = \frac{1}{d} |x_i - y_i|_2^2.

    Parameters:
    ---

    preds: (N, d) or (N, d)
        N query predictions or a single query.
    targets: (d,) or (n, d)
        Targets in accordance with preds.
    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor, (1,) or (n,)

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_squared_error(preds, targets, reduction='none')
    tensor([0.3450, 0.3475])
    >>> mean_squared_error(preds, targets)
    tensor(0.3462)

    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).pow(2).mean(-1)

@_reduce('mean')
def root_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MAE.

    .. math::

        \phi(x_i, y_i) = \sqrt{\frac{1}{d} \|x_i - y_i\|_2^2}.

    Parameters:
    ---

    preds: (N, d) or (N, d)
        N query predictions or a single query.
    targets: (d,) or (n, d)
        Targets in accordance with preds.
    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> root_mse(preds, targets, reduction='none')
    tensor([0.5874, 0.5895])
    >>> root_mse(preds, targets)
    tensor(0.5884)

    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).pow(2).mean(-1).sqrt()


# quality of the set of recommendations ========================================================================

@_reduce('mean')
def precision(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """Precision.

    .. math::

        \phi(x_i, y_i) = (|Topk(x_i) \cap Topk(y_i)|) / |Topk(x_i)|,
    
    where :math: `Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math: `i`,
    and :math: `|\cdot|` measures the cardinality of the set.

    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.
    k: int, optional
        The case of top-K.
        - `None`: k = d

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> precision(preds, targets, k=3, reduction='none')
    tensor([0.3333, 0.6667])
    >>> precision(preds, targets, k=3)
    tensor(0.5000)

    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))
    indices = preds.topk(k)[1]
    relevant = targets.gather(-1, indices).sum(-1)
    torchmetrics.functional.retrieval_precision
    return relevant / k

@_reduce('mean')
def recall(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """Recall.

    .. math::

        \phi(x_i, y_i) = (|Topk(x_i) \cap Topk(y_i)|) / |Topk(y_i)|,
    
    where :math: `Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math: `i`,
    and :math: `|\cdot|` measures the cardinality of the set.

    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.
    k: int, optional
        The case of top-K.
        - `None`: k = d

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> recall(preds, targets, k=3, reduction='none')
    tensor([0.5000, 1.0000])
    >>> recall(preds, targets, k=3)
    tensor(0.7500)

    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))
    
    indices = preds.topk(k)[1]
    relevant = targets.gather(-1, indices).sum(-1)
    total = targets.sum(-1)
    invalid = total == 0
    relevant[invalid] = 0
    relevant[~invalid] /= total[~invalid]
    return relevant

@_reduce('mean')
def f1_score(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """Recall.

    .. math::

        \phi(x_i, y_i) = (2 * Precision@k(x_i, y_i) * Recall@k(x_i, y_i)) / (Precision@k(x_i, y_i) + Recall@k(x_i, y_i)
    
    where :math: `Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math: `i`,
    and :math: `|\cdot|` measures the cardinality of the set.

    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.
    k: int, optional
        The case of top-K.
        - `None`: k = d

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> f1_score(preds, targets, k=3, reduction='none')
    tensor([0.4000, 0.8000])
    >>> f1_score(preds, targets, k=3)
    tensor(0.6000)

    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))
    precision_k = precision(preds, targets, k=k, reduction='none')
    recall_k = recall(preds, targets, k=k, reduction='none')
    score = 2 * precision_k * recall_k
    part2 = precision_k + recall_k
    valid = part2 != 0
    score[valid] /= part2[valid]
    return score

@_reduce("mean") #TODO: This implementation has not been verified.
def auroc(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """AUC.

    .. math::

        \frac{n - (|N_+| - 1) / 2 - (\sum_{v \in |N_+|} rank(v)) / |N_+|}{n - |N_+|}
    
    where :math: `n` and :math: `|N_+|` denote the number of items and positive items, respectively; :math: `rank(v)` returns the rank of the positive item :math: `v`.

    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> auroc(preds, targets, reduction='none')
    tensor([0.5000, 0.5000])
    >>> auroc(preds, targets) 
    tensor(0.5000)

    """
    preds, targets = preds.float(), targets.float()
    preds, indices = torch.sort(preds, descending=True)
    targets = targets.gather(-1, indices)
    ranks = torch.arange(1, targets.size(1) + 1, device=targets.device)
    length = targets.sum(dim=-1)
    results = targets.size(1)
    results -= (ranks * targets).sum(dim=-1).div(length)
    results -= (length - 1).div(2)
    results /= (targets.size(1) - length)
    results[torch.isnan(results)] = 0.
    return results

@_reduce('mean')
def hit_rate(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """Hit Rate.

    .. math::

        \phi(x_i, y_i) = \mathbb{I}(|Topk(x_i) \cap Topk(y_i)| \not = 0).
    
    where :math: `Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math: `i`,
    and :math: `|\cdot|` measures the cardinality of the set.

    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.
    k: int, optional
        The case of top-K.
        - `None`: k = len(preds)

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> hit_rate(preds, targets, k=3, reduction='none')
    tensor([1., 1.])
    >>> hit_rate(preds, targets, k=3)
    tensor(1.)

    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))

    indices = preds.topk(k)[1]
    relevant = targets.gather(-1, indices).sum(-1)
    return (relevant > 0).float()


# quality of the list of recommendations

def _dcg(target: torch.Tensor) -> torch.Tensor:
    """Computes Discounted Cumulative Gain for input tensor."""
    denom = torch.log2(torch.arange(target.shape[-1], device=target.device) + 2.0)
    return (target / denom).sum(dim=-1)

@_reduce('mean')
def normalized_dcg(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """NDCG.

    .. math::

        \phi(x_i, y_i) = _dcg(x_i) / _dcg(y_i).
        
    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.
    k: int, optional
        The case of top-K.
        - `None`: k = len(preds)

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor
        
    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> normalized_dcg(preds, targets, k=3, reduction='none')
    tensor([0.6131, 0.6934])
    >>> normalized_dcg(preds, targets, k=3)
    tensor(0.6533)

    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))

    indices = preds.topk(k)[1]
    sorted_target = targets.gather(-1, indices)
    ideal_target = targets.topk(k)[0]

    dcg = _dcg(sorted_target)
    ideal_dcg = _dcg(ideal_target)

    # filter undefined scores
    invalid = ideal_dcg == 0
    dcg[invalid] = 0
    dcg[~invalid] /= ideal_dcg[~invalid]
    return dcg

@_reduce('mean')
def mean_reciprocal_rank(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MRR.

    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.
    k: int, optional
        The case of top-K.
        - `None`: k = len(preds)

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_reciprocal_rank(preds, targets, reduction='none')
    tensor([1.0000, 0.5000])
    >>> mean_reciprocal_rank(preds, targets)
    tensor(0.7500)

    """
    metric = torchmetrics.functional.retrieval_reciprocal_rank
    if preds.ndim == 2:
        return torch.tensor(list(map(metric, preds, targets)))
    else:
        return metric(preds, targets)

@_reduce('mean')
def mean_average_precision(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MAP.

    Parameters:
    ---

    preds: (N, d) or (N, d), float
        N query predictions or a single query.
    targets: (d,) or (n, d), int or bool
        Targets in accordance with preds.
    k: int, optional
        The case of top-K.
        - `None`: k = len(preds)

    reduction: str, mean|sum|none
        - `mean`: Return :math: `\frac{1}{N} \sum_{i=1}^N \phi(x_i, y_i)` .
        - `sum`: Return :math: `\sum_{i=1}^N \phi(x_i, y_i)` .
        - `None`: Return :math: `[\phi_1(x_i, y_i), \ldots, \phi(x_N, y_N)]` .

    Returns:
    ---

        torch.Tensor

    Examples:
    ---

    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_average_precision(preds, targets, reduction='none')
    tensor([0.7500, 0.5833])
    >>> mean_average_precision(preds, targets)
    tensor(0.6667)
    """
    metric = torchmetrics.functional.retrieval_average_precision
    if preds.ndim == 2:
        return torch.tensor(list(map(metric, preds, targets)))
    else:
        return metric(preds, targets)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
