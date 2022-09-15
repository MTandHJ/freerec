

"""
Refer to
    https://zhuanlan.zhihu.com/p/67287992
for formal definitions.
The implementations below are fully due to torchmetrics.
"""

from typing import Optional, Union, List 

import torch
import torchmetrics
from .utils import warnLogger



__all__ = [
    'mean_abs_error', 'mean_squared_error', 'root_mse',
    'precision', 'recall', 'hit_rate', 
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
                raise ValueError(warnLogger(f"reduction should be 'none'|'mean'|'sum' but {reduction} is received ..."))
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# quality of predictions ========================================================================

@_reduce('mean')
def mean_abs_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).abs().mean(-1)

@_reduce('mean')
def mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).pow(2).mean(-1)

@_reduce('mean')
def root_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).pow(2).mean(-1).sqrt()


# quality of the set of recommendations ========================================================================

@_reduce('mean')
def precision(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        k: topK
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))
    indices = preds.topk(k)[1]
    relevant = targets.gather(-1, indices).sum(-1)
    return relevant / k

@_reduce('mean')
def recall(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        k: topK
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
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
def hit_rate(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        k: topK
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
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
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        k: topK
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
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
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
    """
    metric = torchmetrics.functional.retrieval_reciprocal_rank
    if preds.ndim == 2:
        return torch.tensor(list(map(metric, preds, targets)))
    else:
        return metric(preds, targets)

@_reduce('mean')
def mean_average_precision(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        preds: (d,) or (n, d)
        targets: (d,) or (n, d)
    Kwargs:
        reduction: mean|sum|none
    Returns:
        (1,) or (n,)
    """
    metric = torchmetrics.functional.retrieval_average_precision
    if preds.ndim == 2:
        return torch.tensor(list(map(metric, preds, targets)))
    else:
        return metric(preds, targets)


if __name__ == "__main__":
    preds = [torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.1, 0.3, 0.5, 0.2])]
    targets = [torch.tensor([False, False, True]), torch.tensor([False, True, False, True])]
    print(mean_reciprocal_rank(preds, targets))
