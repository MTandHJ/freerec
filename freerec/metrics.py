

"""
Refer to
    https://zhuanlan.zhihu.com/p/67287992
for formal definitions.
"""

from typing import Optional, Union, List 

import torch
import torchmetrics

from functools import partial


__all__ = [
    'mean_abs_error', 'mean_squared_error', 'root_mse',
    'precision', 'recall', 'hit_rate', 
    'normalized_dcg', 'mean_reciprocal_rank', 'mean_average_precision'
]

def _reduce(reduction='mean'):
    def decorator(func):
        def wrapper(*args, reduction = reduction, **kwargs):
            results = func(*args, **kwargs)
            if reduction == 'none':
                return results
            elif reduction == 'mean':
                return results.mean()
            elif reduction == 'sum':
                return results.sum()
            else:
                raise ValueError(f"reduction should be 'none'|'mean'|'sum' but {reduction} is received ...")
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


# quality of predictions ========================================================================

@_reduce('mean')
def mean_abs_error(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = torchmetrics.functional.mean_absolute_error

    return torch.tensor(list(map(metric, preds, targets)))

@_reduce('mean')
def mean_squared_error(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = partial(
        torchmetrics.functional.mean_squared_error,
        squared=True
    )

    return torch.tensor(list(map(metric, preds, targets)))

@_reduce('mean')
def root_mse(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = partial(
        torchmetrics.functional.mean_squared_error,
        squared=False
    )

    return torch.tensor(list(map(metric, preds, targets)))

# quality of the set of recommendations ========================================================================

@_reduce('mean')
def precision(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        k: top-K
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = partial(
        torchmetrics.functional.retrieval_precision,
        k=k
    )

    return torch.tensor(list(map(metric, preds, targets)))

@_reduce('mean')
def recall(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        k: top-K
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = partial(
        torchmetrics.functional.retrieval_recall,
        k=k
    )

    return torch.tensor(list(map(metric, preds, targets)))

@_reduce('mean')
def hit_rate(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        k: top-K
        reduction: 'mean'(default); formally, the definition of hit rate is set specific
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = partial(
        torchmetrics.functional.retrieval_hit_rate,
        k=k
    )

    return torch.tensor(list(map(metric, preds, targets)))


# quality of the list of recommendations ========================================================================

@_reduce('mean')
def normalized_dcg(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        k: top-K
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = partial(
        torchmetrics.functional.retrieval_normalized_dcg,
        k=k
    )

    return torch.tensor(list(map(metric, preds, targets)))

@_reduce('mean')
def mean_reciprocal_rank(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = torchmetrics.functional.retrieval_reciprocal_rank

    return torch.tensor(list(map(metric, preds, targets)))

@_reduce('mean')
def mean_average_precision(preds: Union[List[torch.Tensor], torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    """
    Args: each row in Tensor or each element in List denotes a query
        preds: List or Tensor
        target: List or Tensor
    Kwargs:
        reduction: 'none'|'mean'(default)|'sum'
    """
    if isinstance(preds, torch.Tensor) and preds.ndim == 1:
        preds = preds.unsqueeze(0)
    if isinstance(targets, torch.Tensor) and targets.ndim == 1:
        targets = targets.unsqueeze(0)

    metric = torchmetrics.functional.retrieval_average_precision

    return torch.tensor(list(map(metric, preds, targets)))



if __name__ == "__main__":
    preds = [torch.tensor([0.2, 0.3, 0.5]), torch.tensor([0.1, 0.3, 0.5, 0.2])]
    targets = [torch.tensor([False, False, True]), torch.tensor([False, True, False, True])]
    print(mean_reciprocal_rank(preds, targets))
