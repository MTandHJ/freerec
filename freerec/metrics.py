

"""
Refer to
    https://zhuanlan.zhihu.com/p/67287992
for formal definitions.
The implementations below are fully due to torchmetrics.
"""

from typing import Optional, Union, List 
from functools import  partial

import torch
import torchmetrics


__all__ = [
    'mean_abs_error', 'mean_squared_error', 'root_mse',
    'precision', 'recall', 'f1_score', 'auroc', 'hit_rate',
    'normalized_dcg', 'mean_reciprocal_rank', 'mean_average_precision'
]


def _reduce(reduction='mean'):
    r"""
    A decorator that applies a reduction operation to the output of a given function.

    Parameters:
    -----------
    reduction : str, optional
        The reduction operation to be applied to the output of the function. Should be one of {'none', 'mean', 'sum'}. Default is 'mean'.

    Returns:
    --------
    wrapper : function
        The decorated function.

    Notes:
    ------
    This decorator takes a function that takes two inputs: `preds` and `targets`, 
    both of which are either lists of PyTorch tensors or PyTorch tensors themselves. 
    The output of the function should also be a PyTorch tensor or a list of PyTorch tensors. 
    The decorator then applies a reduction operation specified by the `reduction` parameter to the output of the function.

    Examples:
    ---------
    Here is an example usage of this decorator:

    >>> @_reduce(reduction='mean')
    ... def mean_squared_error(preds, targets):
    ...     return F.mse_loss(preds, targets)

    Here, we define a function `mean_squared_error` that computes the mean squared error between `preds` and `targets` 
    using PyTorch's built-in mean squared error loss function (`F.mse_loss`). 
    The decorator `_reduce` is applied to this function, with `reduction` set to 'mean'. 
    This means that the output of `mean_squared_error` will be the mean of the mean squared error loss values computed for each element in `preds` and `targets`.
    """
    def decorator(func):
        def wrapper(
            preds: Union[List[torch.Tensor], torch.Tensor], 
            targets: Union[List[torch.Tensor], torch.Tensor], 
            reduction: str = reduction, **kwargs
        ):
            func_ = partial(func, **kwargs)
            if isinstance(preds, List):
                results = torch.tensor(list(map(
                    func_, preds, targets
                )))
            else:
                results = func_(preds, targets)
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
def mean_abs_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""
    Computes Mean Absolute Error (MAE).

    .. math::

        \phi(x_i, y_i) = \frac{1}{d} \|x_i - y_i\|_1.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.


    Returns:
    --------
    Computed MAE: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
    r"""
    Calculate mean squared error (MSE).

    .. math::

        \phi(x_i, y_i) = \frac{1}{d} \|x_i - y_i\|_2^2.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.

    Returns:
    --------
    Computed MSE: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
    r"""Root mean squared error (RMSE).

    .. math::

        \phi(x_i, y_i) = \sqrt{\frac{1}{d} \|x_i - y_i\|_2^2}.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.

    Returns:
    --------
    Computed RMSE: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> root_mean_squared_error(preds, targets, reduction='none')
    tensor([0.5874, 0.5895])
    >>> root_mean_squared_error(preds, targets)
    tensor(0.5884)
    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).pow(2).mean(-1).sqrt()


# quality of the set of recommendations ========================================================================

@_reduce('mean')
def precision(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    r"""
    Compute the precision of the top-K predicted items for a given query.

    The precision is defined as the intersection between the top-K predicted items and the target items divided by
    the size of the top-K set.

    .. math::

        \phi(x_i, y_i) = (|Topk(x_i) \cap Topk(y_i)|) / |Topk(x_i)|,
    
    where :math: `Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math: `i`,
    and :math: `|\cdot|` measures the cardinality of the set.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.
    k: int, optional
        - `int': Top-(K=k).
        - `None': Top-(K=d)

    Returns:
    --------
    Computed Precision: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
    return relevant / k

@_reduce('mean')
def recall(preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None) -> torch.Tensor:
    r"""
    Recall is a measure of how well a ranking algorithm retrieves relevant items for a given query. 
    It is defined as the intersection of the top-K predicted items with the top-K relevant items, 
    divided by the total number of relevant items.

    .. math::

        \phi(x_i, y_i) = (|Topk(x_i) \cap Topk(y_i)|) / |y_i|,

    where :math:`Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math:`i`,
    and :math:`|\cdot|` measures the cardinality of the set.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.
    k: int, optional
        - `int': Top-(K=k).
        - `None': Top-(K=d)

    Returns:
    --------
    Computed Recall: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
    r"""
    The F1 score is the harmonic mean of precision and recall, where precision is the number of true positive predictions
    divided by the total number of positive predictions, and recall is the number of true positive predictions divided by
    the total number of positive ground truth labels.

    The F1 score is computed as follows:

    .. math::

        \phi(x_i, y_i) = (2 * Precision@k(x_i, y_i) * Recall@k(x_i, y_i)) / (Precision@k(x_i, y_i) + Recall@k(x_i, y_i))

    where :math: `Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math: `i`,
    and :math: `|\cdot|` measures the cardinality of the set.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.
    k: int, optional
        - `int': Top-(K=k).
        - `None': Top-(K=d)

    Returns:
    --------
    Computed F1-score: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
    r"""
    AUC (Area Under the ROC Curve) represents the area under the Receiver Operating Characteristic (ROC) curve, 
    which is a curve showing the performance of a binary classifier system.

    .. math::

        \frac{n - (|N_+| - 1) / 2 - (\sum_{v \in |N_+|} rank(v)) / |N_+|}{n - |N_+|}

    where :math: `n` and :math: `|N_+|` denote the number of items and positive items, respectively; :math: `rank(v)` returns the rank of the positive item :math: `v`.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.

    Returns:
    --------
    Computed AUC: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
    r"""
    Compute the Hit Rate score between predictions and targets.

    .. math::

        \phi(x_i, y_i) = \mathbb{I}(|Topk(x_i) \cap Topk(y_i)| \not = 0).
    
    where :math: `Topk(\cdot)` returns the Top-K (predicted or ideal) items of the query :math: `i`,
    and :math: `|\cdot|` measures the cardinality of the set.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.
    k: int, optional
        - `int': Top-(K=k).
        - `None': Top-(K=d)

    Returns:
    --------
    Computed HitRate: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
    r"""
    NDCG (Normalized Discounted Cumulative Gain).

    .. math::

        \phi(x_i, y_i) = \frac{_dcg(x_i)}{_dcg(y_i)}

    where :math: `_dcg(\cdot)` computes the Discounted Cumulative Gain of the input.
        
    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.
    k: int, optional
        - `int': Top-(K=k).
        - `None': Top-(K=d)

    Returns:
    --------
    Computed NDCG: torch.Tensor, shape (1,) or (N,)
        
    Examples:
    ---------
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
    r"""
    The mean reciprocal rank (MRR) is defined as the average of 
    the reciprocal rank of the first correct item in the predicted ranking of items for each query.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.

    Returns:
    --------
    Computed MRR: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------

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
    r"""
    MAP calculates the mean average precision for a set of predictions and targets.

    Parameters:
    -----------
    preds: torch.Tensor, shape (N, d) or (d,)
        Predictions or a single prediction tensor.
    targets: torch.Tensor, shape (N, d) or (d,)
        Targets tensor in accordance with preds.
    reduction: str, optional
        Specifies the reduction to apply to the output:
        - 'mean': the mean value of the error is returned. Default.
        - 'sum': the sum of the error is returned.
        - 'none': no reduction will be applied and a tensor of shape (n,) will be returned.

    Returns:
    --------
    Computed MAP: torch.Tensor, shape (1,) or (N,)

    Examples:
    ---------
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
