r"""Evaluation metrics for recommendation systems.

Includes regression metrics (MAE, MSE, RMSE), ranking metrics
(Precision, Recall, F1, Hit Rate, NDCG, MRR, MAP), and
classification metrics (LogLoss, AUC, GAUC).
"""

from collections import defaultdict
from functools import partial
from typing import Iterable, List, Literal, Optional, Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

__all__ = [
    "mean_abs_error",
    "mean_squared_error",
    "root_mse",
    "precision",
    "recall",
    "f1_score",
    "hit_rate",
    "normalized_dcg",
    "mean_reciprocal_rank",
    "mean_average_precision",
    "log_loss",
    "auroc",
    "group_auroc",
]


def _reduce(reduction: Literal["mean", "sum", "none"] = "mean"):
    r"""Decorator that applies a reduction to the output of a metric function.

    The decorated function should accept ``preds`` and ``targets`` as its
    first two positional arguments and return a :class:`torch.Tensor`.
    When *preds* is a list the function is mapped element-wise and the
    results are stacked before the reduction is applied.

    Parameters
    ----------
    reduction : str, optional
        Reduction operation: ``'none'``, ``'mean'`` (default), or
        ``'sum'``.

    Returns
    -------
    callable
        A decorator that wraps the target function.

    Examples
    --------
    >>> @_reduce(reduction='mean')
    ... def mean_squared_error(preds, targets):
    ...     return (preds - targets).pow(2).mean(-1)
    """

    def decorator(func):
        def wrapper(
            preds: Union[List[torch.Tensor], torch.Tensor],
            targets: Union[List[torch.Tensor], torch.Tensor],
            *,
            reduction: str = reduction,
            **kwargs,
        ):
            func_ = partial(func, **kwargs)
            if isinstance(preds, List):
                results = torch.tensor(list(map(func_, preds, targets)))
            else:
                results = func_(preds, targets)
            if reduction == "none":
                return results
            elif reduction == "mean":
                return results.mean()
            elif reduction == "sum":
                return results.sum()
            else:
                raise ValueError(
                    f"reduction should be 'none'|'mean'|'sum' but {reduction} is received ..."
                )

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# quality of predictions ========================================================================


@_reduce("mean")
def mean_abs_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""Compute Mean Absolute Error (MAE).

    .. math::

        \phi(x_i, y_i) = \frac{1}{d} \|x_i - y_i\|_1

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Predictions of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Targets of shape ``(N, d)`` or ``(d,)``.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        MAE value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_abs_error(preds, targets, reduction='none')
    tensor([0.5000, 0.5250])
    >>> mean_abs_error(preds, targets)
    tensor(0.5125)
    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).abs().mean(-1)


@_reduce("mean")
def mean_squared_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""Compute Mean Squared Error (MSE).

    .. math::

        \phi(x_i, y_i) = \frac{1}{d} \|x_i - y_i\|_2^2

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Predictions of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Targets of shape ``(N, d)`` or ``(d,)``.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        MSE value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_squared_error(preds, targets, reduction='none')
    tensor([0.3450, 0.3475])
    >>> mean_squared_error(preds, targets)
    tensor(0.3462)
    """
    preds, targets = preds.float(), targets.float()
    return (preds - targets).pow(2).mean(-1)


@_reduce("mean")
def root_mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    r"""Compute Root Mean Squared Error (RMSE).

    .. math::

        \phi(x_i, y_i) = \sqrt{\frac{1}{d} \|x_i - y_i\|_2^2}

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Predictions of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Targets of shape ``(N, d)`` or ``(d,)``.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        RMSE value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
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


@_reduce("mean")
def precision(
    preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None
) -> torch.Tensor:
    r"""Compute Precision at K.

    .. math::

        \text{Precision@K}(x_i, y_i)
        = \frac{|\operatorname{TopK}(x_i) \cap \operatorname{TopK}(y_i)|}
               {|\operatorname{TopK}(x_i)|}

    where :math:`\operatorname{TopK}(\cdot)` returns the top-K items
    and :math:`|\cdot|` denotes set cardinality.

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Binary relevance labels of shape ``(N, d)`` or ``(d,)``.
    k : int, optional
        Number of top predictions to consider.  When ``None``, all
        ``d`` items are used.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        Precision value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
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


@_reduce("mean")
def recall(
    preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None
) -> torch.Tensor:
    r"""Compute Recall at K.

    .. math::

        \text{Recall@K}(x_i, y_i)
        = \frac{|\operatorname{TopK}(x_i) \cap \operatorname{TopK}(y_i)|}
               {|y_i|}

    where :math:`|y_i|` is the total number of relevant items for
    query :math:`i`.

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Binary relevance labels of shape ``(N, d)`` or ``(d,)``.
    k : int, optional
        Number of top predictions to consider.  When ``None``, all
        ``d`` items are used.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        Recall value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
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


@_reduce("mean")
def f1_score(
    preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None
) -> torch.Tensor:
    r"""Compute the F1 score at K.

    The F1 score is the harmonic mean of precision and recall:

    .. math::

        \text{F1@K} = \frac{2 \cdot \text{Precision@K} \cdot \text{Recall@K}}
                           {\text{Precision@K} + \text{Recall@K}}

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Binary relevance labels of shape ``(N, d)`` or ``(d,)``.
    k : int, optional
        Number of top predictions to consider.  When ``None``, all
        ``d`` items are used.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        F1 value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
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
    precision_k = precision(preds, targets, k=k, reduction="none")
    recall_k = recall(preds, targets, k=k, reduction="none")
    score = 2 * precision_k * recall_k
    part2 = precision_k + recall_k
    valid = part2 != 0
    score[valid] /= part2[valid]
    return score


@_reduce("mean")
def hit_rate(
    preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None
) -> torch.Tensor:
    r"""Compute Hit Rate at K.

    Returns 1 if at least one relevant item appears in the top-K
    predictions, and 0 otherwise:

    .. math::

        \text{HR@K}(x_i, y_i)
        = \mathbb{1}\!\bigl(|\operatorname{TopK}(x_i) \cap y_i| \neq 0\bigr)

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Binary relevance labels of shape ``(N, d)`` or ``(d,)``.
    k : int, optional
        Number of top predictions to consider.  When ``None``, all
        ``d`` items are used.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        Hit rate value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
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
    r"""Compute Discounted Cumulative Gain for a relevance vector.

    .. math::

        \mathrm{DCG} = \sum_{i=1}^{K} \frac{\mathrm{rel}_i}{\log_2(i + 1)}

    Parameters
    ----------
    target : :class:`torch.Tensor`
        Relevance scores ordered by predicted rank.

    Returns
    -------
    :class:`torch.Tensor`
        Scalar or batch DCG values.
    """
    denom = torch.log2(torch.arange(target.shape[-1], device=target.device) + 2.0)
    return (target / denom).sum(dim=-1)


@_reduce("mean")
def normalized_dcg(
    preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None
) -> torch.Tensor:
    r"""Compute Normalized Discounted Cumulative Gain (NDCG) at K.

    .. math::

        \text{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}

    where :math:`\mathrm{IDCG@K}` is the ideal (maximum possible) DCG
    obtained by sorting items by true relevance.

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Relevance labels of shape ``(N, d)`` or ``(d,)``.
    k : int, optional
        Number of top predictions to consider.  When ``None``, all
        ``d`` items are used.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        NDCG value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
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


def _single_reciprocal_rank(preds: torch.Tensor, targets: torch.Tensor):
    r"""Compute the reciprocal rank for a single query.

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores (unused, ordering is assumed pre-sorted).
    targets : :class:`torch.Tensor`
        Binary relevance labels sorted by predicted rank.

    Returns
    -------
    float
        Reciprocal rank, or 0 if no relevant item exists.
    """
    if not targets.sum():
        return 0.0
    positions = torch.nonzero(targets).view(-1)
    res = 1.0 / (positions[0] + 1.0)
    return res


@_reduce("mean")
def mean_reciprocal_rank(
    preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None
) -> torch.Tensor:
    r"""Compute Mean Reciprocal Rank (MRR) at K.

    .. math::

        \text{MRR@K} = \frac{1}{N} \sum_{i=1}^{N}
        \frac{1}{\operatorname{rank}_i}

    where :math:`\operatorname{rank}_i` is the position of the first
    relevant item in the top-K list for query :math:`i`.

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Binary relevance labels of shape ``(N, d)`` or ``(d,)``.
    k : int, optional
        Number of top predictions to consider.  When ``None``, all
        ``d`` items are used.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        MRR value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_reciprocal_rank(preds, targets, reduction='none')
    tensor([1.0000, 0.5000])
    >>> mean_reciprocal_rank(preds, targets)
    tensor(0.7500)
    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))

    preds, indices = preds.topk(k=k, dim=-1)
    targets = targets.gather(-1, indices)

    if preds.ndim == 2:
        return torch.tensor(list(map(_single_reciprocal_rank, preds, targets)))
    else:
        return _single_reciprocal_rank(preds, targets)


def _single_average_precision(preds: torch.Tensor, targets: torch.Tensor):
    r"""Compute Average Precision for a single query.

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores (unused, ordering is assumed pre-sorted).
    targets : :class:`torch.Tensor`
        Binary relevance labels sorted by predicted rank.

    Returns
    -------
    float
        Average precision, or 0 if no relevant item exists.
    """
    if not targets.sum():
        return 0.0
    positions = torch.arange(
        1, len(targets) + 1, device=targets.device, dtype=torch.float32
    )[targets > 0]
    res = torch.div(
        (
            torch.arange(len(positions), device=positions.device, dtype=torch.float32)
            + 1
        ),
        positions,
    ).mean()
    return res


@_reduce("mean")
def mean_average_precision(
    preds: torch.Tensor, targets: torch.Tensor, *, k: Optional[int] = None
) -> torch.Tensor:
    r"""Compute Mean Average Precision (MAP) at K.

    .. math::

        \text{AP@K}(x_i, y_i)
        = \frac{1}{|\{j : y_{i,j}=1\}|}
          \sum_{j=1}^{K} y_{i,\pi(j)}
          \cdot \text{Precision@}j

    where :math:`\pi` is the ranking permutation induced by the
    predicted scores.  MAP is the mean of AP over all queries.

    Parameters
    ----------
    preds : :class:`torch.Tensor`
        Prediction scores of shape ``(N, d)`` or ``(d,)``.
    targets : :class:`torch.Tensor`
        Binary relevance labels of shape ``(N, d)`` or ``(d,)``.
    k : int, optional
        Number of top predictions to consider.  When ``None``, all
        ``d`` items are used.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.

    Returns
    -------
    :class:`torch.Tensor`
        MAP value(s) of shape ``(1,)`` or ``(N,)``.

    Examples
    --------
    >>> preds = torch.tensor([[0.2, 0.3, 0.5, 0.], [0.1, 0.3, 0.5, 0.2]])
    >>> targets = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    >>> mean_average_precision(preds, targets, reduction='none')
    tensor([0.7500, 0.5833])
    >>> mean_average_precision(preds, targets)
    tensor(0.6667)
    """
    preds, targets = preds.float(), targets.float()
    if k is None:
        k = preds.size(-1)
    else:
        k = min(k, preds.size(-1))

    preds, indices = preds.topk(k=k, dim=-1)
    targets = targets.gather(-1, indices)

    if preds.ndim == 2:
        return torch.tensor(list(map(_single_average_precision, preds, targets)))
    else:
        return _single_average_precision(preds, targets)


# quality of CTR


def log_loss(
    preds: Iterable,
    targets: Iterable,
    *,
    reduction: Literal["mean", "sum", "none"] = "mean",
    eps: float = 1.0e-8,
) -> np.ndarray:
    r"""Compute binary logarithmic loss (log-loss).

    .. math::

        \ell(p, y) = -\bigl[y \ln p + (1 - y) \ln(1 - p)\bigr]

    Predictions are clipped to :math:`[\varepsilon,\; 1 - \varepsilon]`
    for numerical stability.

    Parameters
    ----------
    preds : iterable
        Predicted probabilities of shape ``(N,)``.
    targets : iterable
        Binary labels (0 or 1) of shape ``(N,)``.
    reduction : str, optional
        ``'mean'`` (default), ``'sum'``, or ``'none'``.
    eps : float, optional
        Clipping epsilon.  Default is ``1e-8``.

    Returns
    -------
    :class:`numpy.ndarray`
        Scalar or array of log-loss values.

    Examples
    --------
    >>> preds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.75, 0.33]
    >>> targets = [0, 1, 0, 1, 1, 0, 1, 1]
    >>> log_loss(preds, targets)
    0.5804524803061438
    >>> log_loss(preds, targets, reduction='none')
    array([0.1053605 , 1.20397277, 0.51082561, 0.69314716, 0.51082561,
        0.22314354, 0.28768206, 1.10866259])
    """
    preds = np.array(preds, dtype=float)
    preds = np.clip(preds, eps, 1 - eps)
    targets = np.array(targets)
    loss = -(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError(
            f"reduction should be 'none'|'mean'|'sum' but {reduction} is received ..."
        )


def auroc(preds: Iterable, targets: Iterable) -> np.ndarray:
    r"""Compute the Area Under the ROC Curve (AUC).

    .. math::

        \text{AUC} = \int_0^1 \text{TPR}(t)\,d\text{FPR}(t)

    Falls back to 1.0 when only a single class is present in *targets*.

    Parameters
    ----------
    preds : iterable
        Predicted scores of shape ``(N,)``.
    targets : iterable
        Binary labels (0 or 1) of shape ``(N,)``.

    Returns
    -------
    float
        AUC score.

    Examples
    --------
    >>> preds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.75, 0.33]
    >>> targets = [0, 1, 0, 1, 1, 0, 1, 1]
    >>> auroc(preds, targets)
    0.8666666666666667
    """
    try:
        return roc_auc_score(targets, preds)
    except ValueError:
        # Return `1` if only one class presents in `targets`
        return 1.0


def group_auroc(
    preds: Iterable,
    targets: Iterable,
    groups: Iterable,
    *,
    reduction: Literal["mean", "none"] = "mean",
) -> np.ndarray:
    r"""Compute Group AUC (GAUC).

    The AUC is computed independently within each group (e.g., per user)
    and the results are aggregated by a weighted mean where weights are
    group sizes (excluding single-class groups).

    Parameters
    ----------
    preds : iterable
        Predicted scores of shape ``(N,)``.
    targets : iterable
        Binary labels (0 or 1) of shape ``(N,)``.
    groups : iterable
        Group identifiers of shape ``(N,)``.
    reduction : str, optional
        ``'mean'`` (default) or ``'none'``.

    Returns
    -------
    :class:`numpy.ndarray`
        Scalar or per-group AUC values.

    Examples
    --------
    >>> preds = [0.1, 0.3, 0.4, 0.5, 0.6, 0.2, 0.75, 0.33]
    >>> targets = [0, 1, 0, 1, 1, 0, 1, 1]
    >>> groups = [0, 0, 0, 0, 0, 1, 1, 1]
    >>> group_auroc(preds, targets, groups)
    0.8958333333333333
    >>> group_auroc(preds, targets, groups, reduction='none')
    array([0.83333333, 1.        ])
    """
    group_preds = defaultdict(list)
    group_targets = defaultdict(list)
    for group, pred, target in zip(groups, preds, targets):
        group_preds[group].append(pred)
        group_targets[group].append(target)
    group_sizes = np.array(
        [
            len(targets) if len(set(targets)) > 1 else 0
            for targets in group_targets.values()
        ]
    )
    aurocs = np.fromiter(
        map(auroc, group_preds.values(), group_targets.values()), dtype=float
    )
    if reduction == "none":
        return aurocs
    elif reduction == "mean":
        return np.sum(aurocs * group_sizes) / np.sum(group_sizes)
    else:
        raise ValueError(
            f"reduction should be 'none'|'mean' but {reduction} is received ..."
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
