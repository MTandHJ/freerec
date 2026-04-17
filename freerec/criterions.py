from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BaseCriterion",
    "CrossEntropy4Logits",
    "KLDivLoss4Logits",
    "BPRLoss",
    "BCELoss4Logits",
    "MSELoss",
    "L1Loss",
]


class BaseCriterion(nn.Module):
    r"""Base class for all FreeRec loss criteria.

    Provides a common ``reduction`` attribute and a static helper for
    parameter regularization.

    Parameters
    ----------
    reduction : str, optional
        Reduction applied to the loss output.  One of ``'none'``,
        ``'sum'``, or ``'mean'`` (default).
    """

    def __init__(self, reduction: str = "mean") -> None:
        r"""Initialize BaseCriterion with the given reduction mode."""
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def regularize(
        params: Union[torch.Tensor, Iterable[torch.Tensor]], rtype: str = "l2"
    ):
        r"""Compute a regularization penalty over the given parameters.

        Parameters
        ----------
        params : :class:`torch.Tensor` or iterable of :class:`torch.Tensor`
            Parameters to regularize.
        rtype : str, optional
            Regularization type.  ``'l1'`` for L1 norm, ``'l2'`` (default)
            for squared L2 norm divided by 2.

        Returns
        -------
        :class:`torch.Tensor`
            Scalar regularization loss.

        Raises
        ------
        NotImplementedError
            If *rtype* is not ``'l1'`` or ``'l2'``.
        """
        params = [params] if isinstance(params, torch.Tensor) else params
        if rtype == "l1":
            return sum(param.abs().sum() for param in params)
        elif rtype == "l2":
            return sum(param.pow(2).sum() for param in params) / 2
        else:
            raise NotImplementedError(f"{rtype} regularization is not supported ...")


class CrossEntropy4Logits(BaseCriterion):
    r"""Cross-entropy loss operating on raw logits.

    Wraps :func:`torch.nn.functional.cross_entropy`.
    """

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ):
        r"""Compute the cross-entropy loss.

        Parameters
        ----------
        logits : :class:`torch.Tensor`
            Unnormalized logits of shape ``(C,)``, ``(B, C)``, or
            ``(B, C, ...)``.
        targets : :class:`torch.Tensor`
            Ground-truth class indices of shape ``(,)``, ``(B,)``, or
            ``(B, ...)``.
        reduction : str, optional
            Override the instance-level reduction if provided.

        Returns
        -------
        :class:`torch.Tensor`
            Computed loss.
        """
        reduction = reduction if reduction is not None else self.reduction
        return cross_entropy_with_logits(logits, targets, reduction)


class BCELoss4Logits(BaseCriterion):
    r"""Binary cross-entropy loss operating on raw logits.

    Wraps :func:`torch.nn.functional.binary_cross_entropy_with_logits`.
    """

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ):
        r"""Compute the binary cross-entropy loss.

        Parameters
        ----------
        logits : :class:`torch.Tensor`
            Unnormalized logits.
        targets : :class:`torch.Tensor`
            Binary targets with the same shape as *logits*.
        reduction : str, optional
            Override the instance-level reduction if provided.

        Returns
        -------
        :class:`torch.Tensor`
            Computed loss.
        """
        reduction = reduction if reduction is not None else self.reduction
        return binary_cross_entropy_with_logits(logits, targets, reduction)


class KLDivLoss4Logits(BaseCriterion):
    r"""KL divergence loss operating on raw logits.

    Both *logits* and *targets* are expected to be unnormalized; softmax /
    log-softmax is applied internally.

    Parameters
    ----------
    reduction : str, optional
        Reduction mode.  Default is ``'batchmean'``.
    """

    def __init__(self, reduction: str = "batchmean") -> None:
        r"""Initialize KLDivLoss4Logits with the given reduction mode."""
        super().__init__(reduction)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ):
        r"""Compute the KL divergence loss.

        Parameters
        ----------
        logits : :class:`torch.Tensor`
            Unnormalized logits.
        targets : :class:`torch.Tensor`
            Unnormalized target logits with the same shape as *logits*.
        reduction : str, optional
            Override the instance-level reduction if provided.

        Returns
        -------
        :class:`torch.Tensor`
            Computed loss.
        """
        reduction = reduction if reduction is not None else self.reduction
        return kl_div_loss_with_logits(logits, targets, reduction)


class BPRLoss(BaseCriterion):
    r"""Bayesian Personalized Ranking (BPR) pairwise loss.

    Computes :math:`\log\sigma(\text{neg\_scores} - \text{pos\_scores})`
    via :func:`torch.nn.functional.softplus`.
    """

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        reduction: Optional[str] = None,
    ):
        r"""Compute the BPR loss.

        Parameters
        ----------
        pos_scores : :class:`torch.Tensor`
            Scores for positive (observed) items.
        neg_scores : :class:`torch.Tensor`
            Scores for negative (unobserved) items.
        reduction : str, optional
            Override the instance-level reduction if provided.

        Returns
        -------
        :class:`torch.Tensor`
            Computed loss.
        """
        reduction = reduction if reduction is not None else self.reduction
        return bpr_loss_with_logits(pos_scores, neg_scores, reduction)


class MSELoss(BaseCriterion):
    r"""Mean squared error loss.

    Wraps :func:`torch.nn.functional.mse_loss`.
    """

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ):
        r"""Compute the MSE loss.

        Parameters
        ----------
        inputs : :class:`torch.Tensor`
            Predicted values.
        targets : :class:`torch.Tensor`
            Ground-truth values with the same shape as *inputs*.
        reduction : str, optional
            Override the instance-level reduction if provided.

        Returns
        -------
        :class:`torch.Tensor`
            Computed loss.
        """
        reduction = reduction if reduction is not None else self.reduction
        return F.mse_loss(inputs, targets, reduction=reduction)


class L1Loss(BaseCriterion):
    r"""L1 (mean absolute error) loss.

    Wraps :func:`torch.nn.functional.l1_loss`.
    """

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: Optional[str] = None,
    ):
        r"""Compute the L1 loss.

        Parameters
        ----------
        inputs : :class:`torch.Tensor`
            Predicted values.
        targets : :class:`torch.Tensor`
            Ground-truth values with the same shape as *inputs*.
        reduction : str, optional
            Override the instance-level reduction if provided.

        Returns
        -------
        :class:`torch.Tensor`
            Computed loss.
        """
        reduction = reduction if reduction is not None else self.reduction
        return F.l1_loss(inputs, targets, reduction=reduction)


def cross_entropy_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    r"""Compute cross-entropy loss from unnormalized logits.

    Parameters
    ----------
    logits : :class:`torch.Tensor`
        Unnormalized logits of shape ``(C,)``, ``(B, C)``, or
        ``(B, C, ...)``.
    targets : :class:`torch.Tensor`
        Class indices of shape ``(,)``, ``(B,)``, or ``(B, ...)``.
    reduction : str, optional
        One of ``'none'``, ``'sum'``, or ``'mean'`` (default).

    Returns
    -------
    :class:`torch.Tensor`
        Computed loss.
    """
    return F.cross_entropy(logits, targets, reduction=reduction)


def binary_cross_entropy_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    r"""Compute binary cross-entropy loss from unnormalized logits.

    Parameters
    ----------
    logits : :class:`torch.Tensor`
        Unnormalized logits.
    targets : :class:`torch.Tensor`
        Binary targets with the same shape as *logits*.
    reduction : str, optional
        One of ``'none'``, ``'sum'``, or ``'mean'`` (default).

    Returns
    -------
    :class:`torch.Tensor`
        Computed loss.
    """
    return F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction=reduction
    )


def kl_div_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str = "batchmean"
) -> torch.Tensor:
    r"""Compute KL divergence loss from unnormalized logits.

    Applies log-softmax to *logits* and softmax to *targets* before
    calling :func:`torch.nn.functional.kl_div`.

    Parameters
    ----------
    logits : :class:`torch.Tensor`
        Unnormalized logits.
    targets : :class:`torch.Tensor`
        Unnormalized target logits with the same shape as *logits*.
    reduction : str, optional
        One of ``'none'``, ``'sum'``, ``'mean'``, or ``'batchmean'``
        (default).

    Returns
    -------
    :class:`torch.Tensor`
        Computed loss.
    """
    assert logits.size() == targets.size()
    inputs = F.log_softmax(logits, dim=-1)
    targets = F.softmax(targets, dim=-1)
    return F.kl_div(inputs, targets, reduction=reduction)


def bpr_loss_with_logits(
    pos_scores: torch.Tensor, neg_scores: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    r"""Compute Bayesian Personalized Ranking loss.

    .. math::

        \mathcal{L} = \mathrm{softplus}(s^{-} - s^{+})

    where :math:`s^{+}` and :math:`s^{-}` denote positive and negative
    scores, respectively.

    Parameters
    ----------
    pos_scores : :class:`torch.Tensor`
        Scores for positive items.
    neg_scores : :class:`torch.Tensor`
        Scores for negative items.
    reduction : str, optional
        One of ``'none'``, ``'sum'``, or ``'mean'`` (default).

    Returns
    -------
    :class:`torch.Tensor`
        Computed loss.

    Raises
    ------
    NotImplementedError
        If *reduction* is not a supported value.
    """
    loss = F.softplus(neg_scores - pos_scores)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise NotImplementedError(
            f"reduction mode of '{reduction}' is not supported ..."
        )
