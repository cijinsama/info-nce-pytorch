import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['InfoNCE', 'info_nce']


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', mode='default'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.mode = mode

    def forward(self, query, positive_key=None, negative_keys=None, labels=None):
        return info_nce(query, 
                        positive_key=positive_key,
                        negative_keys=negative_keys,
                        labels=labels,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        mode=self.mode)


def info_nce(query, positive_key=None, negative_keys=None, labels=None, temperature=0.1,
             reduction='mean', negative_mode='unpaired', mode='default'):
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')

    if mode == 'label':
        if labels is None:
            raise ValueError('In <label> mode, <labels> must be provided.')
        if labels.shape[0] != query.shape[0]:
            raise ValueError('<labels> must match the number of samples in <query>.')
    else:
        if positive_key is None:
            raise ValueError('In <default> mode, <positive_key> must be provided.')

        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')

        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must have the same number of samples.')

        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')

    # Normalize embeddings
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)

    if mode == 'label':
        sim_matrix = query @ query.T  # [N, N]

        # construct positive
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]

        if not positive_mask.any(dim=1).all():
            raise ValueError("Each sample must have at least one sample with the same label (including itself).")

        logits = sim_matrix / temperature  # [N, N]

        # numeric stable：reduce max each row
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # log-softmax
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        mean_log_prob_pos = (log_probs * positive_mask.float()).sum(dim=1) / positive_mask.float().sum(dim=1)

        loss = -mean_log_prob_pos
        return loss.mean() if reduction == 'mean' else loss

    else:
        if negative_keys is not None:
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                negative_logits = query @ transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)

        else:
            logits = query @ transpose(positive_key)
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
