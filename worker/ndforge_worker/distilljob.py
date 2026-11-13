"""Healing a thinned model by distillation ("Slim or grow a model" version 2).

    distill_cache   the original model (teacher) writes its top-k next-token probabilities for every token of the training rows to disk
    distill         the thinned model (student, LoRA) learns to give the same probabilities: KL(teacher || student) per token
    distill_merge   the LoRA is added into the student's weights (numpy, no torch) -> a new base model folder

The loss follows NVIDIA's Minitron recipe (KL on the logits, temperature 1, alpha 1.0 = no cross-entropy term) with the teacher's distribution cut to its
top-k entries plus one "everything else" bucket (Model-Optimizer's logit_kl_top_k idea). Full logits are never held in memory: the vocabulary of
Qwen3.5 has 248,320 entries, 2 GB per 2048-token row in float32, so the loss is computed in chunks of tokens that are recomputed in the backward pass.
torch and numpy are imported inside the functions that need them.
"""
from __future__ import annotations

import math

TINY = 1e-10


# ------------------------------------------------------------------------------------------------------- the math (torch)

def topk_from_logits(logits, k: int):
    """Teacher side. logits [n, V] -> (ids int64 [n, k], logp float32 [n, k] = log-probabilities of the k most likely tokens, resid float32 [n] = the
    probability left over for all other tokens)."""
    import torch
    lp = torch.log_softmax(logits.float(), dim=-1)
    top, ids = lp.topk(min(k, lp.shape[-1]), dim=-1)
    resid = (1.0 - top.exp().sum(-1)).clamp(min=0.0)
    return ids, top, resid


def kd_topk_loss(student_logp, ids, logp, resid, mask):
    """Student side, for a chunk of tokens: student_logp [n, V] log-probabilities, the teacher's (ids, logp, resid) as made by topk_from_logits, mask [n]
    (1 for real tokens). KL = sum_k p_T,k (log p_T,k - log p_S,k) + r_T (log r_T - log r_S) with r = the probability of everything outside the top-k.
    Returns (sum of the per-token KL over the real tokens, number of real tokens)."""
    import torch
    ls = student_logp.float().gather(-1, ids)
    lt = logp.float()
    main = (lt.exp() * (lt - ls)).sum(-1)
    r_t = resid.float()
    r_s = (1.0 - ls.exp().sum(-1)).clamp(min=TINY)
    rest = torch.where(r_t > 0, r_t * (torch.log(r_t.clamp(min=TINY)) - torch.log(r_s)), torch.zeros_like(r_t))
    m = mask.float()
    return ((main + rest) * m).sum(), m.sum()


def _chunk_losses(h, weight, ids, logp, resid, mask, labels, label_mask):
    import torch
    import torch.nn.functional as F
    logits = (h @ weight.t()).float()
    kd, _ = kd_topk_loss(torch.log_softmax(logits, dim=-1), ids, logp, resid, mask)
    if labels is None:
        return kd, kd.new_zeros(())
    ce = F.cross_entropy(logits, labels.clamp(min=0), reduction="none")
    return kd, (ce * label_mask.float()).sum()


def chunked_kd_loss(hidden, weight, targets, mask, chunk: int = 256, labels=None, label_mask=None):
    """hidden [n, C] (the student's last hidden states), weight [V, C] (its output layer), targets = the teacher's (ids [n, k], logp [n, k], resid [n]),
    mask [n]. The logits are made `chunk` tokens at a time inside torch.utils.checkpoint, so at most chunk x V floats exist (the backward pass makes them again).
    With labels [n] (the next token, any value where label_mask is 0) the cross-entropy over label_mask is returned as well.
    Returns (kd_sum, n_tokens, ce_sum, n_label_tokens); divide to get means."""
    import torch
    from torch.utils.checkpoint import checkpoint
    ids, logp, resid = targets
    n = hidden.shape[0]
    kd_total, ce_total = hidden.new_zeros((), dtype=torch.float32), hidden.new_zeros((), dtype=torch.float32)
    for a in range(0, n, max(1, int(chunk))):
        b = min(n, a + chunk)
        args = (hidden[a:b], weight, ids[a:b], logp[a:b], resid[a:b], mask[a:b], None if labels is None else labels[a:b], None if label_mask is None else label_mask[a:b])
        kd, ce = checkpoint(_chunk_losses, *args, use_reentrant=False)
        kd_total = kd_total + kd
        ce_total = ce_total + ce
    n_label = label_mask.float().sum() if label_mask is not None else hidden.new_zeros(())
    return kd_total, mask.float().sum(), ce_total, n_label


def forward_hidden(model, input_ids, attention_mask):
    """(hidden [b, t, C], the model's output): the hidden states after the final norm of every position, taken with a hook on the final norm. The model is
    called with logits_to_keep=1, so only the last position goes through the (huge) output layer; the training loss makes the other logits in chunks."""
    base = model
    for name in ("get_decoder",):
        if hasattr(base, name):
            try:
                base = getattr(base, name)()
            except Exception:       # noqa: BLE001 - some wrappers have a get_decoder that needs arguments
                base = model
    norm = getattr(base, "norm", None) or getattr(getattr(model, "model", None), "norm", None)
    if norm is None:
        raise ValueError("The final normalisation layer of this model was not found.")
    held: dict = {}
    handle = norm.register_forward_hook(lambda m, a, o: held.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
        except TypeError:                                  # [A] an older transformers without logits_to_keep: full logits are made once (memory!)
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return held["h"], out


# ------------------------------------------------------------------------------------------------------- numpy twin (tests)

def kd_topk_reference(student_logits, teacher_logits, k: int) -> float:
    """The same loss in plain float64 numpy, one row [n, V] of logits each: the mean KL per token of the teacher's top-k distribution (plus its bucket)
    against the student's. Used by the tests as an independent check of kd_topk_loss."""
    import numpy as np
    t = np.asarray(teacher_logits, dtype=np.float64)
    s = np.asarray(student_logits, dtype=np.float64)
    lt = t - t.max(-1, keepdims=True)
    lt = lt - np.log(np.exp(lt).sum(-1, keepdims=True))
    ls = s - s.max(-1, keepdims=True)
    ls = ls - np.log(np.exp(ls).sum(-1, keepdims=True))
    total = 0.0
    for i in range(t.shape[0]):
        top = np.argsort(-lt[i], kind="stable")[:k]
        pt = np.exp(lt[i][top])
        ps = np.exp(ls[i][top])
        r_t, r_s = max(1.0 - pt.sum(), 0.0), max(1.0 - ps.sum(), TINY)
        kl = float((pt * (lt[i][top] - ls[i][top])).sum())
        if r_t > 0:
            kl += r_t * (math.log(max(r_t, TINY)) - math.log(r_s))
        total += kl
    return total / t.shape[0]
