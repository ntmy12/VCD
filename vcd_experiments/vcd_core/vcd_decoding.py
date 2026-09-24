import math
import torch

def vcd_contrast(logits: torch.Tensor, logits_cd: torch.Tensor,
                 cd_alpha: float = 1.0, cd_beta: float = 0.1) -> torch.Tensor:
    """logits, logits_cd: [B, V] float32. Returns cd_logits masked with Adaptive Plausibility Cutoff (APC)."""
    cutoff = math.log(cd_beta) + logits.max(dim=-1, keepdim=True).values
    diffs = (1.0 + cd_alpha) * logits - cd_alpha * logits_cd
    return diffs.masked_fill(logits < cutoff, float("-inf"))

def _top_p_filter(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return scores
    sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    probs = torch.softmax(sorted_scores, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) >= top_p            # highest probability token is always preserved
    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
    return torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_scores)

def select_token(scores: torch.Tensor, *, do_sample: bool, temperature: float,
                 top_p: float, generator: torch.Generator | None) -> torch.Tensor:
    if not do_sample:
        return scores.argmax(dim=-1)
    scores = _top_p_filter(scores / temperature, top_p)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

def apply_vcd_penalty(logits, logits_cd, alpha, beta):
    cutoff = math.log(beta) + logits.max(dim=-1, keepdim=True).values
    diffs = (1.0 + alpha) * logits - alpha * logits_cd
    return diffs.masked_fill(logits < cutoff, float("-inf"))