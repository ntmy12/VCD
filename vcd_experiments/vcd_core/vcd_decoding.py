import math
import torch

def vcd_contrast(logits: torch.Tensor, logits_cd: torch.Tensor,
                 cd_alpha: float = 1.0, cd_beta: float = 0.1) -> torch.Tensor:
    """logits, logits_cd: [B, V] float32. Trả về cd_logits đã mask APC (chưa temperature/top-p)."""
    cutoff = math.log(cd_beta) + logits.max(dim=-1, keepdim=True).values
    diffs = (1.0 + cd_alpha) * logits - cd_alpha * logits_cd
    return diffs.masked_fill(logits < cutoff, float("-inf"))

def _top_p_filter(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return scores
    sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
    probs = torch.softmax(sorted_scores, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) >= top_p
    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
    return torch.full_like(scores, float("-inf")).scatter(-1, sorted_idx, sorted_scores)

def select_token(scores: torch.Tensor, *, do_sample: bool, temperature: float,
                 top_p: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if not do_sample:
        return scores.argmax(dim=-1)
    scores = _top_p_filter(scores / temperature, top_p)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

def apply_vcd_penalty(original_logits, distorted_logits, cd_alpha=1.0, cd_beta=0.1):
    """
    Applies the Visual Contrastive Decoding (VCD) penalty to logits.
    
    Args:
        original_logits: Logits from the model with the original image.
        distorted_logits: Logits from the model with the distorted (noisy) image.
        cd_alpha: Contrastive weight (alpha).
        cd_beta: Adaptive plausibility constraint threshold (beta).
        
    Returns:
        The contrastive logits after applying VCD and the plausibility constraint.
    """
    cutoff = torch.log(torch.tensor(cd_beta, device=original_logits.device)) + original_logits.max(dim=-1, keepdim=True).values
    diffs = (1 + cd_alpha) * original_logits - cd_alpha * distorted_logits
    cd_logits = diffs.masked_fill(original_logits < cutoff, -float("inf"))
    return cd_logits
