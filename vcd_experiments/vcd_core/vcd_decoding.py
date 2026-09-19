import torch

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
    # Calculate cutoff for Adaptive Plausibility Constraints
    # cutoff = log(beta) + max(original_logits)
    cutoff = torch.log(torch.tensor(cd_beta, device=original_logits.device)) + original_logits.max(dim=-1, keepdim=True).values
    
    # Calculate contrastive differences: (1 + alpha) * original - alpha * distorted
    diffs = (1 + cd_alpha) * original_logits - cd_alpha * distorted_logits
    
    # Apply plausibility constraint: mask out tokens with original probability below the cutoff
    cd_logits = diffs.masked_fill(original_logits < cutoff, -float("inf"))
    
    return cd_logits
