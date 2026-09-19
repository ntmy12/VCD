import torch
import pytest
from vcd_add_noise import add_diffusion_noise
from vcd_decoding import apply_vcd_penalty

def test_add_diffusion_noise():
    # Test shape preservation
    image = torch.randn(1, 3, 224, 224)
    noisy_image = add_diffusion_noise(image, noise_step=500)
    assert noisy_image.shape == image.shape
    
    # Test output differs from input
    assert not torch.allclose(image, noisy_image)

def test_apply_vcd_penalty():
    vocab_size = 1000
    original_logits = torch.randn(1, vocab_size)
    distorted_logits = torch.randn(1, vocab_size)
    
    # Simple setup
    cd_alpha = 1.0
    cd_beta = 0.1
    
    cd_logits = apply_vcd_penalty(original_logits, distorted_logits, cd_alpha, cd_beta)
    
    # Shape test
    assert cd_logits.shape == original_logits.shape
    
    # Test cutoff logic (tokens with probability < beta * max_prob are set to -inf)
    cutoff = torch.log(torch.tensor(cd_beta)) + original_logits.max(dim=-1, keepdim=True).values
    invalid_mask = original_logits < cutoff
    assert torch.all(cd_logits[invalid_mask] == -float("inf"))
    
    # Test contrastive logic for valid tokens
    valid_mask = ~invalid_mask
    expected_valid_logits = (1 + cd_alpha) * original_logits[valid_mask] - cd_alpha * distorted_logits[valid_mask]
    assert torch.allclose(cd_logits[valid_mask], expected_valid_logits)

if __name__ == "__main__":
    pytest.main([__file__])
