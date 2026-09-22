import math
import torch

_NUM_STEPS = 1000

def vcd_alpha_bar(num_steps: int = _NUM_STEPS) -> torch.Tensor:
    betas = torch.sigmoid(torch.linspace(-6, 6, num_steps)) * (0.5e-2 - 1e-5) + 1e-5
    return torch.cumprod(1.0 - betas, dim=0)

def add_diffusion_noise_qwen2vl(pixel_values: torch.Tensor,
                                noise_step: int,
                                *, in_channels: int = 3,
                                temporal_patch_size: int = 2,
                                patch_size: int = 14,
                                generator: torch.Generator | None = None) -> torch.Tensor:
    assert 0 <= noise_step < _NUM_STEPS
    assert pixel_values.ndim == 2 and pixel_values.shape[1] == in_channels * temporal_patch_size * patch_size**2
    ab = float(vcd_alpha_bar()[noise_step])
    n = pixel_values.shape[0]
    x = pixel_values.float().view(n, in_channels, temporal_patch_size, patch_size, patch_size)
    eps = torch.randn(n, in_channels, 1, patch_size, patch_size,
                      device=x.device, dtype=torch.float32, generator=generator)
    eps = eps.expand(-1, -1, temporal_patch_size, -1, -1)
    out = math.sqrt(ab) * x + math.sqrt(1.0 - ab) * eps
    return out.reshape(n, -1).to(pixel_values.dtype)

def add_diffusion_noise(image_tensor, noise_step):
    """
    Adds diffusion-style noise to an image tensor.
    Args:
        image_tensor: The original image tensor.
        noise_step: The diffusion step (e.g., 500) determining the noise level.
    Returns:
        The noisy image tensor.
    """
    num_steps = 1000  # Number of diffusion steps

    # Decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # Decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    noise_step = int(noise_step)
    # Clamp noise step to valid range
    noise_step = max(0, min(noise_step, num_steps - 1))
    
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step) 

    return image_tensor_cd
