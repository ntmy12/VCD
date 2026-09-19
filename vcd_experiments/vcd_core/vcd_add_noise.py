import torch

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
