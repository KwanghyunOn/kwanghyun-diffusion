import torch
from tqdm import tqdm


def sample_ddpm(x, model, betas):
    T = len(betas)
    betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
    alphas_cumprod = (1.0 - betas).cumprod(dim=0)
    with torch.no_grad():
        xt = x
        for t in tqdm(reversed(range(1, T+1))):
            eps = model(xt, torch.FloatTensor(t).to(xt.device))
            x0_pred = 1.0 / alphas_cumprod[t].sqrt() * xt - (1.0 / alphas_cumprod[t].sqrt() - 1.0).sqrt() * eps
            xt_mean = (alphas_cumprod[t-1].sqrt() * betas[t] * x0_pred + (1.0 - betas[t]).sqrt() * (1.0 - alphas_cumprod[t-1]) * xt) / (1.0 - alphas_cumprod[t])
            xt = xt_mean + betas[t].sqrt() * torch.randn_like(xt)

