# modules/sampling.py
import torch
from ModelCode.timediffusion import get_gp_covariance, alphas, betas


@torch.no_grad()
def sample(t, emb, model, time_info, device):
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    x = L @ torch.randn_like(emb)

    for step in reversed(range(len(betas))):
        alpha, beta = alphas[step], betas[step]
        noise = L @ torch.randn_like(emb)
        i = torch.tensor([step]).expand_as(x[..., :1]).to(device)
        pred_noise = model(x, t, i, time_info)
        x = (1 / (1 - beta).sqrt()) * (x - beta * pred_noise / (1 - alpha).sqrt()) + beta.sqrt() * noise

    return x
