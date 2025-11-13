# src/loss.py

import torch


def barlow_twins_loss(z1, z2, lambda_param=5e-3):
    batch_size, embedding_dim = z1.shape

    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)

    c = (z1_norm.T @ z2_norm) / batch_size

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

    off_diag = (
        c.flatten()[:-1].view(embedding_dim - 1, embedding_dim + 1)[:, 1:].flatten()
    )
    off_diag = off_diag.pow_(2).sum()

    loss = on_diag + lambda_param * off_diag
    return loss
