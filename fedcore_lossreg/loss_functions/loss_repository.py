import torch
import torch.nn as nn


class LaiLoss(nn.Module):
    def __init__(self, alpha=1, base_loss=nn.MSELoss):
        super().__init__()
        self.alpha = alpha
        self.base_loss = base_loss()

    def forward(self, y_true, y_pred):
        base_loss_value = self.base_loss(y_true, y_pred)

        # Compute gradient scalar
        temp_loss = self.base_loss(y_true, y_pred)
        grads = torch.autograd.grad(temp_loss, y_pred, create_graph=True)[0]
        k_i = torch.mean(torch.abs(grads))

        # Lai regularization
        lai_func = None
        if self.alpha >= 1:
            a = torch.abs(k_i) / (torch.sqrt(1 + k_i**2))
            b = self.alpha / (torch.sqrt(1 + k_i**2))
            lai_func = torch.max(a, b)
        elif self.alpha < 1:
            a = torch.abs(k_i) / (self.alpha * torch.sqrt(1 + k_i**2))
            b = 1 / (torch.sqrt(1 + k_i**2))
            lai_func = torch.max(a, b)

        return base_loss_value * lai_func


# in progress...
class NormLoss(nn.Module):
    def __init__():
        super().__init__()


class WeightBased(nn.Module):
    def __init__():
        super().__init__()