import torch
import torch.nn as nn


class LaiLoss(nn.Module):
    def __init__(self, alpha=1, base_loss=nn.MSELoss):
        super().__init__()
        self.alpha = alpha
        self.base_loss = base_loss

# NOW ONLY MSE FOR TEST
    def forward(self, y_pred, y_true):
        base_loss = self.base_loss()
        loss_value = base_loss(y_pred, y_true)

        # Compute gradient scalar
        grads = torch.autograd.grad(loss_value, y_pred, create_graph=True)[0]
        k_i = torch.mean(torch.abs(grads))

        # Lai regularization
        if self.alpha >= 1:
            a = k_i**2 / (1 + k_i**2)
            b = self.alpha / (1 + k_i**2)
            lai_func = max(a, b)
        elif self.alpha < 1:
            a = k_i**2 / (self.alpha * (1 + k_i**2))
            b = 1 / (1 + k_i**2)
            lai_func = max(a, b)
        else: lai_func = 1

        return loss_value * lai_func


# in progress...
class NormLoss(nn.Module):
    def __init__():
        super().__init__()


class WeightBased(nn.Module):
    def __init__():
        super().__init__()