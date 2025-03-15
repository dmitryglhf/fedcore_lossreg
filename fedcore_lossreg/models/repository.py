import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self, input_size, output_size, 
                 loss_func=nn.MSELoss, 
                 optimizer=torch.optim.Adam, 
                 loss_kwargs={}, 
                 opt_kwargs={'lr': 1e-4}):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        self.loss_func = loss_func(**loss_kwargs)  
        self.optimizer = optimizer(self.parameters(), **opt_kwargs)

    def fit(self, num_epochs: int = 100, train_dl: DataLoader = None):
        self.train()
        for epoch in range(num_epochs):
            loss = 0.0
            for x_batch, y_batch in train_dl:
                self.optimizer.zero_grad()
                pred = self(x_batch)
                loss = self.loss_func(pred, y_batch)  
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

    def predict(self, X_test):
        self.eval()
        preds = self(X_test)
        return preds.detach().numpy()

    def forward(self, x):
        return self.layers(x)


class L2Regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)