import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import seaborn as sns
from fedcore_lossreg.loss_functions.loss_repository import LaiLoss


torch.manual_seed(42)

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

    def fit(self, num_epochs: int = 100,
            train_dl: DataLoader = None,
            plot_loss: bool = False):
        self.train()
        loss_array = []
        epoch_array = []
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
                loss_array.append(loss.item())
                epoch_array.append(epoch+1)
        print('Fitting complete!')
        if plot_loss:
            plot_loss_curve(epoch_array, loss_array)

    def predict(self, X_test):
        self.eval()
        preds = self(X_test)
        return preds.detach().numpy()

    def forward(self, x):
        return self.layers(x)


class L2Regression(nn.Module):
    def __init__(self, input_size, output_size, 
                 loss_func=nn.LaiLoss, 
                 optimizer=torch.optim.Adam, 
                 loss_kwargs={}, 
                 opt_kwargs={'lr': 1e-4}):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss_func = loss_func(**loss_kwargs)  
        self.optimizer = optimizer(self.parameters(),
                                   **opt_kwargs,
                                   weight_decay=0.01)

    def fit(self, num_epochs: int = 100,
            train_dl: DataLoader = None,
            plot_loss: bool = False):
        self.train()
        loss_array = []
        epoch_array = []
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_dl:
                self.optimizer.zero_grad()
                pred = self(x_batch)
                loss = self.loss_func(pred, y_batch)
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, L2 loss: {loss:.4f}')
                loss_array.append(loss.item())
                epoch_array.append(epoch+1)
        print('Fitting complete!')
        if plot_loss:
            plot_loss_curve(epoch_array, loss_array)

    def predict(self, X_test):
        self.eval()
        preds = self(X_test)
        return preds.detach().numpy()

    def forward(self, x):
        return self.linear(x)


class L2RegressionLai(nn.Module):
    def __init__(self, input_size, output_size, 
                 loss_func=nn.MSELoss, 
                 optimizer=torch.optim.Adam, 
                 loss_kwargs={}, 
                 opt_kwargs={'lr': 1e-4}):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss_func = loss_func(**loss_kwargs)  
        self.optimizer = optimizer(self.parameters(),
                                   **opt_kwargs,
                                   weight_decay=0.01)

    def fit(self, num_epochs: int = 100,
            train_dl: DataLoader = None,
            plot_loss: bool = False):
        self.train()
        loss_array = []
        epoch_array = []
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_dl:
                self.optimizer.zero_grad()
                pred = self(x_batch)
                loss = self.loss_func(pred, y_batch)
                loss.backward()
                self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, L2 loss: {loss:.4f}')
                loss_array.append(loss.item())
                epoch_array.append(epoch+1)
        print('Fitting complete!')
        if plot_loss:
            plot_loss_curve(epoch_array, loss_array)

    def predict(self, X_test):
        self.eval()
        preds = self(X_test)
        return preds.detach().numpy()

    def forward(self, x):
        return self.linear(x)


def plot_loss_curve(epoch_array, loss_array):
    print('Plotting the loss curve:')
    sns.lineplot(x=epoch_array, y=loss_array)
