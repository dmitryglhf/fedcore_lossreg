# %%
import pandas as pd

import torch

from fedcore_lossreg.preprocessing.convert import convert_df_to_dl
from fedcore_lossreg.models.model_repository import MLP, L2Regression
from fedcore_lossreg.loss_functions.loss_repository import LaiLoss

# %%
torch.manual_seed(42)

# %%
# Get data
train = pd.read_csv('data/kaggle_s3e1/train.csv')
X_test = pd.read_csv('data/kaggle_s3e1/test.csv')

train.drop(columns='id', inplace=True)
X_test.drop(columns='id', inplace=True)

X_train, y_train = train.drop(columns='MedHouseVal'), train['MedHouseVal']

# Convert to data loader
train_dl = convert_df_to_dl(X_train, y_train)
X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)

# %%
# # Base MLP model
# mlp_model = MLP(input_size=X_train.shape[1], output_size=1)
# mlp_model.fit(num_epochs=1000, train_dl=train_dl, silent=True)
# y_pred = mlp_model.predict(X_test)

# %%
# # Base Ridge model
# ridge_model = L2Regression(input_size=X_train.shape[1], output_size=1)
# ridge_model.fit(num_epochs=1000, train_dl=train_dl, plot_loss=True)
# y_pred = ridge_model.predict(X_test)

# %%
# Ridge model with LaiLoss
ridge_model_lai = L2Regression(input_size=X_train.shape[1],output_size=1, loss_func=LaiLoss)
ridge_model_lai.fit(
    num_epochs=300,
    train_dl=train_dl,
    plot_loss=True,
    verbosity=10,
    loss_step=10
)
y_pred_lai = ridge_model_lai.predict(X_test)

# %%
# submission = pd.read_csv('data/kaggle_s3e1/sample_submission.csv')
# submission['MedHouseVal'] = y_pred
# submission.to_csv('submission.csv', index=False)

# %%



