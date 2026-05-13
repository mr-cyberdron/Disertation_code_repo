import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import tensorflow as tf

from changebleAFNNstruct.p1LoadData import loadMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train test split
X_train, X_test, y_train, y_test = loadMNIST()

# create tensor from data
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

input(tf.shape(train_data))

train_loader = DataLoader(train_data, shuffle=True, batch_size=100)
test_loader = DataLoader(test_data, batch_size=100)


# Custom Sigmoid activation with trainable parameters
class CustomSigmoid(nn.Module):
    def __init__(self, initial_a=1.0, initial_b=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial_a))
        self.b = nn.Parameter(torch.tensor(initial_b))

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.a * (x + self.b)))


# create network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.act1 = CustomSigmoid(initial_a=1.0, initial_b=0.0)

        # Добавление нелинейности и batch normalization
        self.layer2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act2 = CustomSigmoid(initial_a=0.5, initial_b=-0.5)

        self.layer3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.layer4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.bn1(self.act2(self.layer2(x)))
        x = self.dropout(self.layer3(x))
        x = self.layer4(x)
        return x


model = Net()

# Configure training details
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=100)

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    y_pred = model(X_test)
    acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc * 100))

# Plot the resulting sigmoid functions
x_vals = torch.linspace(-10, 10, 100)
with torch.no_grad():
    y_vals_1 = model.act1(x_vals)
    y_vals_2 = model.act2(x_vals)

plt.figure(figsize=(12, 6))
plt.plot(x_vals.numpy(), y_vals_1.numpy(),
         label=f'Sigmoid 1 (a={model.act1.a.item():.4f}, b={model.act1.b.item():.4f})')
plt.plot(x_vals.numpy(), y_vals_2.numpy(),
         label=f'Sigmoid 2 (a={model.act2.a.item():.4f}, b={model.act2.b.item():.4f})')
plt.title("Trained Sigmoid Functions")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.legend()
plt.grid(True)
plt.show()
