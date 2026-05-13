import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Трансформации для данных
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Загрузка датасета MNIST
batch_size = 64

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)


# Определение обучаемой функции активации
class TrainableActivation(nn.Module):
    def __init__(self):
        super(TrainableActivation, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(1.0))
        self.c = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.a * x + self.b * torch.tanh(self.c * x)


# Построение нейронной сети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.act1 = TrainableActivation()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.act2 = TrainableActivation()
        self.fc1 = nn.Linear(320, 50)
        self.act3 = TrainableActivation()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net().to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Обучение модели
num_epochs = 5
train_losses = []
train_accuracies = []
test_accuracies = []


def compute_accuracy(loader, net):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return 100 * correct / total


for epoch in range(num_epochs):
    running_loss = 0.0
    total = 0
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total
    test_accuracy = compute_accuracy(testloader, net)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        f'Эпоха {epoch + 1}, Потеря: {train_loss:.3f}, Точность на обучении: {train_accuracy:.2f}%, Точность на тесте: {test_accuracy:.2f}%')

print('Обучение завершено')

# Визуализация результатов обучения
plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Точность на обучении')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Точность на тесте')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.legend()
plt.title('График точности модели')
plt.show()


# Построение графиков функций активации
def plot_activation(act, name):
    a = act.a.item()
    b = act.b.item()
    c = act.c.item()
    x = torch.linspace(-5, 5, 100).to(device)
    y = a * x + b * torch.tanh(c * x)
    y = y.cpu().detach().numpy()
    x_np = x.cpu().numpy()
    plt.figure()
    plt.plot(x_np, y)
    plt.title(f'Функция активации {name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


plot_activation(net.act1, 'act1')
plot_activation(net.act2, 'act2')
plot_activation(net.act3, 'act3')
