import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import os

def get_folder_names(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]




# === 1. Функция загрузки данных ===
def load_data():
    X_train = np.load("./PREPARED_data/SiamseNet/X_train.npy")
    X1_train = np.load("./PREPARED_data/SiamseNet/X1_train.npy")
    Y_train = np.load("./PREPARED_data/SiamseNet/Y_train.npy")

    X_test = np.load("./PREPARED_data/SiamseNet/X_test.npy")
    X1_test = np.load("./PREPARED_data/SiamseNet/X1_test.npy")
    Y_test = np.load("./PREPARED_data/SiamseNet/Y_test.npy")

    return X_train, X1_train, Y_train, X_test, X1_test, Y_test



# === 2. Сиамская нейросеть с полносвязным выходом ===
class SiameseECGNet(nn.Module):
    def __init__(self, in_channels, signal_length):
        super(SiameseECGNet, self).__init__()

        # Блок свертки
        self.conv_net = nn.Sequential(
            # 32
            nn.Conv1d(in_channels, 8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 64
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 128
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Определение выходного размера после свертки
        self.flat_size = self._get_flat_size(in_channels, signal_length)

        # Полносвязный слой для эмбеддинга
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Полносвязный слой для определения схожести
        self.similarity_layer = nn.Sequential(
            nn.Linear(128, 1),  # Выход один (вероятность схожести)
            nn.Sigmoid()  # Чтобы получить предсказание в диапазоне [0,1]
        )

    def _get_flat_size(self, in_channels, signal_length):
        """Определяем размер выхода перед полносвязным слоем."""
        with torch.no_grad():
            x = torch.zeros(1, in_channels, signal_length)
            x = self.conv_net(x)
            return x.numel()

    def forward(self, x1, x2):
        out1 = self.conv_net(x1)
        out2 = self.conv_net(x2)

        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)

        out1 = self.fc(out1)
        out2 = self.fc(out2)

        # Вычисляем абсолютную разницу эмбеддингов
        diff = torch.abs(out1 - out2)

        # Пропускаем через слой схожести
        similarity_score = self.similarity_layer(diff)

        return similarity_score


# === 3. Обучение и тестирование модели ===

X_train, X1_train, Y_train, X_test, X1_test, Y_test = load_data()

# Определяем параметры
batch_size = 50
leads = X_train.shape[1]
signal_length = X_train.shape[2]

# Преобразуем данные в PyTorch тензоры
X_train, X1_train, Y_train = map(torch.tensor, (X_train, X1_train, Y_train))
X_test, X1_test, Y_test = map(torch.tensor, (X_test, X1_test, Y_test))

indices = torch.randperm(X_test.shape[0])
X_test = X_test[indices]
X1_test = X1_test[indices]
Y_test = Y_test[indices]

# Создаем DataLoader
train_dataset = TensorDataset(X_train.float(), X1_train.float(), Y_train.float().unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Инициализируем модель
model = SiameseECGNet(in_channels=leads, signal_length=signal_length)
criterion = nn.BCELoss()  # Бинарная кросс-энтропия
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for x1, x2, y in train_loader:
        optimizer.zero_grad()
        similarity_score = model(x1, x2)
        loss = criterion(similarity_score, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Эпоха [{epoch + 1}/{num_epochs}], Потери: {total_loss:.4f}")

# === 4. Проверка точности ===
model.eval()
with torch.no_grad():
    similarity_scores = model(X_test.float(), X1_test.float()).squeeze()
    y_pred = (similarity_scores > 0.5).float()  # Порог = 0.5

    accuracy = accuracy_score(Y_test.numpy(), y_pred.numpy())

    print(f"\nТочность модели: {accuracy:.4f}")

    floders_path = "./PREPARED_data/TestData/"
    folders = get_folder_names(floders_path)
    for f in folders:
        print(f'Class: {f}')
        X_test = np.load(floders_path+f'{f}/X_test.npy')
        X1_test = np.load(floders_path + f'{f}/X1_test.npy')
        Y_test = np.load(floders_path + f'{f}/Y_test.npy')
        X_test, X1_test, Y_test = map(torch.tensor, (X_test, X1_test, Y_test))

        indices = torch.randperm(X_test.shape[0])
        X_test = X_test[indices]
        X1_test = X1_test[indices]
        Y_test = Y_test[indices]

        similarity_scores = model(X_test.float(), X1_test.float()).squeeze()
        y_pred = (similarity_scores > 0.5).float()  # Порог = 0.5

        accuracy = accuracy_score(Y_test.numpy(), y_pred.numpy())

        print(f"\nТочность {f}: {accuracy:.4f}")





