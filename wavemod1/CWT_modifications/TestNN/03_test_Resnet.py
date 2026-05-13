import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def scale_data(data):
    scaled_data = []
    for scalogram in data:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
        scaler = MaxAbsScaler()
        data_normalized = scaler.fit_transform(scalogram)
        scaled_data.append(data_normalized)
    data = np.array(scaled_data)
    return data
def test_model(norm, pat):

    # Предполагается, что у вас есть данные в numpy массивах norm и pat
    # Создаем метки для данных
    labels_norm = np.zeros((norm.shape[0],))
    labels_pat = np.ones((pat.shape[0],))

    # Объединяем данные и метки
    data = np.concatenate([norm, pat], axis=0)
    labels = np.concatenate([labels_norm, labels_pat], axis=0)

    #-----
    scaled_data = []
    for scalogram in data:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler,RobustScaler,QuantileTransformer
        scaler = MaxAbsScaler()
        data_normalized = scaler.fit_transform(scalogram)
        scaled_data.append(data_normalized)
    data = np.array(scaled_data)

    #------

    # Преобразование данных в тензоры PyTorch
    data_tensor = torch.tensor(data[:, None, :, :], dtype=torch.float32)  # Добавляем канал
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Преобразование данных в тензоры PyTorch и добавление нормализации
    dataset = TensorDataset(data_tensor, labels_tensor)


    train_loader = DataLoader(dataset, batch_size=30, shuffle=True)

    # Изменение архитектуры ResNet
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Два выходных класса

    # Перенос модели на GPU, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Тренировка модели
    num_epochs = 10
    train_losses = []
    train_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

    return train_losses, train_accuracy


def test_model_with_split(norm, pat):
    # Предполагается, что у вас есть данные в numpy массивах norm и pat

    # Создаем метки для данных
    labels_norm = np.zeros((norm.shape[0],))
    labels_pat = np.ones((pat.shape[0],))

    # Объединяем данные и метки
    data = np.concatenate([norm, pat], axis=0)
    labels = np.concatenate([labels_norm, labels_pat], axis=0)

    # Разбиение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    X_train_normalized = scale_data(X_train)
    X_test_normalized = scale_data(X_test)

    # Преобразование данных в тензоры PyTorch
    train_data_tensor = torch.tensor(X_train_normalized[:, None, :, :], dtype=torch.float32)  # Добавляем канал
    train_labels_tensor = torch.tensor(y_train, dtype=torch.long)
    test_data_tensor = torch.tensor(X_test_normalized[:, None, :, :], dtype=torch.float32)  # Добавляем канал
    test_labels_tensor = torch.tensor(y_test, dtype=torch.long)

    # Преобразование данных в тензоры PyTorch и добавление нормализации
    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    batch_size = 70
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = models.resnext50_32x4d(pretrained=False)

    # Замена первого слоя для соответствия размеру входных данных
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Изменение выходного слоя для соответствия количеству классов (2 класса)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # # Изменение архитектуры ResNet
    # model = models.resnet18(pretrained=False)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(model.fc.in_features, 2)  # Два выходных класса

    # Перенос модели на GPU, если доступно
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Тренировка модели
    num_epochs = 10
    train_losses = []
    train_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # print(np.shape(inputs))
            #torch.Size([70, 1, 40, 500])

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_acc)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")

    # Оценка модели на тестовой выборке
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Рассчет чувствительности и специфичности
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)

    # Рассчет общей точности
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Рассчет Положительного Предсказательного Значения (PPV) и Отрицательного Предсказательного Значения (NPV)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    print(f"Sensitivity: {sensitivity}, Specificity: {specificity}, f1{f1}, accuracy{accuracy} ,ppv {ppv}, npv {npv}")

    return train_losses, train_accuracy, sensitivity, specificity




norm = np.load('D:/Bases/only_sinus/train/CWT/avg_card_norm.npy')
pat = np.load('D:/Bases/only_sinus/train/CWT/avg_card_LVP.npy')

norm_mod = np.load('D:/Bases/only_sinus/train/CWTmodif/old/avg_card_norm.npy')
pat_mod = np.load('D:/Bases/only_sinus/train/CWTmodif/old/avg_card_LVP.npy')


# norm_mod = np.load('D:/Bases/only_sinus/train/CWTmodif/avg_card_norm.npy')
# pat_mod = np.load('D:/Bases/only_sinus/train/CWTmodif/avg_card_LVP.npy')


# train_loss_mod, train_accuracy_mod,sens_mod,spec_mod = test_model(norm_mod,pat_mod)
# train_loss_class, train_accuracy_class,sens,spec = test_model(norm,pat)

train_loss_mod, train_accuracy_mod,sens_mod,spec_mod = test_model_with_split(norm_mod,pat_mod)
train_loss_class, train_accuracy_class,sens,spec = test_model_with_split(norm,pat)

# Визуализация потерь и точности
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_mod, label='Training Loss mod')
plt.plot(train_loss_class, label='Training Loss classic')
plt.title('ResNet Training Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracy_mod, label='Training Accuracy mod')
plt.plot(train_accuracy_class, label='Training Accuracy classic')
plt.title('Training Accuracy')
plt.title('ResNet Training Accuracy')
plt.legend()
plt.show()

# Для оценки модели на валидационном наборе добавьте соответствующий код
