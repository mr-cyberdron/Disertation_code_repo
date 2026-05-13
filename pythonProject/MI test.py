import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma


def mi_knn(signal1, signal2, k=5):
    """Вычисляет взаимную информацию между сигналами через KNN."""
    N = min(len(signal1), len(signal2))  # Берём минимальную длину
    data = np.vstack((signal1[:N], signal2[:N])).T  # Объединяем сигналы в 2D

    # Ищем K ближайших соседей
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Вычисляем энтропию через средние расстояния до соседей
    rho = distances[:, -1]  # Расстояние до k-го соседа
    n_x = np.zeros(N)
    n_y = np.zeros(N)

    for i in range(N):
        n_x[i] = np.sum(np.abs(signal1 - signal1[i]) < rho[i]) - 1
        n_y[i] = np.sum(np.abs(signal2 - signal2[i]) < rho[i]) - 1

    # Используем дигамма-функцию
    mi_value = digamma(N) + digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))

    return mi_value



# Параметры сигнала
fs = 1000  # Частота дискретизации (Гц)

# Создание первого сигнала (15 + 5 Гц, 3000 семплов)
t1 = np.linspace(0, 3, 3000, endpoint=False)  # Время от 0 до 3 секунд
signal1 = np.sin(2 * np.pi * 15 * t1) + np.sin(2 * np.pi * 5 * t1)

# Создание второго сигнала (15 + 1 Гц, 1000 семплов)
t2 = np.linspace(0, 1, 1000, endpoint=False)  # Время от 0 до 1 секунд
signal2 = np.sin(2 * np.pi * 150 * t2) + np.sin(2 * np.pi * 1 * t2)

print(len(signal1),len(signal2))
# Вычисляем MI-KNN
mi_value_knn = mi_knn(signal1, signal2)
print(f"Mutual Information (KNN): {mi_value_knn}")
