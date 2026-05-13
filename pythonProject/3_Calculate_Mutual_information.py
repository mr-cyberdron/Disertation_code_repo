import numpy as np
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


initial_sigs = np.load('./PREPARED_data/Untransformed_Sig/ECGnet/X_test.npy')
changed_sigs = np.load('./PREPARED_data/ECGnet/X_test.npy')

changed_sigs = np.delete(changed_sigs, [2,3], axis=1)

assert(np.shape(initial_sigs)[0] == np.shape(changed_sigs)[0])

mi_mass = []
counter = 0
for init_sig, changed_sig in zip(initial_sigs,changed_sigs):
    counter+=1
    init = init_sig.flatten()
    changed = changed_sig.flatten()
    obs_mi_val = mi_knn(init,changed)
    print(f'Calculate mutual information: {counter}/{len(initial_sigs)} - {obs_mi_val}')
    mi_mass.append(obs_mi_val)

print(f'Averaged MI compared with 12 lead ECG: {np.mean(mi_mass)}')
# 0.0	Полная независимость
# 0.1 - 0.5	Слабая зависимость
# 0.5 - 1.0	Умеренная зависимость
# 1.0 - 2.0	Сильная зависимость
# > 2.0	Очень сильная зависимость