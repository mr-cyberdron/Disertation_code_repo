import numpy as np
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
import numpy as np
from scipy.fftpack import fft
from scipy.stats import entropy

def spectral_cross_entropy(signal1, signal2, n_fft=512):
    """
    Вычисляет спектральную кросс-энтропию между двумя сигналами.
    """
    # Вычисляем амплитудный спектр для обоих сигналов (только положительные частоты)
    spectrum1 = np.abs(fft(signal1, n=n_fft))[:n_fft // 2]
    spectrum2 = np.abs(fft(signal2, n=n_fft))[:n_fft // 2]

    # Нормализуем спектры до вероятностных распределений
    p1 = spectrum1 / np.sum(spectrum1)
    p2 = spectrum2 / np.sum(spectrum2)

    # Вычисляем кросс-энтропию (используем KL-дивергенцию)
    cross_entropy = entropy(p1, p2)

    return cross_entropy




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
    obs_mi_val = spectral_cross_entropy(init,changed)
    print(f'Calculate mutual information: {counter}/{len(initial_sigs)} - {obs_mi_val}')
    mi_mass.append(obs_mi_val)

print(f'Averaged SPECTRum Cross entropy compared with 12 lead ECG: {np.mean(mi_mass)}')

# 0.0	Спектры полностью совпадают	Два одинаковых сигнала
# 0.1 - 0.5	Незначительное отличие	Сигнал с небольшими шумами
# 0.5 - 1.0	Умеренные различия	Два похожих, но слегка смещенных сигнала
# 1.0 - 2.0	Сильные различия	Два разных сигнала с похожими частотами
# > 2.0	Полностью разные спектры	Белый шум vs. синусоида