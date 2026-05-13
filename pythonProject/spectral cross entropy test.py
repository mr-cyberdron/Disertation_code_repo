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

fs = 1000  # Частота дискретизации (Гц)

# Создание первого сигнала (15 + 5 Гц, 3000 семплов)
t1 = np.linspace(0, 3, 3000, endpoint=False)  # Время от 0 до 3 секунд
signal1 = np.sin(2 * np.pi * 15 * t1) + np.sin(2 * np.pi * 5 * t1)

# Создание второго сигнала (15 + 1 Гц, 1000 семплов)
t2 = np.linspace(0, 1, 1000, endpoint=False)  # Время от 0 до 1 секунд
signal2 = np.sin(2 * np.pi * 15 * t2) + np.sin(2 * np.pi * 5 * t2)

print(len(signal1),len(signal2))

# Вычисляем спектральную кросс-энтропию между двумя ранее созданными сигналами
cross_entropy_value = spectral_cross_entropy(signal1, signal2)
print(cross_entropy_value)
