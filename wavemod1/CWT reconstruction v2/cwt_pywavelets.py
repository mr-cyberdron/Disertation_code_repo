import numpy as np
import matplotlib.pyplot as plt
import pywt

t = np.linspace(0, 5, 1000)

signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 8 * t) * 1
scales = np.linspace(1, 300, 300)

# Compute the continuous wavelet transform
coefficients, frequencies = pywt.cwt(signal, scales, 'morl')

# Visualize the scalogram
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), extent=[t.min(), t.max(), scales[-1], scales[0]], cmap='coolwarm', aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel('Time (seconds)')
plt.ylabel('Scale')
plt.title('CWT Scalogram Using PyWavelets')
plt.show()