import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2

t = np.linspace(0, 5, 1000)

signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 8 * t) * 1
scales = np.linspace(1, 300, 300)


cwt_coefficients = cwt(signal, morlet2, scales)

# Visualizing the scalogram
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(cwt_coefficients), extent=[t.min(), t.max(), scales.max(), scales.min()], cmap='coolwarm', aspect='auto')
plt.colorbar(label='Magnitude')
plt.xlabel('Time (seconds)')
plt.ylabel('Scale')
plt.title('CWT Scalogram using SciPy')
plt.show()