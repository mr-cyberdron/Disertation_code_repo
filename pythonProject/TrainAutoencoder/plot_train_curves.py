import numpy as np
import matplotlib.pyplot as plt

# Завантаження масивів
mse_values = np.load('recon_loss_mass2.npy')
bce_values = np.load('class_loss_mass2.npy')

# Епохи
epochs = np.arange(1, len(mse_values) + 1)

# Графік
plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
plt.subplot(2,1,1)
plt.plot(epochs, mse_values, label='Середньоквадратична помилка')
plt.title('Графік навчання розобленого автоенкодера', fontsize=14, fontweight='bold')
plt.ylabel('Втрати', fontsize=12, fontweight='bold')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(epochs, bce_values, label='Перехресна ентропія')
plt.xlabel('Епоха', fontsize=12, fontweight='bold')
plt.ylabel('Втрати', fontsize=12, fontweight='bold')
plt.grid(True)

# Водяний знак TimeShenoman
for y in np.linspace(min(min(mse_values), min(bce_values)),
                     max(max(mse_values), max(bce_values)), 4):
    plt.text(len(epochs)/2, y, 'TimeShenoman',
             fontsize=10, color='gray', alpha=0.1,
             ha='center', va='center', rotation=15)

plt.legend()
plt.show()
