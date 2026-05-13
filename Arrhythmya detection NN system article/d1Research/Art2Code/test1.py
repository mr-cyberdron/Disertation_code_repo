import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import wfdb

# Часовий інтервал
time_points = np.linspace(0, 0.1, 20)  # Час у секундах

sigs, fields = wfdb.rdsamp('s0306lre', channels=[12, 13, 14], sampfrom=0, sampto=2000)

def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()



fig = plt.figure(figsize=(8, 5))

signal_x = filter_ecg(sigs[:,0],1000)
signal_y =filter_ecg(sigs[:,1],1000)
signal_z = filter_ecg(sigs[:,2],1000)




# Основной 3D-график: QRS loop
ax3d = fig.add_subplot(121, projection='3d')
ax3d.plot(signal_x, signal_y, signal_z, label='QRS vector', color='green', linewidth=1)
# ax3d.set_xlim([-0.5, 1.2])  # обмеження для осі X
# ax3d.set_ylim([-1, 1.2])
ax3d.set_zlim([-1, 1])
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('3D QRS Loop')
ax3d.legend()


# Візуалізація координатних осей
ax3d.quiver(0, 0, 0, 2, 0, 0, color='red', label='Vx', arrow_length_ratio=0.1)
ax3d.quiver(0, 0, 0, 0, 0.8, 0, color="#90EE90", label='Vy', arrow_length_ratio=0.1)
ax3d.quiver(0, 0, 0, 0, 0, -0.8, color='blue', label='Vz', arrow_length_ratio=0.1)


# Сигналы справа в столбик
fig.subplots_adjust(wspace=0.5)  # Увеличиваем расстояние между графиками
fig.subplots_adjust(left=0.0, right=0.975, top=0.930, bottom=0.105, wspace=0.0, hspace=0.480)
# Сигнал X
t = np.array(list(range(len(signal_x))))/1000
ax_x = fig.add_subplot(333)
ax_x.plot(t,signal_x, color='red', label='Lead VX')
# ax_x.set_xlabel('Time')
ax_x.set_ylabel('Amplitude [mV]')
ax_x.set_title('Signal VX')
# ax_x.legend()

# Сигнал Y
ax_y = fig.add_subplot(336)
ax_y.plot(t,signal_y, color="#90EE90", label='Lead VY')
# ax_y.set_xlabel('Time')
ax_y.set_ylabel('Amplitude [mV]')
ax_y.set_title('Signal VY')
# ax_y.legend()

# Сигнал Z
ax_z = fig.add_subplot(339)
ax_z.plot(t,signal_z, color='blue', label='Lead VZ')
ax_z.set_xlabel('Time [sec]')
ax_z.set_ylabel('Amplitude [mV]')
ax_z.set_title('Signal VZ')
# ax_z.legend()

# Отображение графиков
plt.tight_layout()
plt.show()