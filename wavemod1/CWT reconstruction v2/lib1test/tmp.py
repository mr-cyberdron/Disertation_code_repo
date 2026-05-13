import matplotlib.pyplot as plt
import numpy as np

# Создаем данные для графика
x = np.linspace(0, 10, 100)  # Основная ось X (например, время в секундах)
y = np.sin(x)                # Данные для оси Y

fig, ax1 = plt.subplots()

# Создаем первую ось X
ax1.plot(x, y, 'b-')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Sine value', color='b')

# Создаем вторую ось X, которая будет совмещена с первой
ax2 = ax1.twiny()
ax2.set_xlabel('Time (minutes)')
new_tick_locations = np.linspace(min(x), max(x), num=10)

# Переводим значения секунд в минуты для второй оси X
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels((new_tick_locations / 60).round(2))

plt.show()