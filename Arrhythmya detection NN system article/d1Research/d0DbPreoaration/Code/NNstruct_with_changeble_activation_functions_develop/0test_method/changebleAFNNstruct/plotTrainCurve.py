import matplotlib.pyplot as plt
import numpy as np




sig1 = np.load('./testCurve/ShiftRelu.npy')
sig2 = np.load('./testCurve/PAramRelu.npy')
sig3 = np.load('./testCurve/Relu.npy')
sig4 = np.load('./testCurve/PiecwiceActivations.npy')
sig5 = np.load('./testCurve/Sigmoid.npy')
sig6 = np.load('./testCurve/CustomSigmoid.npy')
sig7 = np.load('./testCurve/Mycustomfunc.npy')

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Пример данных: замените эти массивы на ваши реальные данные
# Предполагается, что у вас есть несколько кривых обучения, хранящихся в списках или массивах numpy
learning_curves = {
    'ReLU': sig3,
# # 'ReLU із зсувом': sig1,
# # 'Параметрична ReLU': sig2,
# 'Кусково визначена лінійна функція ': sig4,
    'Розроблена функція зі збіжністю':sig7

}

# Определяем ось x (например, эпохи или шаги времени)
x = np.arange(len(next(iter(learning_curves.values()))))

# Строим график кривых обучения
plt.figure(figsize=(12, 6))

for name, y in learning_curves.items():
    # Построение исходных данных
    plt.plot(x, y, label=name)

    # Аппроксимация кривой с помощью полиномиальной регрессии
    z = np.polyfit(x, y, 20)  # Полином 3-й степени
    p = np.poly1d(z)
    plt.plot(x, p(x), linestyle='--')  # Построение аппроксимации

    # Расчет площади под кривой
    auc = integrate.trapz(y, x)
    print(f'Площадь под {name}: {auc:.2f}')

    # Расчет коэффициента вариации для каждой кривой
    mean_y = np.mean(y)
    std_y = np.std(y)
    cv_y = std_y / mean_y
    print(f'Коэффициент вариации для {name}: {cv_y:.4f}')

# Настройка графика
plt.title('Криві навчання з апроксимаціями')
plt.xlabel('Епохи')
plt.ylabel('Значення функції втрат')
plt.legend()
plt.grid(True)
plt.show()
