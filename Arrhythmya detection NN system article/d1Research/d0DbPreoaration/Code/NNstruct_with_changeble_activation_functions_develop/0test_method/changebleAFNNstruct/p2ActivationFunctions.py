import torch.nn as nn
import torch

class CustomSigmoid(nn.Module):
    def __init__(self, initial_a=1.0, initial_b=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial_a))
        self.b = nn.Parameter(torch.tensor(initial_b))

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.a * (x + self.b)))

class CustomPReLU(nn.Module):
    def __init__(self, a1=1.0, a2=0.25):
        super().__init__()
        # Определим обучаемые параметры `a1` и `a2`
        self.a1 = nn.Parameter(torch.tensor(a1))
        self.a2 = nn.Parameter(torch.tensor(a2))

    def forward(self, x):
        # Реализация параметрической ReLU с двумя параметрами
        return torch.where(x > 0, self.a1 * x, self.a2 * x)

class ShiftReLU(nn.Module):
    def __init__(self, a1=1.0, a2=0.25, b=0.0):
        super().__init__()
        # Определяем обучаемые параметры: a1, a2 и b
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor(a2, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        # Реализация функции активации по формуле:
        # y = a1 * (x - b), если x > b
        # y = a2 * (x - b), если x <= b
        return torch.where(x > self.b, self.a1 * (x - self.b), self.a2 * (x - self.b))




class CustomPiecewiseActivation(nn.Module):
    def  __init__(self):
        super(CustomPiecewiseActivation, self).__init__()

        # Определяем обучаемые параметры для каждого куска
        # Параметры наклона (a1, a2, a3) и сдвига (b1, b2, b3)
        self.a1 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(0.0))

        self.a2 = nn.Parameter(torch.tensor(5.0))
        self.b2 = nn.Parameter(torch.tensor(0.0))

        self.a3 = nn.Parameter(torch.tensor(1.0))
        self.b3 = nn.Parameter(torch.tensor(0.0))

        # Обучаемые точки перехода между кусками
        self.t1 = nn.Parameter(torch.tensor(-1.0))
        self.t2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # Вычисляем значения функции в каждом интервале
        # Первый кусок: x <= t1
        y1 = self.a1 * x + self.b1
        # Второй кусок: t1 < x <= t2
        y2 = self.a2 * x + self.b2
        # Третий кусок: x > t2
        y3 = self.a3 * x + self.b3

        # Определяем значения функции с учетом интервалов
        y = torch.where(x <= self.t1, y1, torch.where(x <= self.t2, y2, y3))
        return y

#SawtoothActivation

class SawtoothActivation(nn.Module):
    def __init__(self, a1=1.0, a2=1.0):
        """
        Конструктор для функции активации с обучаемыми параметрами наклона.

        Args:
            a1 (float): Начальное значение наклона в области x < -1.
            a2 (float): Начальное значение наклона в области x > 1.
        """
        super(SawtoothActivation, self).__init__()
        # Определяем обучаемые параметры a1 и a2
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor(a2, dtype=torch.float32))

    def forward(self, x):
        """
        Вычисление функции активации.

        Args:
            x (torch.Tensor): Входной тензор.

        Returns:
            torch.Tensor: Выходной тензор.
        """
        # Область x < -1
        left = torch.where(x < 0, torch.sigmoid(self.a1) * x * (-1), torch.zeros_like(x))

        # Область -1 <= x <= 1
        middle = torch.where((x >= 0) & (x <= 1), x, torch.zeros_like(x))

        # Область x > 1
        right = torch.where(x > 1, torch.sigmoid(self.a2) *(-1) * (x - 1) + 1, torch.zeros_like(x))

        return left+middle+right

class CustomPolynom(nn.Module):
    def __init__(self):
        super(CustomPolynom, self).__init__()
        # Объявляем обучаемые параметры для полинома третьей степени: a, b, c и d
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # Полиномиальная активационная функция: a * x^3 + b * x^2 + c * x + d
        return self.a * x ** 3 + self.b * x ** 2 + self.c * x + self.d


class CustomPolynom2(nn.Module):
    def __init__(self):
        super(CustomPolynom2, self).__init__()
        # Объявляем обучаемые параметры для полинома третьей степени: a, b, c и d
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))
        self.e = nn.Parameter(torch.randn(1))
        self.f = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # Полиномиальная активационная функция: a * x^3 + b * x^2 + c * x + d
        return self.a * x ** 5 + self.b * x ** 4 + self.c * x ** 3 + self.d * x ** 2 + self.e * x + self.f



class CustomSigmoidForAllNeurons(nn.Module):
    def __init__(self, num_neurons, initial_a=1.0, initial_b=0.0):
        super().__init__()
        self.a = nn.Parameter(torch.ones(num_neurons)*initial_a)
        self.b = nn.Parameter(torch.ones(num_neurons)*initial_b)

    def forward(self, x):
        return torch.sigmoid(self.a * x + self.b)