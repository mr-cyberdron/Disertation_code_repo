import torch
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import numpy as np

def plotActivations(*functions, fnames = [], af = None):


    x_vals = torch.linspace(-10, 10, 100)
    plt.figure(figsize=(12, 6))

    if af:
       af_ex = af
       af_y = af_ex.forward(x_vals)
       plt.plot(x_vals.detach().numpy(), af_y.detach().numpy(),
                label=f'Функція активації до навчання', linestyle='--', color='gray')

    y_storing_mass = []
    counter = 0
    for func in functions:
        counter+=1
        y_vals = func(x_vals)
        y_storing_mass.append(y_vals.detach().numpy())
        if fnames:
            plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy(),
                     label=f'{fnames[counter-1]}.')
        else:
            plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy(),
                 label=f'{counter}.')

    # plt.plot(x_vals.detach().numpy(), np.mean(y_storing_mass, axis=0),
    #          label=f'AF Averaged',  linestyle='-.', color='black')

    plt.title("Функція активації")
    plt.xlabel("x")
    plt.ylabel("Func(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_layer_sigmoids(layer_activation, layer_name):
    print(f'plotting {layer_name}')
    def plot_sigmoid(alpha, beta, neuron_idx, layer_name):
        x = np.linspace(-10, 10, 100)  # Диапазон значений для x
        y = 1 / (1 + np.exp(-alpha * x - beta))  # Сигмоидальная функция

        plt.plot(x, y, label=f'Neuron {neuron_idx}')
        plt.title(f'Sigmoids in Layer {layer_name}')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()

    num_neurons = layer_activation.a.size(0)
    plt.figure(figsize=(10, 6))
    for i in range(num_neurons):
        alpha = layer_activation.a[i].item()
        beta = layer_activation.b[i].item()
        plot_sigmoid(alpha, beta, neuron_idx=i, layer_name=layer_name)
    plt.show()


def trainNNepoch(model, train_dataloader, loss_fn, optimizer, device = None):
    if device:
        model.to(device)
    model.train()
    for X_batch, y_batch in train_dataloader:
        if device:
            X_batch = X_batch.to(device)
            # y_batch = y_batch.to(device).unsqueeze(1)
            y_batch = y_batch.to(device)


        optimizer.zero_grad()
        y_pred = model(X_batch)


        loss = loss_fn(y_pred, y_batch)
        # print(loss)



        loss.backward()
        optimizer.step()
    return model



def testACC(model, X_test, y_test, epoch):
    device = torch.device("cpu")
    model.to(device)
    # model.eval()
    y_pred = model(X_test)
    true_counter = 0
    counter = 0
    for i, ii in zip(y_pred, y_test):
        i = np.round(i.detach().numpy())[0]
        if i == ii:
            true_counter+=1
        counter+=1
    acc = true_counter/counter
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc * 100))

def test_metrics(model, X_test, y_test, epoch):
    import torch
    import numpy as np

    device = torch.device("cpu")
    model.to(device)
    model.eval()  # Переводим модель в режим оценки

    with torch.no_grad():  # Отключаем вычисление градиентов
        y_pred = model(X_test)

    # Инициализация счетчиков
    TP = 0  # Истинно положительные
    TN = 0  # Истинно отрицательные
    FP = 0  # Ложно положительные
    FN = 0  # Ложно отрицательные

    # Преобразование предсказаний и истинных меток в numpy массивы
    y_pred_np = y_pred.detach().numpy()
    y_test_np = y_test.detach().numpy()

    # Округление предсказаний до 0 или 1
    y_pred_np = np.round(y_pred_np)




    counter = 0
    # Проходим по всем предсказаниям и истинным меткам
    for pred, true in zip(y_pred_np, y_test_np):
        pred_label = int(pred[0])
        true_label = int(true)

        if pred_label == 1 and true_label == 1:
            TP += 1
        elif pred_label == 0 and true_label == 0:
            TN += 1
        elif pred_label == 1 and true_label == 0:
            FP += 1
        elif pred_label == 0 and true_label == 1:
            FN += 1


    # Расчет метрик
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if (TP + FN) > 0:
        sensitivity = TP / (TP + FN)
    else:
        sensitivity = 0
    if (TN + FP) > 0:
        specificity = TN / (TN + FP)
    else:
        specificity = 0

    print(TN)
    print(TN+FP)

    # Вывод результатов
    print(f"Эпоха {epoch}:")
    print(f"Точность модели: {accuracy * 100:.2f}%")
    print(f"Чувствительность (Полнота): {sensitivity * 100:.2f}%")
    print(f"Специфичность: {specificity * 100:.2f}%")


def calc_der(act, x_vals):
    af_y = act.forward(x_vals)
    dy_dx = np.gradient(af_y.detach().numpy(), x_vals.detach().numpy())
    return af_y.detach().numpy(), dy_dx

def calc_der2(act, x_vals):
    y, dy_dx = calc_der(act,x_vals)
    d2y_dx = np.gradient(dy_dx, x_vals.detach().numpy())
    return y, dy_dx, d2y_dx

def calc_layers_AF_deviation_cofs(cofs_mass, initial_cofs_mass = []):
    #!!!! додати врахувати напрям коефіцієнтів!!!!
    cofs_mass = np.array(cofs_mass)
    cofs_t = cofs_mass.T
    if not initial_cofs_mass:
        initial_cofs_mass = np.mean(cofs_t, axis=1)

    assert np.shape(initial_cofs_mass)[0] == np.shape(cofs_mass)[1], "Initial cofs len should be equal the num of cofs in AF"

    deviation_cofs_matrix = []
    for cof_l_mass, cof_l_init in zip(cofs_t, initial_cofs_mass):
        deviation_cof_for_l = []
        c_max = np.max(np.append(cof_l_mass,cof_l_init))
        c_min = np.min(np.append(cof_l_mass,cof_l_init))
        dev_cof_0 = calc_layer_AF_deviation_by_cof(c_max,c_min,cof_l_mass[0],cof_l_init,cof_l_init)
        deviation_cof_for_l.append(dev_cof_0)
        for i in range(1,len(cof_l_mass)):
            dev_cof = calc_layer_AF_deviation_by_cof(c_max,c_min,cof_l_mass[i],cof_l_mass[i-1],cof_l_init)
            deviation_cof_for_l.append(dev_cof)

        deviation_cofs_matrix.append(deviation_cof_for_l)
    result_dev_cofs = np.mean(deviation_cofs_matrix, axis=0)
    print(result_dev_cofs)
    return result_dev_cofs

def calc_layer_AF_deviation_by_cof(cof_max, cof_min, cof_total, cof_prev, cof_initial):
    max_deviation_cof = abs(cof_max-cof_min)+0.00000001
    layer_deviaton_cof = abs(cof_total-cof_prev)
    from_initial_deviation_cof = abs(cof_total-cof_initial)
    layer_deviation_part = layer_deviaton_cof/max_deviation_cof
    from_init_deviation_part = from_initial_deviation_cof/max_deviation_cof
    return (0.6*layer_deviation_part)+(0.4 * from_init_deviation_part)




class CountExecutionTime:
    def __init__(self):
        self.start_time = None
        self.timer_name = 'process'

    def start(self, timer_name = 'process'):
        self.start_time = time.time()
        self.timer_name = timer_name


    def printTime(self):
        if self.start_time:
            end_time = time.time()
            execution_time = end_time - self.start_time
            formatted_time = str(timedelta(seconds=execution_time))
            print(f'Execution time of {self.timer_name} is {formatted_time}')
        else:
            raise AttributeError('No Timer start!')


