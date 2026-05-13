import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def generate_svd_vector(mass_for_svd):

    U, S, Vt = np.linalg.svd(mass_for_svd, full_matrices=False)
    top_3_singular_values = S[:3]
    top_3_right_singular_vectors = Vt[:3, :]

    # rotation avoid
    for i in range(3):
        if top_3_right_singular_vectors[i, 0] < 0:
            top_3_right_singular_vectors[i] *= -1

    weighted_sum = np.sum(top_3_singular_values[:, np.newaxis] * top_3_right_singular_vectors, axis=0)

    return weighted_sum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

class SiameseNetwork1D(nn.Module):
    def __init__(self):
        super(SiameseNetwork1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10, stride=1)
        self.fc1 = nn.Linear(64 * 491, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward_once(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        diff = torch.abs(output1 - output2)

        similarity_score = torch.sigmoid(self.fc3(diff))

        return similarity_score



model = SiameseNetwork1D().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

print('load npy')
X1 = np.load('trainDAta/Train_X1.npy')
X2 = np.load('trainDAta/Train_X2.npy')
Y = np.load('trainDAta/Train_Y.npy')

print('load tensor')
X1_tensor = torch.tensor(X1, dtype=torch.float32).unsqueeze(1).to(device)  # добавляем измерение канала
X2_tensor = torch.tensor(X2, dtype=torch.float32).unsqueeze(1).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)


dataset = TensorDataset(X1_tensor, X2_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=150, shuffle=True)



def train_one_epoch(model, dataloader, criterion, optimizer):
    print('epoch')
    model.train()
    total_loss = 0
    for input1, input2, label in dataloader:
        input1, input2, label = input1.to(device), input2.to(device), label.to(device)
        optimizer.zero_grad()
        similarity_score = model(input1, input2)
        loss = criterion(similarity_score, label.unsqueeze(1))  # добавляем измерение для label
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Основной цикл обучения
num_epochs = 100
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, dataloader, criterion, optimizer)
    print(f"Эпоха [{epoch+1}/{num_epochs}], Средняя потеря: {avg_loss:.4f}")


# тест

norm_data_for_svd = np.load('trainDAta/norm_test_QRS_for_svd.npy')
print(np.shape(norm_data_for_svd))
lvp_data_for_svd = np.load('trainDAta/lvp_test_QRS_for_svd.npy')
lap_data_for_svd = np.load('trainDAta/lap_test_QRS_for_svd.npy')
lvp_lap_data_for_svd = np.load('trainDAta/lap_lvp_test_QRS_for_svd.npy')

norm_data_for_test = np.load('trainDAta/norm_test_QRS.npy')
lvp_data_for_test = np.load('trainDAta/lvp_test_QRS.npy')
lap_data_for_test = np.load('trainDAta/lap_test_QRS.npy')
lvp_lap_data_for_test = np.load('trainDAta/lap_lvp_test_QRS.npy')

norm_test_svd = generate_svd_vector(norm_data_for_svd)
lvp_test_svd = generate_svd_vector(lvp_data_for_svd)
lap_test_svd = generate_svd_vector(lap_data_for_svd)
lvp_lap_test_svd = generate_svd_vector(lvp_lap_data_for_svd)



def test_case (svd_to_case_mass, case_true_mass, cf1,cf2,cf3):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    svd = torch.tensor(svd_to_case_mass, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    def process_case (svd, case_mass,not_case_mas, true_pred,false_pred, tp,tn,fp,fn):
        print('process case')
        for case in case_mass:
            qrs = torch.tensor(case, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            similarity_score = model(svd, qrs)
            if torch.round(similarity_score[0][0]) == true_pred:
                tp += 1
            else:
                fn += 1
        print('process case2')
        for case2 in not_case_mas:
            qrs = torch.tensor(case2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            similarity_score = model(svd, qrs)
            if torch.round(similarity_score[0][0]) == false_pred:
                tn += 1
            else:
                fp += 1

        return tp,tn,fp,fn

    case_false_mass = np.concatenate([cf1,cf2,cf3])


    tp,tn,fp,fn = process_case(svd, case_true_mass, case_false_mass,1,0, tp,tn,fp,fn)

    return tp,tn,fp,fn


def calculate_metrics_and_plot(tp, tn, fp, fn):
    # Сумма всех случаев
    total = tp + tn + fp + fn

    # Пересчет в проценты
    tp_percent = (tp / total) * 100
    tn_percent = (tn / total) * 100
    fp_percent = (fp / total) * 100
    fn_percent = (fn / total) * 100

    # Расчет метрик
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    accuracy = (tp + tn) / total

    # Вывод результатов
    print(f"TP: {tp_percent:.2f}%")
    print(f"TN: {tn_percent:.2f}%")
    print(f"FP: {fp_percent:.2f}%")
    print(f"FN: {fn_percent:.2f}%")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Accuracy: {accuracy:.2f}")



plt.figure()
plt.plot(generate_svd_vector(lvp_lap_data_for_test[:150])*-1)
plt.show()

model = model.to("cpu")
print('------------------------NORM SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(norm_data_for_svd), norm_data_for_test,lvp_data_for_test,lap_data_for_test,lvp_lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_data_for_svd), lvp_data_for_test,norm_data_for_test,lap_data_for_test,lvp_lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lap_data_for_svd),lap_data_for_test, norm_data_for_test,lvp_data_for_test,lvp_lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_lap_data_for_svd),lvp_lap_data_for_test, norm_data_for_test,lvp_data_for_test,lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)

print('-----------------------------2/////////////////////-----------------------------------------------------------')

print('------------------------NORM SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(norm_data_for_test[:150]), norm_data_for_test,lvp_data_for_test,lap_data_for_test,lvp_lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_data_for_test[150:300]), lvp_data_for_test,norm_data_for_test,lap_data_for_test,lvp_lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lap_data_for_test[300:450]),lap_data_for_test, norm_data_for_test,lvp_data_for_test,lvp_lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_lap_data_for_test[:150]),lvp_lap_data_for_test, norm_data_for_test,lvp_data_for_test,lap_data_for_test)
calculate_metrics_and_plot(tp,tn,fp,fn)


print('-----------------------------3/////////////////////-----------------------------------------------------------')

print('------------------------NORM SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(norm_data_for_test[:150]), norm_data_for_svd,lvp_data_for_svd,lap_data_for_svd,lvp_lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_data_for_test[150:300]), lvp_data_for_svd, norm_data_for_svd,lap_data_for_svd, lvp_lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lap_data_for_test[300:450]), lap_data_for_svd, lvp_data_for_svd, norm_data_for_svd, lvp_lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_lap_data_for_test[:150]),lvp_lap_data_for_svd, norm_data_for_svd,lvp_data_for_svd,lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)

print('-----------------------------4/////////////////////-----------------------------------------------------------')

print('------------------------NORM SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(norm_data_for_svd), norm_data_for_svd,lvp_data_for_svd,lap_data_for_svd,lvp_lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_data_for_svd), lvp_data_for_svd, norm_data_for_svd,lap_data_for_svd, lvp_lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lap_data_for_svd), lap_data_for_svd, lvp_data_for_svd, norm_data_for_svd, lvp_lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)
print('------------------------LVP LAP SCORE-----------------')
tp,tn,fp,fn = test_case(generate_svd_vector(lvp_lap_data_for_svd),lvp_lap_data_for_svd, norm_data_for_svd,lvp_data_for_svd,lap_data_for_svd)
calculate_metrics_and_plot(tp,tn,fp,fn)





