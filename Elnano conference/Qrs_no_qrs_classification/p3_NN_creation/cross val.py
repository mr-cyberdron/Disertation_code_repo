import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import models
from FILES_processing_lib import scandir

# =====================
# CUDA
# =====================
print(torch.cuda.is_available())
print(torch.version.cuda)

counting_base = 'GPU'
if counting_base == 'GPU':
    torch.set_default_device("cuda")

# =====================
# Custom loss
# =====================
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        x = (targets - inputs) * -1
        pokazatel_stepeni = torch.sum(torch.abs(torch.maximum(0.05 * x, x)))
        loss_body = torch.sum(torch.abs(targets - inputs))
        loss = loss_body * pokazatel_stepeni
        return loss

# =====================
# Metrics
# =====================
def evaluate_metrics(model, dataloader):
    tp = tn = fp = fn = 0

    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:

            if mod == 'v2':
                data = data.unsqueeze(1)


            data = data.float().cuda()
            labels = labels.float().cuda()

            output = model(data).squeeze()
            preds = torch.round(output)

            for p, y in zip(preds, labels):
                if p == 1 and y == 1:
                    tp += 1
                elif p == 0 and y == 0:
                    tn += 1
                elif p == 1 and y == 0:
                    fp += 1
                elif p == 0 and y == 1:
                    fn += 1

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return sensitivity, specificity, accuracy

# =====================
# Block k-fold
# =====================
def block_kfold_indices(n_samples, n_splits=5):
    block_size = n_samples // n_splits
    indices = np.arange(n_samples)

    folds = []
    for i in range(n_splits):
        start = i * block_size
        end = (i + 1) * block_size if i != n_splits - 1 else n_samples
        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))
        folds.append((train_idx, val_idx))

    return folds

# =====================
# Load data
# =====================
print('load data')
np_data = np.load('D:/Bases/1QRS_noise_base/prepared/QRS_Noise_prep.npz')
X_mass = np_data['x']
Y_mass = np_data['y']


# при необходимости можно ограничить
# target_len = 100000
# perm = np.random.permutation(len(X_mass))[:target_len]
# X_mass = X_mass[perm]
# Y_mass = Y_mass[perm]

step = 3
X_mass = X_mass[::step]
Y_mass = Y_mass[::step]

# =====================
# CV params
# =====================
batch_size = 100
epochs = 2
n_folds = 5
lr = 0.005

folds = block_kfold_indices(len(X_mass), n_folds)

cv_results = []

# =====================
# Cross-validation
# =====================
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f'\n========== Fold {fold_idx + 1}/{n_folds} ==========')

    X_train, y_train = X_mass[train_idx], Y_mass[train_idx]
    X_val, y_val = X_mass[val_idx], Y_mass[val_idx]

    trainloader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)),
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device='cuda')
    )

    valloader = torch.utils.data.DataLoader(
        list(zip(X_val, y_val)),
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device='cuda')
    )

    mod = 'v1'
    model = models.QRSnet1().cuda()
    # mod = 'v2'
    # model = models.QRSnet2().cuda()


    criterion = CustomLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # ---- training ----
    for e in range(epochs):
        model.train()
        running_loss = 0
        progress = tqdm(trainloader, desc=f'epoch {e+1}/{epochs}', leave=False)

        for data, labels in progress:
            if mod == 'v2':
                data = data.unsqueeze(1)

            data = data.float().cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        progress.close()

    # ---- evaluation ----
    sens, spec, acc = evaluate_metrics(model, valloader)
    cv_results.append((sens, spec, acc))

    print(f'Sensitivity: {sens * 100:.2f}')
    print(f'Specificity: {spec * 100:.2f}')
    print(f'Accuracy: {acc * 100:.2f}')

# =====================
# Mean results
# =====================
mean_sens = np.mean([r[0] for r in cv_results])
mean_spec = np.mean([r[1] for r in cv_results])
mean_acc = np.mean([r[2] for r in cv_results])

print('\n========== Mean CV results ==========')
print(f'Sensitivity: {mean_sens * 100:.2f}')
print(f'Specificity: {mean_spec * 100:.2f}')
print(f'Accuracy: {mean_acc * 100:.2f}')
