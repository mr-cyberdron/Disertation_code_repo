import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from FILES_processing_lib import create_floder

def compute_svd_sum(data, num_vectors=5):
    if not isinstance(data, np.ndarray):
        raise ValueError("input data shpuld be NumPy.")
    if data.ndim != 2:
        raise ValueError()
    if num_vectors <= 0 or num_vectors > min(data.shape):
        raise ValueError(f"vector num should be from 1 to {min(data.shape)}.")

    U, S, Vt = np.linalg.svd(data, full_matrices=False)

    weighted_vectors = (Vt[:num_vectors, :].T * S[:num_vectors]).T  # (num_vectors, 5500)

    summed_vector = np.sum(weighted_vectors, axis=0)  # (5500,)

    # Оновлення: Додано легенди та заголовок для всієї фігури

    num_components = 5  # Доступна кількість компонент
    rows = num_components + 1 +1 # Останній графік — сумований вектор
    #
    # plt.figure(figsize=(8, rows * 1.2))  # Компактний розмір фігури
    #
    # # Заголовок для всієї фігури
    # plt.suptitle("Візуалізація компонент SVD", fontsize=14, y=0.95)
    # fsize = 10
    # # Візуалізація кожного компонента
    # for i in range(num_components):
    #     plt.subplot(rows, 1, i + 1)
    #     plt.plot(weighted_vectors[i]*-1, color='blue', label=f"SVD {i + 1}")
    #     plt.ylabel("Амплітуда", fontsize=8)
    #     plt.grid(True, linestyle='--', linewidth=0.5)
    #     plt.legend(fontsize=fsize)
    #
    # # Сумований вектор на більшому subplot
    # plt.subplot(rows, 1, (rows - 1, rows))
    # plt.plot(summed_vector*-1, color='red', label="Сумований вектор")
    # plt.xlabel("Семпли", fontsize=fsize)
    # plt.ylabel("Амплітуда", fontsize=fsize)
    # plt.grid(True, linestyle='--', linewidth=0.7)
    # plt.legend(fontsize=10)
    #
    # plt.tight_layout(rect=[0, 0, 1, 0.94])
    # plt.show()

    return summed_vector

def generate_100_svd(df_slice, n_batch_for_svd, num_svds):
    svd_mass = []

    for i in range(num_svds):
        vect_num = 5
        mass_for_svd1 = []
        mass_for_svd2 = []
        df_slice_For_svd = df_slice.sample(n=n_batch_for_svd, random_state=42, replace=True)
        for j, svd_slice_row in df_slice_For_svd.iterrows():
            sig_loaded = np.load(svd_slice_row['fpath'])
            mass_for_svd1.append(sig_loaded[:500])
            mass_for_svd2.append(sig_loaded[500:])
        mass_for_svd1 = np.array(mass_for_svd1)
        mass_for_svd2 = np.array(mass_for_svd2)
        SVD_res1 = compute_svd_sum(mass_for_svd1,num_vectors=vect_num)

        SVD_res2 = compute_svd_sum(mass_for_svd2, num_vectors=vect_num)
        svd_res = np.concatenate((SVD_res1,SVD_res2))
        svd_mass.append(svd_res)
    return svd_mass


def generate_input_data(TP_data, TN_data, SVD_mass):
    X1 = []
    X2 = []
    y = []
    counter = 0
    for i, line in TP_data.iterrows():
        X1.append(np.load(line['fpath']))
        X2.append(SVD_mass[counter])
        counter+=1
        if counter>(len(SVD_mass)-1):
            counter = 0
        y.append(1)

    for i, line in TN_data.iterrows():
        X1.append(np.load(line['fpath']))
        X2.append(SVD_mass[counter])
        counter += 1
        if counter > (len(SVD_mass) - 1):
            counter = 0
        y.append(0)

    return np.array(X1), np.array(X2), np.array(y)


for_svd_slice_size = 100
num_svds_for_pat = 100

train_df = pd.read_csv('train_data_slices.csv')
test_df = pd.read_csv('test_data_slices.csv')



Y_ref_dict = {'SVTAC': 0, 'PSVT': 0, 'BIGU': 0, 'RVH': 0, 'ANEUR': 0, 'LPFB': 0, 'WPW': 0, 'LMI': 0,
                      'LPR': 0, 'RAO/RAE': 0, 'ISCIN': 0, 'ISCIL': 0, 'ISCAS': 0, 'IPLMI': 0, 'ALMI': 0, 'LNGQT': 0,
                      'SBRAD': 0, 'CRBBB': 0, 'LAO/LAE': 0, 'PAC': 0, 'SVARR': 0, 'CLBBB': 0, 'ISCAL': 0, 'AMI': 0,
                      'ILMI': 0, 'VCLVH': 0, 'STACH': 0, '1AVB': 0, 'IRBBB': 0, 'ISC_': 0, 'PVC': 0, 'LAP': 0,
                      'LVP': 0, 'AFIB': 0, 'SARRH': 0, 'LAFB': 0, 'LVH': 0, 'ASMI': 0, 'NORM': 0, 'IMI': 0}

pat_codes = list(Y_ref_dict.keys())

X1_train = []
X2_train = []
y_train = []

counter = 0
for scp_code in pat_codes:
    print(scp_code)
    counter+=1
    print(f'{counter}/{len(pat_codes)}')

    code_df_slice_train_TP = train_df[train_df['pathologies'].str.contains(scp_code, case=False, na=False)].drop(columns=['Unnamed: 0'], errors='ignore')
    code_df_slice_test_TP = test_df[test_df['pathologies'].str.contains(scp_code, case=False, na=False)].drop(columns=['Unnamed: 0'], errors='ignore')

    Not_code_slice_train = train_df[~train_df['pathologies'].str.contains(scp_code, case=False, na=False)].drop(
        columns=['Unnamed: 0'], errors='ignore')
    Not_code_slice_test = test_df[~test_df['pathologies'].str.contains(scp_code, case=False, na=False)].drop(
        columns=['Unnamed: 0'], errors='ignore')


    Not_code_slice_train_TN = Not_code_slice_train.sample(n=len(code_df_slice_train_TP)*2, random_state=42)
    Not_code_slice_test_TN = Not_code_slice_test.sample(n=len(code_df_slice_test_TP), random_state=42)

    svd_100 = generate_100_svd(code_df_slice_train_TP, for_svd_slice_size, num_svds_for_pat)

    # y = 1 - same, y = 0 - diff
    x1train, x2train,ytrain = generate_input_data(code_df_slice_train_TP,Not_code_slice_train_TN, svd_100)
    x1test, x2test, ytest = generate_input_data(code_df_slice_test_TP, Not_code_slice_test_TN, [svd_100[0]])


    scp_code_for_fpath = scp_code.replace('/','_')
    path_to_store_test_data = f'./FinalNetData/TrainDataForFinalNET/TestData/{scp_code_for_fpath}/'
    create_floder(path_to_store_test_data)
    np.save(path_to_store_test_data+f'{scp_code_for_fpath}_X1_test.npy', x1test)
    np.save(path_to_store_test_data + f'{scp_code_for_fpath}_X2_test.npy', x2test)
    np.save(path_to_store_test_data + f'{scp_code_for_fpath}_y_test.npy', ytest)

    if len(list(X1_train)) == 0:
        X1_train = x1train
        X2_train = x2train
        y_train = ytrain
    else:
        X1_train = np.concatenate((X1_train,x1train))
        X2_train = np.concatenate((X2_train,x2train))
        y_train = np.concatenate((y_train,ytrain))

    print(np.shape(X1_train))
    print(np.shape(X2_train))
    print(np.shape(y_train))

    input('sss')

print(np.shape(X1_train))
print(np.shape(X2_train))
print(np.shape(y_train))

np.save('./FinalNetData/TrainDataForFinalNET/TrainData/X1_train.npy', X1_train)
np.save('./FinalNetData/TrainDataForFinalNET/TrainData/X2_train.npy', X2_train)
np.save('./FinalNetData/TrainDataForFinalNET/TrainData/y_train.npy', y_train)




