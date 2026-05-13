import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FILES_processing_lib import scandir
import random

qrs_avg_dir = 'D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/QRS_averaged/'

metadata_df = pd.read_csv(qrs_avg_dir+'AVG_QRS_metadata.csv')

qrs_avg_files = scandir(qrs_avg_dir,ext='.npy')

norm_data_slice = metadata_df[metadata_df['scp_codes'] == "['NORM', 'SR']"].sample(n=2000, random_state=42).reset_index(drop=True)
LAP_data_slice = metadata_df[metadata_df['scp_codes'] == "['LAP']"]
LVP_data_slice = metadata_df[metadata_df['scp_codes'] == "['LVP']"]
LAP_LVP_data_slice = metadata_df[metadata_df['scp_codes'] == "['LAP', 'LVP']"]

def generate_svd_vector(slice_df_for_svd):
    nums_list = slice_df_for_svd['num'].to_numpy()
    samples_for_svd_mass = []
    for num in nums_list:
        qrs_averaged = np.load(qrs_avg_dir+str(num)+'_AVG_QRS_500_hz.npy')
        if list(samples_for_svd_mass):
            samples_for_svd_mass = np.concatenate([samples_for_svd_mass,[qrs_averaged]], axis=0)
        else:
            samples_for_svd_mass = np.array([qrs_averaged])

    U, S, Vt = np.linalg.svd(samples_for_svd_mass, full_matrices=False)
    top_3_singular_values = S[:3]
    top_3_right_singular_vectors = Vt[:3, :]

    # rotation avoid
    for i in range(3):
        if top_3_right_singular_vectors[i, 0] < 0:
            top_3_right_singular_vectors[i] *= -1

    weighted_sum = np.sum(top_3_singular_values[:, np.newaxis] * top_3_right_singular_vectors, axis=0)

    return weighted_sum


def generate_mass(df_slice):
    slice_mass = []
    for i, row in df_slice.iterrows():
        qrs_mass = np.load(qrs_avg_dir+str(row['num'])+'_AVG_QRS_500_hz.npy')
        qrs_mass = np.array([qrs_mass])
        if list(slice_mass):
            slice_mass = np.concatenate([slice_mass,qrs_mass])
        else:
            slice_mass = qrs_mass
    return slice_mass


norm_data_train = generate_mass(norm_data_slice.iloc[:1000])
LAP_train = generate_mass(LAP_data_slice.iloc[:650])
LVP_train = generate_mass(LVP_data_slice.iloc[:650])
LAP_LVP_train = generate_mass(LAP_LVP_data_slice[:200])


norm_test = generate_mass(norm_data_slice.iloc[1820:])
lap_test = generate_mass(LAP_data_slice.iloc[650:])
lvp_test = generate_mass(LVP_data_slice.iloc[650:])
lap_lvp_test = generate_mass(LAP_LVP_data_slice[200:])

# lap norm test
X_train = np.concatenate((norm_data_train,LAP_train,LVP_train,LAP_LVP_train))
Y_train = np.concatenate((np.zeros(len(norm_data_train)),
                          np.ones(len(LAP_train)),
np.ones(len(LVP_train))*2,
np.ones(len(LAP_LVP_train))*3

                          ))

X_test = np.concatenate((norm_test,lap_test,lvp_test,lap_lvp_test))
Y_test = np.concatenate((np.zeros(len(norm_test)),
                          np.ones(len(lap_test)),
np.ones(len(lvp_test))*2,
np.ones(len(lap_lvp_test))*3

                          ))

def fix_y(y_mass):
    new_y = []
    for i in y_mass:
        if i == 0:
            new_y.append([1,0,0,0])
        if i == 1:
            new_y.append([0,1,0,0])
        if i == 2:
            new_y.append([0,0,1,0])
        if i == 3:
            new_y.append([0,0,0,1])
    return np.array(new_y)

Y_train = fix_y(Y_train)
Y_test = fix_y(Y_test)


def generateX1vect(pat_mass, size):
    def return_svd_res(input_mass):
        U, S, Vt = np.linalg.svd(input_mass, full_matrices=False)
        top_3_singular_values = S[:3]
        top_3_right_singular_vectors = Vt[:3, :]
        weighted_sum = np.sum(top_3_singular_values[:, np.newaxis] * top_3_right_singular_vectors, axis=0)
        return weighted_sum

    X1_vect = []
    for i in range(size):
        pat_sample = pat_mass[np.random.choice(pat_mass.shape[0], 100, replace=False)]

        pat_svd_slice = return_svd_res(pat_sample)
        X1_vect.append(pat_svd_slice)

    return np.array(X1_vect)


np.save('./Art2TestNN/trainDAta/EcgNET2/X_train.npy',X_train)
# X1_lap_train = generateX1vect(LAP_train, len(X_lap_train))
# np.save('./Art2TestNN/trainDAta/EcgNET2/X1_lap_train.npy',X1_lap_train)
np.save('./Art2TestNN/trainDAta/EcgNET2/Y_train.npy',Y_train)

np.save('./Art2TestNN/trainDAta/EcgNET2/X_test.npy',X_test)
# np.save('./trainDAta/SiamseNet/LAP/X1_lap_test.npy',X1_lap_train[:len(X_lap_test)])
np.save('./Art2TestNN/trainDAta/EcgNET2/Y_test.npy',Y_test)

input('ss')





# counter = 0
# for i in generate_mass(LVP_train):
#     print(f'{counter}/{len(LAP_train)}')
#     counter+=1
#     plt.figure()
#     plt.plot(i)
#     plt.grid(True)
#     plt.title("Кардіоцикл з ППШ")
#     plt.xlabel("Відлики")
#     plt.ylabel("Амплітуда")
#     plt.show()




def generate_paring(df_slice, svd_pack, code):
    print('paring')
    num_svd_vec = len(svd_pack)
    counter = 0

    x1_mass = []
    x2_mass = []
    y_mass = []
    for i, row in df_slice.iterrows():
        svd_vector = svd_pack[counter]
        counter += 1
        if counter > (num_svd_vec - 1):
            counter = 0
        row_QRS = np.load(qrs_avg_dir+str(row['num'])+'_AVG_QRS_500_hz.npy')


        x1_mass.append(svd_vector)
        x2_mass.append(row_QRS)
        y_mass.append(code)

    return x1_mass,x2_mass,y_mass


def generate_train_data(train_df_slice_for_same,df_for_diff1,df_for_diff2,df_for_diff3 ,batch_for_Svd_num = 150, num_svd_vectors = 100):

    svd_vectors_mass = []

    x1_mas = []
    x2_mass = []

    y_mass = [] # 1 - same 0 - diff

    print('generate_svd')
    for num in range(num_svd_vectors):
        print(f'{num}/{num_svd_vectors}')
        ids_for_svd = train_df_slice_for_same.sample(n=batch_for_Svd_num).reset_index(drop=True)
        svd_vector = generate_svd_vector(ids_for_svd)
        svd_vectors_mass.append(svd_vector)


    x1,x2,y = generate_paring(train_df_slice_for_same,svd_vectors_mass, 1)

    x1_mas = x1
    x2_mass = x2
    y_mass = y

    x1, x2, y = generate_paring(df_for_diff1, svd_vectors_mass, 0)
    x1_mas = np.concatenate([x1_mas,x1])
    x2_mass = np.concatenate([x2_mass,x2])
    y_mass = np.concatenate([y_mass,y])

    x1, x2, y = generate_paring(df_for_diff2, svd_vectors_mass, 0)
    x1_mas = np.concatenate([x1_mas, x1])
    x2_mass = np.concatenate([x2_mass, x2])
    y_mass = np.concatenate([y_mass, y])

    x1, x2, y = generate_paring(df_for_diff3, svd_vectors_mass, 0)
    x1_mas = np.concatenate([x1_mas, x1])
    x2_mass = np.concatenate([x2_mass, x2])
    y_mass = np.concatenate([y_mass, y])

    return x1_mas,x2_mass,y_mass



