import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FILES_processing_lib import scandir


qrs_avg_dir = 'D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/QRS_averaged/'

metadata_df = pd.read_csv(qrs_avg_dir+'AVG_QRS_metadata.csv')

qrs_avg_files = scandir(qrs_avg_dir,ext='.npy')

norm_data_slice = metadata_df[metadata_df['scp_codes'] == "['NORM', 'SR']"].sample(n=2000, random_state=42).reset_index(drop=True)
LAP_data_slice = metadata_df[metadata_df['scp_codes'] == "['LAP']"]
LVP_data_slice = metadata_df[metadata_df['scp_codes'] == "['LVP']"]
LAP_LVP_data_slice = metadata_df[metadata_df['scp_codes'] == "['LAP', 'LVP']"]


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



norm_data_train = norm_data_slice.iloc[:1000]
LAP_train = LAP_data_slice.iloc[:650]
LVP_train = LVP_data_slice.iloc[:650]
LAP_LVP_train = LAP_LVP_data_slice[:200]


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




np.save('trainDAta/norm_test_QRS.npy',generate_mass(norm_data_train))
np.save('trainDAta/lap_test_QRS.npy',generate_mass(LAP_train))
np.save('trainDAta/lvp_test_QRS.npy',generate_mass(LVP_train))
np.save('trainDAta/lap_lvp_test_QRS.npy',generate_mass(LAP_LVP_train))

norm_test = norm_data_slice.iloc[1820:]
lap_test = LAP_data_slice.iloc[650:]
lvp_test = LVP_data_slice.iloc[650:]
lap_lvp_test = LAP_LVP_data_slice[200:]



np.save('trainDAta/norm_test_QRS_for_svd.npy',generate_mass(norm_test))
np.save('trainDAta/lap_test_QRS_for_svd.npy',generate_mass(lap_test))
np.save('trainDAta/lvp_test_QRS_for_svd.npy',generate_mass(lvp_test))
np.save('trainDAta/lap_lvp_test_QRS_for_svd.npy',generate_mass(lap_lvp_test))


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


X1 = []
X2 = []
Y = []

x1_mas,x2_mass,y_mass = generate_train_data(norm_data_train, LAP_train,LVP_train,LAP_LVP_train)

X1 = x1_mas
X2 = x2_mass
Y = y_mass

x1_mas,x2_mass,y_mass = generate_train_data(LAP_train, norm_data_train, LVP_train, LAP_LVP_train)

X1 = np.concatenate([X1, x1_mas])
X2 = np.concatenate([X2, x2_mass])
Y = np.concatenate([Y,y_mass])

x1_mas,x2_mass,y_mass = generate_train_data(LVP_train, norm_data_train, LAP_train, LAP_LVP_train)

X1 = np.concatenate([X1, x1_mas])
X2 = np.concatenate([X2, x2_mass])
Y = np.concatenate([Y,y_mass])

x1_mas,x2_mass,y_mass = generate_train_data(LAP_LVP_train, norm_data_train, LAP_train,LVP_train)

X1 = np.concatenate([X1, x1_mas])
X2 = np.concatenate([X2, x2_mass])
Y = np.concatenate([Y,y_mass])

np.save('trainDAta/Train_X1.npy',X1)
np.save('trainDAta/Train_X2.npy',X2)
np.save('trainDAta/Train_Y.npy',Y)

