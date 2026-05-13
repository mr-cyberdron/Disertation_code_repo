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



norm_data_train = norm_data_slice.iloc[:1900]
LAP_train = LAP_data_slice.iloc[:700]
LVP_train = LVP_data_slice.iloc[:700]
LAP_LVP_train = LAP_LVP_data_slice[:200]



np.save('trainDAta/norm_test_QRS.npy',generate_mass(norm_data_train))
np.save('trainDAta/lap_test_QRS.npy',generate_mass(LAP_train))
np.save('trainDAta/lvp_test_QRS.npy',generate_mass(LVP_train))
np.save('trainDAta/lap_lvp_test_QRS.npy',generate_mass(LAP_LVP_train))

norm_test = norm_data_slice.iloc[1900:]
lap_test = LAP_data_slice.iloc[700:]
lvp_test = LVP_data_slice.iloc[700:]
lap_lvp_test = LAP_LVP_data_slice[200:]

np.save('trainDAta/norm_test_QRS_for_svd.npy',generate_mass(norm_test))
np.save('trainDAta/lap_test_QRS_for_svd.npy',generate_mass(lap_test))
np.save('trainDAta/lvp_test_QRS_for_svd.npy',generate_mass(lvp_test))
np.save('trainDAta/lap_lvp_test_QRS_for_svd.npy',generate_mass(lap_lvp_test))


X_train1 = np.load('trainDAta/norm_test_QRS.npy')
X_train2 = np.load('trainDAta/lap_test_QRS.npy')
X_train3 = np.load('trainDAta/lvp_test_QRS.npy')
X_train4 = np.load('trainDAta/lap_lvp_test_QRS.npy')
X_train = np.concatenate((X_train1,X_train2,X_train3,X_train4))
print(np.shape(X_train))
Y_train = []
for i1 in X_train1:
    Y_train.append(0)
for i2 in X_train2:
    Y_train.append(1)
for i3 in X_train3:
    Y_train.append(1)
for i4 in X_train4:
    Y_train.append(1)

print(np.shape(Y_train))


X_test1 = np.load('trainDAta/norm_test_QRS_for_svd.npy')
X_test2 = np.load('trainDAta/lap_test_QRS_for_svd.npy')
X_test3 = np.load('trainDAta/lvp_test_QRS_for_svd.npy')
X_test4 = np.load('trainDAta/lap_lvp_test_QRS_for_svd.npy')
X_test = np.concatenate((X_test1,X_test2,X_test3,X_test4))
print(np.shape(X_test))
Y_test = []

for j1 in X_test1:
    Y_test.append(0)
for j2 in X_test2:
    Y_test.append(1)
for j3 in X_test3:
    Y_test.append(1)
for j4 in X_test4:
    Y_test.append(1)

print(np.shape(Y_test))


np.save('trainDAta/X_train2.npy',X_train)
np.save('trainDAta/Y_train2.npy',Y_train)
np.save('trainDAta/X_test2.npy',X_test)
np.save('trainDAta/Y_test2.npy',Y_test)


