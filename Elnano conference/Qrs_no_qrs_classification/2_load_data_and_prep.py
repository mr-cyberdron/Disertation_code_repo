from FILES_processing_lib import scandir
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import matplotlib.pyplot as plt


def shuffle_and_reduce(X_array,y_array, num_to_select):
    print('shuffle_and_reduce')
    permuted_indices = np.random.permutation(np.shape(X_array)[0])
    indices_sample = np.random.choice(permuted_indices, size=num_to_select, replace=False)

    shuffled_X = X_array[indices_sample]
    shuffled_y = y_array[indices_sample]
    return shuffled_X,shuffled_y

def creating_complex_XY(X_qrs,y_qrs,X_noise,y_noise):
    print('creating_complex_XY')
    X = np.concatenate((X_qrs,X_noise),axis=0)
    y = np.concatenate((y_qrs,y_noise),axis=0)

    assert len(X) == len(y)
    permuted_indices = np.random.permutation(np.shape(X)[0])
    X_shufeled = X[permuted_indices]
    y_shufeled = y[permuted_indices]
    return X_shufeled,y_shufeled

def dubble_mass(X_array,y_array):
    print('doubling mass')
    X_store = np.concatenate((X_array,X_array),axis=0)
    y_store = np.concatenate((y_array,y_array),axis=0)
    return X_store, y_store

def tripple_mass(X_array,y_array):
    print('trippling mass')
    X_store = np.concatenate((X_array, X_array,X_array), axis=0)
    y_store = np.concatenate((y_array, y_array,y_array), axis=0)
    return X_store, y_store

def remove_nan(data_mass_xx, data_mass_yy):
    print('remove_nan')
    nan_counterr = 0
    items_to_dropp = []
    for x_itemm, y_itemm in zip(data_mass_xx, data_mass_yy):
        if np.isnan((x_itemm)[0]):
            items_to_dropp.append(nan_counterr)
        nan_counterr = nan_counterr + 1

    if items_to_dropp:
        cleaned_xx = np.delete(data_mass_xx,items_to_dropp,axis=0)
        cleaned_yy = np.delete(data_mass_yy,items_to_dropp,axis=0)
    else:
        cleaned_xx = data_mass_xx
        cleaned_yy = data_mass_yy

    return cleaned_xx, cleaned_yy

def load_xy(data_dir, X_files_mass,y_files_mass):
    X = np.array([])
    y = np.array([])
    counter = 0
    for x_fname, y_fname in zip(X_files_mass, y_files_mass):
        counter+=1
        print(f'loading {counter}/{len(X_files_mass)}')
        total_x = np.load(data_dir+x_fname)['arr_0']
        total_y = np.load(data_dir+y_fname)['arr_0']

        total_x,total_y = remove_nan(total_x,total_y)

        if not X.any():
            X = total_x
        else:
            X = np.concatenate((X,total_x),axis=0)

        if not np.any((y == 0)|(y == 1)):
            y = total_y
        else:
            y = np.concatenate((y,total_y),axis=0)

    print(f'X:{np.shape(X)}')
    print(f'y:{np.shape(y)}')
    return X,y


qrs_noise_data_source = 'E:/Bases/1QRS_noise_base/shortage/'
prepared_store_path = 'E:/Bases/1QRS_noise_base/prepared/'


double_noise = True
''''Индекс на который умножаеться размер выборки шума чтобы получить выборку QRS тогда
size(QRS):size(Noise) = 5:1:'''
QRS_size_by_noise_scale_index = 5



qrs_x_files = ['2021_challange_prepared_QRS_X.npz', 'but_pdb_prepared_QRS_X.npz', 'fantasia_prep_QRS_X.npz', 'long-term-af-database_prep_QRS_X.npz', 'mit-bih-arrhythmia-database_prepared_QRS_X.npz', 'mit-bih-long-term-ecg-dat_prep_QRS_X.npz', 'mit-bih-noise-stress-test-database_QRS_X.npz', 'Noised ECG_QRS_X.npz', 'st-petersburg-incart-12-lead-arrhythmia-database_prep_QRS_X.npz', 't-wave-alternans-challenge-database_prepared_QRS_X.npz']
qrs_y_files = ['2021_challange_prepared_QRS_y.npz', 'but_pdb_prepared_QRS_y.npz', 'fantasia_prep_QRS_y.npz', 'long-term-af-database_prep_QRS_y.npz', 'mit-bih-arrhythmia-database_prepared_QRS_y.npz', 'mit-bih-long-term-ecg-dat_prep_QRS_y.npz', 'mit-bih-noise-stress-test-database_QRS_y.npz', 'Noised ECG_QRS_y.npz', 'st-petersburg-incart-12-lead-arrhythmia-database_prep_QRS_y.npz', 't-wave-alternans-challenge-database_prepared_QRS_y.npz']
noise_x_files = ['Artif_ecg_sig_Noise_X.npz', 'cu-ventricular-tachyarrhythmia-database_Noise_X.npz', 'mit-bih-arrhythmia-database_Noise_X.npz', 'mit-bih-noise-stress-test-database_Noise_X.npz', 'noise_rec_Noise_X.npz']
noise_y_files = ['Artif_ecg_sig_Noise_y.npz', 'cu-ventricular-tachyarrhythmia-database_Noise_y.npz', 'mit-bih-arrhythmia-database_Noise_y.npz', 'mit-bih-noise-stress-test-database_Noise_y.npz', 'noise_rec_Noise_y.npz']

# qrs_x_files = ['2021_challange_prepared_QRS_X.npz']
# qrs_y_files = ['2021_challange_prepared_QRS_y.npz']
# noise_x_files = ['Artif_ecg_sig_Noise_X.npz']
# noise_y_files = ['Artif_ecg_sig_Noise_y.npz']



QRS_x, QRS_y = load_xy(qrs_noise_data_source, qrs_x_files,qrs_y_files)
Noise_x, Noise_y = load_xy(qrs_noise_data_source,noise_x_files,noise_y_files)

Noise_x, Noise_y = tripple_mass(Noise_x,Noise_y)



print(f'QRS_len:{len(QRS_x)}')
print(f'Noise_len:{len(Noise_x)}')
print(f'Noise_lenx5:{len(Noise_x)*5}')

num_of_qrs_to_select = np.shape(Noise_x)[0]*QRS_size_by_noise_scale_index

print(np.shape(QRS_x))
print(num_of_qrs_to_select)

QRS_x_shuffeled, QRS_y_shuffeled = shuffle_and_reduce(QRS_x,QRS_y,num_of_qrs_to_select)
print('shuffle noise')
Noise_x_shuffeled, Noise_y_shuffeled = shuffle_and_reduce(Noise_x,Noise_y,np.shape(Noise_x)[0])

X,y = creating_complex_XY(QRS_x_shuffeled,QRS_y_shuffeled,Noise_x_shuffeled,Noise_y_shuffeled)

input('s')
print('Saving data')
np.savez(prepared_store_path+'QRS_Noise_prep.npz',x = X,y = y)




# qrs_x_files = scandir(qrs_noise_data_source,ext='QRS_X.npz')
# qrs_y_files = scandir(qrs_noise_data_source,ext='QRS_y.npz')
# noise_x_files = scandir(qrs_noise_data_source,ext='Noise_X.npz')
# noise_y_files = scandir(qrs_noise_data_source,ext='Noise_y.npz')
# print(qrs_x_files)
# print(qrs_y_files)
# print(noise_x_files)
# print(noise_y_files)

