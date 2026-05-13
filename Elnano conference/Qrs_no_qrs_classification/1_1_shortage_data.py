import  numpy as np


def shuffle_and_reduce(X_array,y_array, num_to_select):
    print('shuffle_and_reduce')
    permuted_indices = np.random.permutation(np.shape(X_array)[0])
    indices_sample = np.random.choice(permuted_indices, size=num_to_select, replace=False)

    shuffled_X = X_array[indices_sample]
    shuffled_y = y_array[indices_sample]
    return shuffled_X,shuffled_y

def shortage_data(data_dir, X_files_mass,y_files_mass, part_remaining, storage_path):
    counter = 0
    for x_fname, y_fname in zip(X_files_mass, y_files_mass):
        counter+=1
        print(f'shortaging{counter}/{len(X_files_mass)}')
        print('loading')
        total_x = np.load(data_dir+x_fname)['arr_0']
        total_y = np.load(data_dir+y_fname)['arr_0']
        assert len(total_x) == len(total_y)
        desired_len = int(np.round(len(total_x)*part_remaining))
        x_shortage, y_shortage = shuffle_and_reduce(total_x, total_y, desired_len)
        print('saving')
        np.savez(storage_path+x_fname, x_shortage)
        np.savez(storage_path +y_fname, y_shortage)








qrs_noise_data_source = 'E:/Bases/1QRS_noise_base/'
shortage_store_path = 'E:/Bases/1QRS_noise_base/shortage/'

qrs_x_files = ['2021_challange_prepared_QRS_X.npz', 'but_pdb_prepared_QRS_X.npz', 'fantasia_prep_QRS_X.npz', 'long-term-af-database_prep_QRS_X.npz', 'mit-bih-arrhythmia-database_prepared_QRS_X.npz', 'mit-bih-long-term-ecg-dat_prep_QRS_X.npz', 'mit-bih-noise-stress-test-database_QRS_X.npz', 'Noised ECG_QRS_X.npz', 'st-petersburg-incart-12-lead-arrhythmia-database_prep_QRS_X.npz', 't-wave-alternans-challenge-database_prepared_QRS_X.npz']
qrs_y_files = ['2021_challange_prepared_QRS_y.npz', 'but_pdb_prepared_QRS_y.npz', 'fantasia_prep_QRS_y.npz', 'long-term-af-database_prep_QRS_y.npz', 'mit-bih-arrhythmia-database_prepared_QRS_y.npz', 'mit-bih-long-term-ecg-dat_prep_QRS_y.npz', 'mit-bih-noise-stress-test-database_QRS_y.npz', 'Noised ECG_QRS_y.npz', 'st-petersburg-incart-12-lead-arrhythmia-database_prep_QRS_y.npz', 't-wave-alternans-challenge-database_prepared_QRS_y.npz']
noise_x_files = ['Artif_ecg_sig_Noise_X.npz', 'cu-ventricular-tachyarrhythmia-database_Noise_X.npz', 'mit-bih-arrhythmia-database_Noise_X.npz', 'mit-bih-noise-stress-test-database_Noise_X.npz', 'noise_rec_Noise_X.npz']
noise_y_files = ['Artif_ecg_sig_Noise_y.npz', 'cu-ventricular-tachyarrhythmia-database_Noise_y.npz', 'mit-bih-arrhythmia-database_Noise_y.npz', 'mit-bih-noise-stress-test-database_Noise_y.npz', 'noise_rec_Noise_y.npz']

part_remaining = 0.1

shortage_data(qrs_noise_data_source,qrs_x_files,qrs_y_files,part_remaining,shortage_store_path)
