import matplotlib.pyplot as plt
import numpy as np

from FILES_processing_lib import scandir
import random
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb,write_annotated_wfdb
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign

def generate_ecg_noise(t,fs,amp, lowcut_hz, highcut_hz, n_samp = None):
    if n_samp:
        random_array = np.random.normal(0, amp/2, n_samp)
    else:
        random_array = np.random.normal(0, amp / 2, t * fs)
    filtered_signal = AnalogFilterDesign(random_array, fs).bp(order=3, cutoff=[lowcut_hz, highcut_hz]).zerophaze().butter().filtration()
    return filtered_signal

def generate_sig_with_nsr(fs,init_sig,added_snr_db, high_Cut_from, high_cut_to, custom_noise = False):
    init_sig_filtered = AnalogFilterDesign(init_sig, fs).bp(order=3, cutoff=[1, 60]).zerophaze().butter().filtration()
    power_signal = np.sqrt(np.median(init_sig_filtered ** 2))# better change to mean
    desired_nsr_dB = added_snr_db
    power_noise = power_signal / (10 ** (desired_nsr_dB / 10))
    highcut = random.uniform(high_Cut_from, high_cut_to)
    if custom_noise:
        noise = generate_ecg_noise(None,fs,np.sqrt(power_noise)*15,0.5,highcut,n_samp=len(init_sig))
    else:
        noise = np.random.normal(scale=np.sqrt(power_noise), size=len(init_sig))
    return np.array(init_sig)+noise

def noise_sigs(input_sigs,fs,custom_noise = False):
    noised_sigs = []
    nsr_from = 20
    nsr_to = 34
    hc_from = 10
    hc_to = 80
    nsr = random.randint(nsr_from, nsr_to)
    for signal in input_sigs:
        noised_sig = generate_sig_with_nsr(fs,signal,nsr,hc_from,hc_to,custom_noise=custom_noise)
        noised_sigs.append(noised_sig)
    return np.array(noised_sigs)

storing_path = 'E:/Bases/0Prepared_bases/Noised ECG/'

bases_dirs = ['E:/Bases/0Prepared_bases/2021_challange_prepared/',
            'E:/Bases/0Prepared_bases/but_pdb_prepared/',
            'E:/Bases/0Prepared_bases/fantasia_prep/',
            'E:/Bases/0Prepared_bases/long-term-af-database_prep/',
            'E:/Bases/0Prepared_bases/mit-bih-arrhythmia-database_prepared/',
            'E:/Bases/0Prepared_bases/mit-bih-long-term-ecg-dat_prep/',
            'E:/Bases/0Prepared_bases/mit-bih-noise-stress-test-database/',
            'E:/Bases/0Prepared_bases/st-petersburg-incart-12-lead-arrhythmia-database_prep/',
            'E:/Bases/0Prepared_bases/t-wave-alternans-challenge-database_prepared/']

noised_part = 0.3
for base_dir in bases_dirs:
    base_files = scandir(base_dir,ext='.hea')
    num_to_select = int(np.ceil(len(base_files)*noised_part))
    random_generator = random.Random(15)
    files_chosen = random_generator.sample(base_files,num_to_select,)
    file_counter = 0
    for file in files_chosen:
        try:
            file_counter = file_counter+1
            print(f"{base_dir.split('/')[-2]}_{file_counter}/{len(files_chosen)}")
            signals, re_format, fs, units, adc_gain, baseline, \
            coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(base_dir+file)
            if file_counter%2 == 0:
                custom_ns = True
            else:
                custom_ns = False
            noised_sigs_res = noise_sigs(signals, fs,custom_noise=custom_ns)
            base_name = base_dir.split('/')[-2]
            file_name = file.replace('.hea','')
            write_annotated_wfdb(base_name+'_'+file_name+'_nosd',
                                 noised_sigs_res.T,fs,None,
                                 units,None,None,coments,
                                 lead_names,annotations_pos,qrs_annotations,
                                 ecg_events,p=storing_path)
        except:
            print('exeption')
            pass
