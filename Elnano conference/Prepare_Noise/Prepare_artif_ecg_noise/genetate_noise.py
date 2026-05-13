import numpy as np
import matplotlib.pyplot as plt
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import neurokit2 as nk
import random
from Dataformats_tools.WFDB.WFDB_RW import write_annotated_wfdb

def generate_ecg_noise(t,fs,amp, lowcut_hz, highcut_hz, n_samp = None):
    if n_samp:
        random_array = np.random.normal(0, amp/2, n_samp)
    else:
        random_array = np.random.normal(0, amp / 2, t * fs)
    filtered_signal = AnalogFilterDesign(random_array, fs).bp(order=3, cutoff=[lowcut_hz, highcut_hz]).zerophaze().butter().filtration()
    return filtered_signal

def generate_baseline(t,fs,amp):
    return generate_ecg_noise(t,fs,amp, 0.1, 1)

def generate_noise_mass(sig_len, fs, amp_from_uV, amp_to_uV,low_Cut_from,low_cut_to,
                        high_Cut_from,high_cut_to, num_of_sigs):
    sigs_mass = []
    for i in range(num_of_sigs):
        amp_uV = random.uniform(amp_from_uV, amp_to_uV)
        lowcut = random.uniform(low_Cut_from,low_cut_to)
        highcut = random.uniform(high_Cut_from,high_cut_to)
        generated_sig = generate_ecg_noise(sig_len, fs, amp_uV, lowcut, highcut) + generate_baseline(sig_len, fs,
                                                                                                     amp_uV)
        sigs_mass.append(generated_sig)
    return sigs_mass

def generate_mionoise_mass(duration,amp_from,amp_to,fs,bn_from,bn_to, bd_from, bd_to, mionoise_num):
    emg_mass = []
    for i in range(mionoise_num):
        bn = random.randint(bn_from, bn_to)
        bd = random.uniform(bd_from, bd_to)
        amp = random.uniform(amp_from,amp_to)
        emg = nk.emg_simulate(duration=duration, burst_number=bn, burst_duration=bd,sampling_rate=fs)
        emg = np.array(emg)*amp
        emg_mass.append(emg)
    return emg_mass

def generate_eda_noise_mass(duration,amp_from,amp_to,fs,bn_from,bn_to, n_from, n_to, mionoise_num):
    eda_mass = []
    for i in range(mionoise_num):
        bn = random.randint(bn_from, bn_to)
        n = random.uniform(n_from,n_to)
        d = random.uniform(n_from,n_to)
        amp = random.uniform(amp_from, amp_to)
        eda = nk.eda_simulate(duration=duration, scr_number=bn, drift=-d, noise=n,sampling_rate=fs)
        eda = np.array(eda).astype(float) * (amp/2)
        eda = eda[0:np.round(duration*fs)]
        eda_mass.append(eda)
    return np.array(eda_mass)

artif_sigs_storing_path = "E:/Bases/0Prepared_bases/Noise/Artif_ecg_sig/"

nm1 = generate_noise_mass(10,1000,0.01,0.6,0.5,1,2,15,300)
nm2 = generate_noise_mass(10,1000,0.01,0.6,0.5,1,10,50,200)
nm3 = generate_noise_mass(10,1000,0.01,0.1,0.5,1,40,125,100)
mio = generate_mionoise_mass(10,0.01,0.6,1000,2,8,0.2,1.2,50)
eda = generate_eda_noise_mass(10,0.01,0.6,1000,1,8,0.05,0.1,50)

general_mass = np.concatenate((nm1,nm2,nm3,mio,eda))

file_counter = 0
for sig in general_mass:
    file_counter = file_counter+1
    print(f"{file_counter}/{len(general_mass)}")
    write_annotated_wfdb('noise_p' + str(file_counter), np.array([sig]).T, 1000,None,["mV"],None,None,None,
                                     ['Lead1'],
                                     np.array([0]), np.array(['~']), np.array(["Ns"]), p=artif_sigs_storing_path)

