import matplotlib.pyplot as plt

from Dataformats_tools.EDF import load_edf
import os
import numpy as np
import neurokit2 as nk
from tools import generate_LP_signal,filter_LP,mount_LP_to_ecg,add_noise,filter_ecg,filter_ecg2,calc_avg_card,generate_log_scale_system,check_LP_in_avg_sig
from CWT_mod1 import cwt,icwt, morlet_wavelet,scale_to_frequency
files_path = 'D:/Bases/only_sinus/biger100/'

edf_files = [file for file in os.listdir(files_path) if file.endswith('.' + 'edf')]

normal_ecg_mass = []
LVP_ecg_mass = []

fcounter = 0
for file in edf_files:
    fcounter +=1
    print(f'{fcounter}/{len(edf_files)}')
    try:
        signals, signal_headers, header = load_edf(files_path+file)
        fs = signal_headers[0]['sample_rate']
        signal = signals[0]

        Lp_signal = generate_LP_signal(fs) * 0.35  # 0.08

        Lp_signal = filter_LP(Lp_signal, fs)
        Lp_signal = Lp_signal[0:50]
        lp_startpos = 0.035

        filtered_sig = filter_ecg(signal, fs)
        stat, sig_peaks = nk.ecg_peaks(filtered_sig, sampling_rate=fs, method='kalidas2017')  # kalidas2017
        detected_peaks = sig_peaks['ECG_R_Peaks']

        signal = filter_ecg2(signal,fs) / max(abs(filter_ecg2(signal,fs)[200:600]))
        ecg_with_lp = mount_LP_to_ecg(signal, fs, detected_peaks, Lp_signal, LP_start_from_qrs_sec=lp_startpos)

        ecg_with_lp = ecg_with_lp
        signal = signal

        avg_card_etalon = calc_avg_card(ecg_with_lp, fs, detected_peaks)
        avg_card_etalon_clean = calc_avg_card(signal, fs, detected_peaks)

        w0 = 6

        scales = generate_log_scale_system(40, 200, 40, fs, w0)
        t = np.array(list(range(len(avg_card_etalon)))) / fs

        cwt_coefficients_lvp = cwt(avg_card_etalon, scales, morlet_wavelet, dt=1, fs=fs, w0=w0, plot_wavelets_spectrum=False, Amp_correction_target_amp=0.06)
        cwt_coefficients_clean = cwt(avg_card_etalon_clean, scales, morlet_wavelet, dt=1, fs=fs, w0=w0, plot_wavelets_spectrum=False,
                                   Amp_correction_target_amp=0.06)

        normal_ecg_mass.append(np.real(cwt_coefficients_clean))
        LVP_ecg_mass.append(np.real(cwt_coefficients_lvp))

        # cwt_coefficients = cwt_coefficients_lvp
        #
        # sig = avg_card_etalon
        # reconstructed_signal = icwt(cwt_coefficients, scales, morlet_wavelet, dt=1, ds=1)
        #
        # corr_coef_compared_etalon = check_LP_in_avg_sig(reconstructed_signal, fs, Lp_signal, lp_startpos)
        #
        # fig, axs = plt.subplots(3, 1, figsize=(7, 5), gridspec_kw={'height_ratios': [2, 1, 1]})
        #
        # axs[1].plot(t, sig, label='Original Signal')
        # axs[1].legend()
        # axs[1].set_ylabel('Amplitude [mV]')
        #
        #
        # freqs = scale_to_frequency(scales, w0, fs)
        # X, Y = np.meshgrid(t, freqs)
        # im = axs[0].pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')  # Используем цветовую карту 'viridis'
        # plt.colorbar(im, ax=axs[0])
        # plt.colorbar(im, ax=axs[1])
        # axs[0].set_ylabel('Frequency [Hz]')
        # axs[0].set_title('Wavelet transform')
        #
        # axs[2].plot(t, reconstructed_signal, label='Reconstructed Signal', color='orange')
        # axs[2].legend()
        # plt.colorbar(im, ax=axs[2])
        # axs[2].set_xlabel('time [s]')
        # axs[2].set_ylabel('Amplitude')
        #
        # plt.tight_layout()
        # plt.show()



    except:
        print('exeption')

print(np.shape(normal_ecg_mass))
np.save('D:/Bases/only_sinus/train/CWTmodif/avg_card_norm.npy', np.array(normal_ecg_mass))
np.save('D:/Bases/only_sinus/train/CWTmodif/avg_card_LVP.npy', np.array(LVP_ecg_mass))




