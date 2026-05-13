import numpy as np
import neurokit2 as nk
from Artifitial_signal_creation import simulate_ecg_with_VLP_ALP
import matplotlib.pyplot as plt
from BIOMED_SIGNALS.ECG import class_record
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign


def lap_lvp_signal_part(input_sig, r_peak_location, peaks_mass):
    lap_startpos = int(r_peak_location + peaks_mass['ECG_P_Peaks'])
    lap_endpos = int(r_peak_location + peaks_mass['ECG_Q_Peaks'])
    lvp_startpos = int(r_peak_location + peaks_mass['ECG_S_Peaks'])
    lvp_endpos = int(r_peak_location + peaks_mass['ECG_T_Onsets'])
    lap_sigpart = np.array(input_sig)[lap_startpos:lap_endpos]
    lvp_sigpart = np.array(input_sig)[lvp_startpos:lvp_endpos]
    return lap_sigpart,lvp_sigpart

def lp_filt(input_sig):
    signal_filtered = AnalogFilterDesign(input_sig, fs).hp(order=5, cutoff=40).zerophaze().butter() \
        .filtration()
    # signal_filtered = AnalogFilterDesign(signal_filtered, fs).lp(order=5, cutoff=240).zerophaze().butter() \
    #     .filtration()
    return signal_filtered

fs = 1000
duration = 200#80
test_signal = simulate_ecg_with_VLP_ALP(fs=fs,
                                        duration=duration,
                                        noise_level=40, #25db
                                        hr=80,
                                        Std=2,#bpm
                                        unregular_comp= True)

clean_signal = simulate_ecg_with_VLP_ALP(fs=fs,
                                        duration=duration,
                                        noise_level=140, #db
                                        hr=80,
                                        Std=0,#bpm
                                        unregular_comp=False)



test_record = class_record.EcgRecord([fs],np.array([test_signal]),['artificial signal'], ['mV'],None)


remove_strange_cardiocycles = False
#avg_method = 'avg'
avg_method = 'custom'
test_record.saecg_count(all_leads_peaks_average=True,avg_method=avg_method,
                   preprocessing='butter_filter',
                   remove_strange_cardiocycles=remove_strange_cardiocycles,
                        only_lp=False)

test_record.psaecg_count(all_leads_p_peaks_averaged=True, avg_method=avg_method,
                    preprocessing='butter_filter',
                    remove_strange_cardiocycles=remove_strange_cardiocycles,
                         only_lp=False)

saecg_result = test_record.SAECG[0]
psaecg_result = test_record.pSAECG[0]

saecg_peaks = test_record.SAECG_relatively_r_peaks_location
psaecg_peaks = test_record.pSAECG_relatively_r_peaks_location


saecg_r_peak = class_record.EcgRecord.find_r_peak_on_qrs_complex(saecg_result, fs)
psaecg_r_peak = class_record.EcgRecord.find_r_peak_on_qrs_complex(psaecg_result, fs)

signals, info = nk.ecg_peaks(clean_signal, correct_artifacts=True)
clean_signal_r_peak = info["ECG_R_Peaks"][1]


_,lvp_part_saecg = lap_lvp_signal_part(saecg_result, saecg_r_peak, saecg_peaks)
lap_part_psaecg,_ = lap_lvp_signal_part(psaecg_result, psaecg_r_peak, psaecg_peaks)


_, lvp_clean_sig = lap_lvp_signal_part(clean_signal, clean_signal_r_peak, saecg_peaks)
lap_clean_sig,_ = lap_lvp_signal_part(clean_signal, clean_signal_r_peak, psaecg_peaks)

def similarity_coefs(sig1, sig2, pref = ''):
    print(pref)


print('lvp similarity')

similarity_coefs(lp_filt(lvp_clean_sig), lp_filt(lvp_part_saecg))
print('lap similarity')
similarity_coefs(lp_filt(lap_clean_sig), lp_filt(lap_part_psaecg))

input('rr')


# plt.figure()
# ax1 = plt.subplot(2,1,1)
# signal_t = np.array(list(range(len(lvp_clean_sig))))/1000
# ax1.plot(signal_t,lp_filt(lvp_clean_sig),label= 'ППШ Еталон')
# ax1.plot(signal_t, lp_filt(lvp_part_saecg),label= 'ППШ SAECG')
# plt.xlabel('Час [Сек]')
# plt.ylabel('Амплітуда [мВ]')
# ax1.set_title('Пізні потенціали шлуночків')
# plt.legend()
# ax2 = plt.subplot(2,1,2)
# signal_t = np.array(list(range(len(lap_clean_sig))))/1000
# ax2.plot(signal_t, lp_filt(lap_clean_sig),label= 'ППП Еталон')
# ax2.plot(signal_t, lp_filt(lap_part_psaecg),label= 'ППП pSAECG')
# plt.xlabel('Час [Сек]')
# plt.ylabel('Амплітуда [мВ]')
# ax2.set_title('Пізні потенціали передсердь')
# plt.legend()
# plt.show()

input('pp')
