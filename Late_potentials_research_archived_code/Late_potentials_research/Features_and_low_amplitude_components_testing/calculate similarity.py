import numpy as np
import neurokit2 as nk
import scipy.stats

from Artifitial_signal_creation import simulate_ecg_with_VLP_ALP
import matplotlib.pyplot as plt
from BIOMED_SIGNALS.ECG import class_record
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
from scipy.spatial import distance
from scipy import signal


def lap_lvp_signal_part(input_sig, r_peak_location, peaks_mass):
    lap_startpos = int(r_peak_location + peaks_mass['ECG_P_Peaks'])
    lap_endpos = int(r_peak_location + peaks_mass['ECG_Q_Peaks'])
    lvp_startpos = int(r_peak_location + peaks_mass['ECG_S_Peaks'])
    lvp_endpos = int(r_peak_location + peaks_mass['ECG_T_Onsets'])
    lap_sigpart = np.array(input_sig)[lap_startpos:lap_endpos]
    lvp_sigpart = np.array(input_sig)[lvp_startpos:lvp_endpos]
    return lap_sigpart,lvp_sigpart

def lp_filt(input_sig, fs):
    signal_filtered = AnalogFilterDesign(input_sig, fs).hp(order=5, cutoff=40).zerophaze().butter() \
        .filtration()
    # signal_filtered = AnalogFilterDesign(signal_filtered, fs).lp(order=5, cutoff=240).zerophaze().butter() \
    #     .filtration()
    return signal_filtered

def run_similarity(lap_amp, lvp_amp, custom = True):
    fs = 1000
    duration = 100#80
    test_signal = simulate_ecg_with_VLP_ALP(fs=fs,
                                            duration=duration,
                                            noise_level=30, #25db
                                            hr=80,
                                            Std=2,#bpm
                                            unregular_comp= True,
                                            lap_amp=lap_amp,
                                            lvp_amp = lvp_amp)

    clean_signal = simulate_ecg_with_VLP_ALP(fs=fs,
                                            duration=duration,
                                            noise_level=140, #db
                                            hr=80,
                                            Std=0,#bpm
                                            unregular_comp=False)



    test_record = class_record.EcgRecord([fs],np.array([test_signal]),['artificial signal'], ['mV'],None)


    remove_strange_cardiocycles = False


    if custom:
        avg_method = 'custom'
    else:
        avg_method = 'avg'
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
        sig1 = np.array(sig1)
        sig2 = np.array(sig2)

        s1_max = sig1.max()
        s2_max = sig2.max()
        diff = s2_max/s1_max
        sig2 = sig2/diff



        cosine_similarity = np.dot(sig1, sig2)/(np.linalg.norm(sig1)*np.linalg.norm(sig2))
        pearson_correlation = scipy.stats.pearsonr(sig1,sig2).statistic


        # print(cosine_similarity)
        # print(pearson_correlation)
        # plt.figure()
        # plt.plot(sig1)
        # plt.plot(sig2)
        # plt.show()

        # input(cosine_similarity)

        return {'cos':cosine_similarity,'cor':pearson_correlation}

    print('lvp similarity')

    lvp_similarity = similarity_coefs(lp_filt(lvp_clean_sig, fs), lp_filt(lvp_part_saecg, fs))
    print('lap similarity')
    lap_filtered = np.array(lp_filt(lap_clean_sig, fs))[0:-1]
    lap_filtered_2 = np.array(lp_filt(lap_part_psaecg, fs))[1:]
    lap_similarity = similarity_coefs(lap_filtered, lap_filtered_2)

    return lvp_similarity, lap_similarity

amp_list = [1,5,10,15,20,25,30,35,40]
lvp_cos = []
lvp_cor = []
lap_cos = []
lap_cor = []

lvp_cos_avg = []
lvp_cor_avg = []
lap_cos_avg = []
lap_cor_avg = []
for amp in amp_list:
    lvp_similarity, lap_similarity = run_similarity(amp, amp, custom=True)
    lvp_cos.append(lvp_similarity['cos'])
    lvp_cor.append(lvp_similarity['cor'])
    lap_cos.append(lap_similarity['cos'])
    lap_cor.append(lap_similarity['cor'])
    lvp_similarity, lap_similarity = run_similarity(amp, amp, custom=False)
    lvp_cos_avg.append(lvp_similarity['cos'])
    lvp_cor_avg.append(lvp_similarity['cor'])
    lap_cos_avg.append(lap_similarity['cos'])
    lap_cor_avg.append(lap_similarity['cor'])

plt.figure()
ax1 = plt.subplot(2,2,1)
ax1.plot(amp_list, lvp_cos_avg,label= 'ППШ Косинусна подібність')
ax1.plot(amp_list, lap_cos_avg,label= 'ППП Косинусна подібність')
ax1.plot(amp_list, lvp_cor_avg,label= 'ППШ Кореляція Пірсона')
ax1.plot(amp_list, lap_cor_avg,label= 'ППП Кореляція Пірсона')
ax1.set_title('Усереднення сигналу')
plt.xlabel('Амплітуда пізніх потенціалів [мкВ]')
plt.ylabel('Значення показників якості')
plt.legend()

ax1 = plt.subplot(2,2,2)
ax1.plot(amp_list, lvp_cos,label= 'ППШ Косинусна подібність')
ax1.plot(amp_list, lap_cos,label= 'ППП Косинусна подібність')
ax1.plot(amp_list, lvp_cor,label= 'ППШ Кореляція Пірсона')
ax1.plot(amp_list, lap_cor,label= 'ППП Кореляція Пірсона')
ax1.set_title('Запропонований метод на основі SVD')
plt.xlabel('Амплітуда пізніх потенціалів [мкВ]')
plt.ylabel('Значення показників якості')
plt.legend()

ax1 = plt.subplot(2,2,3)
ax1.set_title('Запропонований метод на основі PCA')
plt.xlabel('Амплітуда пізніх потенціалів [мкВ]')
plt.ylabel('Значення показників якості')

ax1 = plt.subplot(2,2,4)
ax1.set_title('Запропонований метод на основі FA')
plt.xlabel('Амплітуда пізніх потенціалів [мкВ]')
plt.ylabel('Значення показників якості')
plt.show()


