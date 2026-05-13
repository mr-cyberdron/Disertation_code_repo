import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import random


def late_potentials_generation(Fs, scale = 1.0):
    RPPt = np.arange(0.201, (0.201 + 0.028), (1 / Fs))
    RPP = 0.005 * np.sin(2 * np.pi * 78 * RPPt) + 0.012 * np.sin(2 * np.pi * 116 * RPPt + (np.pi / 4)) + 0.002 * np.sin(
        2 * np.pi * 102 * RPPt + (np.pi / 2))
    return RPP*57.6*scale

def find_peaks(input_sig, fs):
    _, rpeaks = nk.ecg_peaks(input_sig, sampling_rate=fs)
    signals, waves = nk.ecg_delineate(input_sig, rpeaks, sampling_rate=fs)
    return list(waves["ECG_P_Peaks"]), list(rpeaks['ECG_R_Peaks'])

def add_lap(input_sig, fs, lap_offset = 0.07, lap_scale = 1.0):
    p_peaks,r_peaks = find_peaks(input_sig,fs)
    lap_correction_line = np.zeros(np.shape(input_sig))
    lap_sample = late_potentials_generation(fs,lap_scale)
    #ADD LAP
    for p_peak in p_peaks:
        total_lap_startpos = p_peak+round(lap_offset*fs)
        lap_correction_line[total_lap_startpos:total_lap_startpos+len(lap_sample)] = lap_sample
    print('lap_amplitude_uV')
    print(np.max(lap_correction_line)*1000)
    print('------')

    # signal_t = np.array(list(range(len(lap_correction_line)))) / 1000
    # plt.subplot(2, 1, 1)
    # plt.plot(signal_t, lap_correction_line)
    # plt.xlabel('Час [Сек]')
    # plt.ylabel('Амплітуда [мВ]')

    return input_sig+lap_correction_line

def add_lvp(input_sig, fs, lvp_offset = 0.06, lvp_scale = 1.0):
    p_peaks, r_peaks = find_peaks(input_sig, fs)
    lap_correction_line = np.zeros(np.shape(input_sig))

    lap_sample = late_potentials_generation(fs, lvp_scale)

    # ADD LVP
    for p_peak in r_peaks:
        total_lap_startpos = p_peak + round(lvp_offset * fs)
        if total_lap_startpos< len(lap_correction_line)+len(lap_sample)+2:
            lap_correction_line[total_lap_startpos:total_lap_startpos + len(lap_sample)] = lap_sample
    print('lvp_amplitude_uV')
    print(np.max(lap_correction_line)*1000)
    print('------')
    # signal_t = np.array(list(range(len(lap_correction_line)))) / 1000
    # plt.subplot(2, 1, 1)
    # plt.plot(signal_t, lap_correction_line)
    # plt.xlabel('Час [Сек]')
    # plt.ylabel('Амплітуда [мВ]')
    return input_sig + lap_correction_line

def filter_breath(input_sig,fs):
    signal_filtered = AnalogFilterDesign(input_sig, fs).hp(order=5, cutoff=1).zerophaze().butter() \
        .filtration()
    return signal_filtered


def add_unregular_component(input_sig, fs, position_mass = [1000], unregular_len = 0.07, scale_cof = 0.3,
                            noise = None):
    component_zeroline  = np.zeros(len(input_sig))
    for total_position in position_mass:
        start_impulse = total_position+round(unregular_len*fs)
        end_impulse =start_impulse+round(unregular_len*fs)
        impulse_body = np.linspace(1,0, (end_impulse-start_impulse))
        if noise:
            noise_body = np.random.normal(0,1,(end_impulse-start_impulse))*noise
            impulse_body = impulse_body+noise_body
        new_impulse = []
        for i in impulse_body:
            new_impulse.append(i+(i**0.1))
        component_zeroline[start_impulse:end_impulse] = new_impulse
    print('Unregular_component_amplitude:')
    print(np.max(component_zeroline*scale_cof))
    print('-----')
    # plt.subplot(2, 1, 1)
    # signal_t = np.array(list(range(len(component_zeroline*scale_cof)))) / 1000
    # plt.plot(signal_t, component_zeroline*scale_cof)
    # plt.xlabel('Час [Сек]')
    # plt.ylabel('Амплітуда [мВ]')
    return input_sig+(component_zeroline*scale_cof)

def add_noise(input_signal, fs, snr = 130):
    ecg_power = np.sum(input_signal ** 2) / len(input_signal)
    noise_power = ecg_power / (10 ** (snr / 10))
    noise = np.random.normal(scale=np.sqrt(noise_power), size=len(input_signal))
    noisy_ecg = input_signal + noise
    # noise_power = np.mean(noise**2)
    # snr = 10 * np.log10(ecg_power / noise_power)
    # input(snr)

    # signal_t = np.array(list(range(len(noise)))) / 1000
    # ax1 = plt.subplot(2, 1, 1)
    # plt.plot(signal_t, noise)
    # plt.xlabel('Час [Сек]')
    # plt.ylabel('Амплітуда [мВ]')
    # ax1.legend(['Чистий ЕКГ', 'ППП' ,'ППШ', 'Нерегулярна складова', 'Шум'])

    return noisy_ecg

def simulate_ecg_with_VLP_ALP(duration = 20, #sec
                              fs = 1000, #hz
                              noise_level =130, #db
                              hr = 80,#bpm
                              Std = 2, #bpm
                              unregular_comp = True,
                                random_state = 11,
                              lap_amp = 10,
                              lvp_amp = 30
                              ):
    ecg = nk.ecg_simulate(duration=duration,sampling_rate=fs,
                          noise=0,method="ecgsyn", heart_rate= hr,
                          heart_rate_std=Std)

    # plt.figure()
    # signal_t = np.array(list(range(len(ecg))))/1000
    # plt.subplot(2,1,1)
    # plt.plot(signal_t, ecg)
    # plt.xlabel('Час [Сек]')
    # plt.ylabel('Амплітуда [мВ]')


    #ecg = filter_breath(ecg, fs)
    ecg = add_lap(ecg,fs, lap_scale=lap_amp*0.001)#10
    ecg = add_lvp(ecg,fs, lvp_scale=lvp_amp*0.001)#30
    if unregular_comp:
        r = np.random.RandomState(random_state)#11
        ecg = add_unregular_component(ecg, fs,
                                       position_mass=[r.randint(1, high = (duration*fs)-20) for _
                                                      in range((duration/2).__ceil__())])
    ecg = add_noise(ecg,fs, snr = noise_level)

    # signal_t = np.array(list(range(len(ecg)))) / 1000
    # ax2 = plt.subplot(2, 1, 2)
    # plt.plot(signal_t, ecg)
    # plt.xlabel('Час [Сек]')
    # plt.ylabel('Амплітуда [мВ]')
    # ax2.legend(['Результуючий сигнал'])

    return ecg


# signal = simulate_ecg_with_VLP_ALP(duration=4, noise_level=40, random_state = 26)#26
# plt.show()
# signal_t = np.array(list(range(len(signal))))/1000
# plt.figure()
# plt.plot(signal_t, signal)
# plt.xlabel('Time [sec]')
# plt.ylabel('Amplitude [mV]')
# plt.show()

