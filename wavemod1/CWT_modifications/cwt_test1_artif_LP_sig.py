from Artifitial_signal_creation import simulate_ecg_with_VLP_ALP
from Artifitial_signal_creation import sigTotest
import matplotlib.pyplot as plt
# from CWT_basis import cwt,icwt,morlet_wavelet,scale_to_frequency
from CWT_mod1 import cwt,icwt,morlet_wavelet,scale_to_frequency
import numpy as np
from AnalogFilters import AnalogFilterDesign

def generate_log_scale_system(freq_from, freq_to, N_cofs,fs, w0):
    scale_from = scale_to_frequency([freq_to],w0=w0,fs=fs)[0]
    scale_to = scale_to_frequency([freq_from],w0=w0,fs=fs)[0]

    print(f"""
    log_scale system:
    scale from: {scale_from}
    scale to: {scale_to}
    """)

    N_vect = np.array(list(range(N_cofs)))

    r = (scale_to/scale_from)**(1/(N_cofs-1))
    s_mass = []
    for k in N_vect:
        s_val = scale_from*r**k
        s_mass.append(s_val)

    return s_mass


if __name__ == '__main__':
    w0 = 6
    fs = 1000
    sig = simulate_ecg_with_VLP_ALP(duration = 4, #sec
                                  fs = fs, #hz
                                  noise_level =35, #db
                                  hr = 80,#bpm
                                  Std = 0, #bpm
                                  unregular_comp = False,
                                    random_state = 11,
                                  lap_amp = 25,
                                  lvp_amp = 25)



    # sig = sig[120:600]
    # sig = sig[240:1200]
    # sig = sig[280:700]
    sig = sig[560:1200]







    # sig = sigTotest(fs=fs)
    # print(len(sig))
    # sig = sig[200:480]
    t_vector = np.array(list(range(len(sig))))/fs


    # scales = np.linspace(3, 300, 300)

    # scales = np.linspace(3, 15, 40)
    scales = generate_log_scale_system(40,200,40, fs, w0)
    scales = np.array(scales)

    # def overlap_50_scales(from_scale, num_scales):
    #     result_scales = []
    #     n_mass = np.array(list(range(num_scales)))
    #     for n in n_mass:
    #         result_scales.append(from_scale * ((2 ** 0.5) ** n))
    #     return np.array(result_scales)
    #
    #
    # scales = overlap_50_scales(3, 12)


    cwt_coefficients = cwt(sig, scales, morlet_wavelet, dt=1, fs=fs,w0=w0, plot_wavelets_spectrum=False, Amp_correction_target_amp=0.04)
    reconstructed_signal = icwt(cwt_coefficients, scales, morlet_wavelet, dt=1, ds=1)

    # plt.figure()
    # X, Y = np.meshgrid(t_vector, scales)
    # plt.pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Time')
    # plt.ylabel('Scale')
    # plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
    # plt.show(block=False)
    #
    # freqs = scale_to_frequency(scales, w0, fs)
    # plt.figure()
    # X, Y = np.meshgrid(t_vector, freqs)
    # plt.pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')  # Используем цветовую карту 'viridis'
    # # plt.ylim([0, 50])
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Time')
    # plt.ylabel('frequency')
    # plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
    # plt.show(block=False)
    #
    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(t_vector, sig, label='Original Signal')
    # plt.legend()
    # plt.title('Original Signal')
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(t_vector, reconstructed_signal, label='Reconstructed Signal', color='orange')
    # plt.legend()
    # plt.title('Reconstructed Signal')
    #
    # plt.tight_layout()
    # plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(7, 5), gridspec_kw={'height_ratios': [2, 1, 1]})

    axs[1].plot(t_vector, sig, label='Original Signal')
    axs[1].legend()
    axs[1].set_ylabel('Amplitude [mV]')

    # start_time = 0.093
    # end_time = 0.1267
    # start_time2 = 0.2484
    # end_time2 = 0.2807
    # red_part = copy.deepcopy(signal)
    # indices_to_keep = np.r_[np.round(start_time * fs).astype(int):np.round(end_time * fs).astype(int),
    #                   np.round(start_time2 * fs).astype(int):np.round(end_time2 * fs).astype(
    #                       int)]  # объединяем диапазоны, Python использует включительные диапазоны для срезов
    # red_part[~np.in1d(np.arange(red_part.size), indices_to_keep)] = np.nan  # заменяем все, что вне диапазонов
    #
    # axs[1].plot(t, red_part, c='r')

    freqs = scale_to_frequency(scales, w0, fs)
    X, Y = np.meshgrid(t_vector, freqs)
    im = axs[0].pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')  # Используем цветовую карту 'viridis'
    plt.colorbar(im, ax=axs[0])
    plt.colorbar(im, ax=axs[1])
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_title('Wavelet transform')

    axs[2].plot(t_vector, reconstructed_signal, label='Reconstructed Signal', color='orange')
    axs[2].legend()
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel('time [s]')
    axs[2].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

