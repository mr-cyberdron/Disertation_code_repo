import numpy as np
import matplotlib.pyplot as plt
from Artifitial_signal_creation import sigTotest
from Artifitial_signal_creation import simulate_ecg_with_VLP_ALP
from Convolve_mod import convolve_same2, custom_conv_with_metric, convolve_cosine_sim_based_mod, convolve_mod
import copy



def morlet_wavelet(t, w=6.0):
    wavelet = np.exp(1j * w * t) * np.exp(-0.5 * t ** 2) * np.pi ** (-0.25)
    return wavelet

def scale_to_frequency(scales, w0, fs):
    frequencies = (w0 / (2 * np.pi * scales)) * fs
    return frequencies

def plot_wavelet_fourier_spectrum(wavelet_func, dt, scale_tit = '', freq_tit = ''):
        wavelet_fft = np.fft.fft(wavelet_func)
        wavelet_fft = np.fft.fftshift(wavelet_fft)  # Shift zero frequency component to center

        # Frequency axis
        freq = np.fft.fftfreq(len(wavelet_func), d=dt)
        freq = np.fft.fftshift(freq)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(freq, np.abs(wavelet_fft))
        plt.title(f'Fourier Spectrum of the Morlet Wavelet with scale {scale_tit} ({freq_tit}Hz)')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()


def plot_spectrum_of_wavelets_mass(wavelet_mass, dt, scales):
    fig, ax1 = plt.subplots(figsize=(10, 3))
    for wavelet_data, total_scale in zip(wavelet_mass,scales):
        wavelet_func = wavelet_data

        wavelet_fft = np.fft.fft(wavelet_func)
        wavelet_fft = np.fft.fftshift(wavelet_fft)  # Shift zero frequency component to center

        # Frequency axis
        freq = np.fft.fftfreq(len(wavelet_func), d=dt)
        freq = np.fft.fftshift(freq)

        wavelet_peak_amp = np.max(np.abs(wavelet_fft))
        # Plotting
        ax1.plot(freq, np.abs(wavelet_fft)/wavelet_peak_amp, 'b-')
        ax1.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Magnitude')
        ax1.set_xlim(3,max(freq))
        # ax1.set_xscale('log')
        ax2 = ax1.twiny()
        ax2.set_xlabel('Scales')
        new_tick_locations = np.linspace(min(scales), max(scales), num=10).astype(int)
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(new_tick_locations[::-1])

        plt.grid(True)
    plt.show()

def cwt(signal, scales, wavelet_function, dt=1.0, fs = 100, w0 = 6, plot_wavelets_spectrum = False,
        Amp_correction_target_amp = 0.025):
    output = np.zeros((len(scales), len(signal)), dtype=np.complex_)
    generated_wavelets_mass = []
    for i, scale in enumerate(scales):
        print(f'{i}/{len(scales)}')
        t_wavelet = np.arange(-len(signal) / 2, len(signal) / 2)
        wavelet_data = wavelet_function(t_wavelet / scale, w = w0) / np.sqrt(scale) * dt

        #-----Plot Wavelet spectrum----#
        total_freq = scale_to_frequency(scale,w0,fs)
        wavelet_data = (wavelet_data / max(wavelet_data)) * Amp_correction_target_amp


        # plt.figure()
        # plt.plot(wavelet_data)
        # plt.show()
        # plot_wavelet_fourier_spectrum(wavelet_data,1/fs,scale_tit=str(scale), freq_tit=str(total_freq))
        generated_wavelets_mass.append(wavelet_data)
        # -----------------------------#

        # output[i, :] = np.convolve(signal, np.conj(wavelet_data), mode='same')
        output[i, :] = convolve_mod(signal, np.conj(wavelet_data))
    # -----Plot Wavelet spectrums----#
    if plot_wavelets_spectrum:
        print('Wavelets spectrum plot...')
        plot_spectrum_of_wavelets_mass(generated_wavelets_mass,1/fs,scales)
    # -------------------------------#
    return output

def icwt(cwt_coefficients, scales, wavelet_function, dt=1.0, ds = 1.0, Amp_correction_target_amp = 0.025):
    reconstructed_signal = np.zeros(len(cwt_coefficients[0]))
    for i, scale in enumerate(scales):
        wavelet_data = wavelet_function(
            np.arange(-len(reconstructed_signal) / 2, len(reconstructed_signal) / 2) / scale) / np.sqrt(scale)* dt * ds

        wavelet_data_corr = (wavelet_data / max(wavelet_data)) * Amp_correction_target_amp
        wavelet_data = (max(wavelet_data)/Amp_correction_target_amp)*wavelet_data_corr

        contribution = np.convolve(cwt_coefficients[i, :], wavelet_data[::-1], mode='same')
        reconstructed_signal += np.real(contribution) / (scale**2)

    # Normalization factor
    factor_kernel = wavelet_function(np.arange(-len(reconstructed_signal) / 2, len(reconstructed_signal) / 2))
    reconstruction_factor = np.sum((np.abs(factor_kernel)) ** 2)
    return reconstructed_signal / reconstruction_factor

def scale_to_frequency(scales, w0=6, fs=100):
    scales = np.array(scales)
    """Перевод масштабов вейвлета в частоты для вейвлета Морле."""
    dt = 1 / fs  # Вычисление временного шага из частоты дискретизации
    return w0 / (2 * np.pi * scales * dt)


def generate_log_scale_system(freq_from, freq_to, N_cofs):
    scale_from = scale_to_frequency([freq_to],w0=6,fs=fs)[0]
    scale_to = scale_to_frequency([freq_from],w0=6,fs=fs)[0]

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


def freq_to_scale(freq, w0=6, fs=100):
    dt = 1 / fs
    scale = w0 / (freq * 2 * np.pi * dt)
    return scale

def vlp_sig(fs = 1000):
    sig = simulate_ecg_with_VLP_ALP(duration=4,  # sec
                                    fs=fs,  # hz
                                    noise_level=40,  # db
                                    hr=80,  # bpm
                                    Std=0,  # bpm
                                    unregular_comp=False,
                                    random_state=11,
                                    lap_amp=25,
                                    lvp_amp=25)
    def recalc(fs):
        x1 = int(round((fs*560)/1000))
        x2 = int(round((fs*1400)/1000))
        return x1,x2
    x1, x2 = recalc(fs)
    return sig[x1:x2]


if __name__ == '__main__':
    fs = 500
    w0 = 6
    # signal = sigTotest(fs=fs)
    signal = vlp_sig(fs=fs)

    # signal = signal[360:]

    t = np.array(list(range(len(signal))))/fs


    scales = np.linspace(3, 200, 50)


    # scales = generate_log_scale_system(1.5,250,50)


    freqs = scale_to_frequency(scales,w0 = 6,fs = fs)


    cwt_coefficients = cwt(signal, scales, morlet_wavelet, dt=1, fs=fs,w0=6, plot_wavelets_spectrum=False)
    reconstructed_signal = icwt(cwt_coefficients, scales, morlet_wavelet, dt=1, ds=1)

    plt.figure()
    X, Y = np.meshgrid(t, scales)
    plt.pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
    plt.show(block=False)


    fig, axs = plt.subplots(3, 1, figsize=(7, 5), gridspec_kw={'height_ratios': [2, 1,1]})

    axs[1].plot(t, signal, label='Original Signal')
    axs[1].legend()
    axs[1].set_ylabel('Amplitude [mV]')

    start_time = 0.093
    end_time = 0.1267
    start_time2 = 0.2484
    end_time2 = 0.2807
    red_part = copy.deepcopy(signal)
    indices_to_keep = np.r_[np.round(start_time * fs).astype(int):np.round(end_time * fs).astype(int),
                      np.round(start_time2 * fs).astype(int):np.round(end_time2 * fs).astype(
                          int)]  # объединяем диапазоны, Python использует включительные диапазоны для срезов
    red_part[~np.in1d(np.arange(red_part.size), indices_to_keep)] = np.nan  # заменяем все, что вне диапазонов

    axs[1].plot(t, red_part, c='r')



    freqs = scale_to_frequency(scales, w0, fs)
    X, Y = np.meshgrid(t, freqs)
    im = axs[0].pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')  # Используем цветовую карту 'viridis'
    plt.colorbar(im, ax=axs[0])
    plt.colorbar(im, ax=axs[1])
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_title('Wavelet transform')

    axs[2].plot(t, reconstructed_signal, label='Reconstructed Signal', color='orange')
    axs[2].legend()
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel('time [s]')
    axs[2].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
