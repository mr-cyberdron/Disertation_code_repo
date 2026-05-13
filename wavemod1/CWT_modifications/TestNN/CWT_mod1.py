import copy

import numpy as np
import matplotlib.pyplot as plt
from Convolve_mod import convolve_same2, custom_conv_with_metric, convolve_cosine_sim_based_mod, convolve_mod
def morlet_wavelet(t, w=6.0):
    wavelet = np.exp(1j * w * t) * np.exp(-0.5 * t ** 2) * np.pi ** (-0.25)
    return wavelet

# def scale_to_frequency(scales, w0, fs):
#     scales = np.array(scales)
#     frequencies = (w0 / (2 * np.pi * scales)) * fs
#     return frequencies

def scale_to_frequency(scales, w0=6, fs=100):
    scales = np.array(scales)
    """Перевод масштабов вейвлета в частоты для вейвлета Морле."""
    dt = 1 / fs  # Вычисление временного шага из частоты дискретизации
    return w0 / (2 * np.pi * scales * dt)

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
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()


def plot_spectrum_of_wavelets_mass(wavelet_mass, dt, scales):
    fig, ax1 = plt.subplots()
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
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude')
        ax1.set_xlim(0,max(freq))
        ax1.set_xscale('log')
        ax2 = ax1.twiny()
        ax2.set_xlabel('Scales')
        new_tick_locations = np.linspace(min(scales), max(scales), num=10).astype(int)
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(new_tick_locations[::-1])

        plt.grid(True)
    plt.show()


def cut_wavelet(wavelet_data, cut_treshold = 0.001):
    weighted_treshold = np.max(wavelet_data)*cut_treshold
    wavlet_center = np.round(len(wavelet_data)/2)
    old_count = wavelet_data[0]
    wavelet_startpos = 0
    wavlet_endpos = len(wavelet_data)-1
    for i, wavelet_count in enumerate(wavelet_data):
        delta_count = abs(wavelet_count-old_count)
        old_count = wavelet_count
        if delta_count>=weighted_treshold:
            wavelet_startpos = i
            half_width = wavlet_center-wavelet_startpos#-1
            wavlet_endpos = wavlet_center+half_width
            break

    # return wavelet_data[int(wavelet_startpos):int(wavlet_endpos)]
    return wavelet_data[int(wavelet_startpos):int(wavlet_endpos + 1)]

def cwt(signal, scales, wavelet_function, dt=1.0, fs = 100, w0 = 6, plot_wavelets_spectrum = False,
        Amp_correction_target_amp = 0.025
        ):

    output = np.zeros((len(scales), len(signal)), dtype=np.complex_)
    generated_wavelets_mass = []
    # scale_corrections_mass = []
    for i, scale in enumerate(scales):
        #print(f'scale {i}/{len(scales)}')
        t_wavelet = np.arange(-len(signal) / 2, len(signal) / 2)
        wavelet_data = wavelet_function(t_wavelet / scale, w = w0) / np.sqrt(scale) * dt

        # -----------------mod----------------#
        wavelet_data = cut_wavelet(wavelet_data, cut_treshold=0.03) #0.3 for better reconstruction but bad freq resolution
        # plt.figure()
        # plt.plot(wavelet_data)
        # plt.show()

        wavelet_data_old = copy.deepcopy(wavelet_data)
        # scale wavelet_to_targe
        wavelet_data = (wavelet_data/max(wavelet_data))*Amp_correction_target_amp

        # scale_corrections_mass.append(scale_correction)
        # -------------------------------------

        #-----Plot Wavelet spectrum----#
        # total_freq = scale_to_frequency(scale,w0,fs)
        # plt.figure()
        # plt.plot(wavelet_data)
        # plt.show()
        # plot_wavelet_fourier_spectrum(wavelet_data,1/fs,scale_tit=str(scale), freq_tit=str(total_freq))
        generated_wavelets_mass.append(wavelet_data)
        # -----------------------------#
        # wavelet_data = wavelet_data[::-1]
        # output[i, :] = np.convolve(signal, np.conj(wavelet_data), mode='same')
        # output[i, :] = np.convolve(signal, wavelet_data, mode='same')
        # output[i, :] = convolve_same2(signal, np.conj(wavelet_data))
        # output[i, :] = custom_conv_with_metric(signal, np.conj(wavelet_data))


        # output[i, :] = convolve_cosine_sim_based_mod(signal, wavelet_data, Amp_correction_target_amp)
        output[i, :] = convolve_cosine_sim_based_mod(signal, np.conj(wavelet_data),Amp_correction_target_amp)#*max(wavelet_data_old)
        # output[i, :] = convolve_mod(signal, np.conj(wavelet_data_old))

    # -----Plot Wavelet spectrums----#
    if plot_wavelets_spectrum:
        print('Wavelets spectrum plot...')
        plot_spectrum_of_wavelets_mass(generated_wavelets_mass,1/fs,scales)
    # -------------------------------#
    return output

def icwt(cwt_coefficients, scales, wavelet_function, dt=1.0, ds = 1.0):

    reconstructed_signal = np.zeros(len(cwt_coefficients[0]))
    for i, scale in enumerate(scales):
        print(f'Reconstruction scale {i}/{len(scales)}')
        wavelet_data = wavelet_function(
            np.arange(-len(reconstructed_signal) / 2, len(reconstructed_signal) / 2) / scale) / np.sqrt(scale)* dt * ds
        wavelet_data = wavelet_data[::-1]
        # contribution = np.convolve(cwt_coefficients[i, :], wavelet_data, mode='same')
        contribution = np.convolve(cwt_coefficients[i, :], np.conj(wavelet_data), mode='same')
        # contribution = convolve_same2(cwt_coefficients[i, :], wavelet_data[::-1])
        # contribution = convolve_cosine_sim_based_mod(cwt_coefficients[i, :], wavelet_data[::-1])


        reconstructed_signal += np.real(contribution) / (scale**2)

    # Normalization factor
    factor_kernel = wavelet_function(np.arange(-len(reconstructed_signal) / 2, len(reconstructed_signal) / 2))
    reconstruction_factor = np.sum((np.abs(factor_kernel)) ** 2)
    return reconstructed_signal / reconstruction_factor
