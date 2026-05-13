import numpy as np
import matplotlib.pyplot as plt

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

def cwt(signal, scales, wavelet_function, dt=1.0, fs = 100, w0 = 6, plot_wavelets_spectrum = False):
    output = np.zeros((len(scales), len(signal)), dtype=np.complex_)
    generated_wavelets_mass = []
    for i, scale in enumerate(scales):
        t_wavelet = np.arange(-len(signal) / 2, len(signal) / 2)
        wavelet_data = wavelet_function(t_wavelet / scale, w = w0) / np.sqrt(scale) * dt

        #-----Plot Wavelet spectrum----#
        total_freq = scale_to_frequency(scale,w0,fs)
        # plt.figure()
        # plt.plot(wavelet_data)
        # plt.show()
        # plot_wavelet_fourier_spectrum(wavelet_data,1/fs,scale_tit=str(scale), freq_tit=str(total_freq))
        generated_wavelets_mass.append(wavelet_data)
        # -----------------------------#

        output[i, :] = np.convolve(signal, np.conj(wavelet_data), mode='same')
    # -----Plot Wavelet spectrums----#
    if plot_wavelets_spectrum:
        print('Wavelets spectrum plot...')
        plot_spectrum_of_wavelets_mass(generated_wavelets_mass,1/fs,scales)
    # -------------------------------#
    return output

def icwt(cwt_coefficients, scales, wavelet_function, dt=1.0, ds = 1.0):
    reconstructed_signal = np.zeros(len(cwt_coefficients[0]))
    for i, scale in enumerate(scales):
        wavelet_data = wavelet_function(
            np.arange(-len(reconstructed_signal) / 2, len(reconstructed_signal) / 2) / scale) / np.sqrt(scale)* dt * ds
        contribution = np.convolve(cwt_coefficients[i, :], wavelet_data[::-1], mode='same')
        reconstructed_signal += np.real(contribution) / (scale**2)

    # Normalization factor
    factor_kernel = wavelet_function(np.arange(-len(reconstructed_signal) / 2, len(reconstructed_signal) / 2))
    reconstruction_factor = np.sum((np.abs(factor_kernel)) ** 2)
    return reconstructed_signal / reconstruction_factor

if __name__ == '__main__':
    fs = 500
    t_sec = 3
    w0 = 6
    n_samps = fs * t_sec
    t = np.linspace(0, t_sec, n_samps)
    signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 8 * t) * 1
    scales = np.linspace(3, 100, 20)

    scales = np.linspace(3, 50, 10)



    def scale_to_frequency(scales, w0=6, fs=100):
        scales = np.array(scales)
        """Перевод масштабов вейвлета в частоты для вейвлета Морле."""
        dt = 1 / fs  # Вычисление временного шага из частоты дискретизации
        return w0 / (2 * np.pi * scales * dt)


    def generate_log_scale_system(freq_from, freq_to, N_cofs):
        scale_from = scale_to_frequency([freq_to],w0=6,fs=fs)[0]
        scale_to = scale_to_frequency([freq_from],w0=6,fs=fs)[0]

        N_vect = np.array(list(range(N_cofs)))

        r = (scale_to/scale_from)**(1/(N_cofs-1))
        s_mass = []
        for k in N_vect:
            s_val = scale_from*r**k
            s_mass.append(s_val)
        return s_mass



    scales = generate_log_scale_system(10,150,10)


    def freq_to_scale(freq, w0=6, fs=100):
        dt = 1 / fs
        scale = w0 / (freq * 2 * np.pi * dt)
        return scale


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

    freqs = scale_to_frequency(scales, w0, fs)
    plt.figure()
    X, Y = np.meshgrid(t, freqs)
    plt.pcolormesh(X, Y, np.abs(cwt_coefficients), cmap='viridis')  # Используем цветовую карту 'viridis'
    # plt.ylim([0, 50])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('frequency')
    plt.title('Continuous Wavelet Transform (CWT) with Morlet Wavelet')
    plt.show(block=False)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='Original Signal')
    plt.legend()
    plt.title('Original Signal')

    plt.subplot(2, 1, 2)
    plt.plot(t, reconstructed_signal, label='Reconstructed Signal', color='orange')
    plt.legend()
    plt.title('Reconstructed Signal')

    plt.tight_layout()
    plt.show()
