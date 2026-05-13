import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywt



def morletc(t, w=5.0, s=1.0, complete=True):

    x = np.linspace(-t[-1], t[-1], len(t))
    x = x/s
    output = np.exp(1j * w * x)

    if complete:
        output -= np.exp(-0.5 * (w**2))

    output *= np.exp(-0.5 * (x**2)) * np.pi**(-0.25)
    return output


def continuous_wavelet_transform(signal, scales):
    n = len(signal)
    cwt_matrix = np.zeros((len(scales), n), dtype=complex)

    for i, scale in enumerate(scales):
        wavelet = morletc(time, s=scale, complete=True)
        # plt.figure()
        # plt.plot(wavelet)
        # plt.show()
        scale_corr = 1/(scale**0.5)
        cwt_matrix[i, :] = scale_corr*np.convolve(signal, wavelet, mode='same')

    return cwt_matrix


def icwt(cwt_coeffs, scales):
    reconstructed_signal = np.sum(cwt_coeffs / np.sqrt(scales[:, None]), axis=0)
    return reconstructed_signal.real


def morlet_wavelet_conjugate(t, omega0=6):
    """
    Compute the complex conjugate of the Morlet wavelet.
    :param t: Time or position array.
    :param omega0: Central frequency of the Morlet wavelet.
    :return: Complex conjugate of the Morlet wavelet at each time t.
    """
    # Morlet wavelet function (complex conjugate)
    C = np.pi ** (-0.25)
    wavelet = C * (np.exp(-1j * omega0 * t) - np.exp(-0.5 * omega0 ** 2)) * np.exp(-0.5 * t ** 2)
    return np.conj(wavelet)
def reconstruct_signal_from_cwt(cwt_coeffs, scales, positions,delta_a,delta_b, omega0=5):
    """
    Reconstruct a signal from its CWT coefficients using the Morlet wavelet.
    :param cwt_coeffs: 2D array of CWT coefficients, indexed by scale and position.
    :param scales: Array of scales used in CWT.
    :param positions: Array of positions used in CWT.
    :param delta_a: Scale discretization interval.
    :param delta_b: Position discretization interval.
    :param omega0: Central frequency of the Morlet wavelet.
    :return: Reconstructed signal as a 1D array.
    """
    # Assuming scales and positions are 1D arrays
    n_positions = len(positions)
    reconstructed_signal = np.zeros(n_positions)

    # Normalization constant, for simplicity assuming a fixed value
    # This should be calculated based on the actual Morlet wavelet used
    C_psi = 1  # This is a placeholder; you need to compute this based on your wavelet

    for i, b in enumerate(positions):
        sum_over_scales = 0
        for j, a in enumerate(scales):
            wavelet_conj = morlet_wavelet_conjugate((i - b) / a, omega0)
            sum_over_scales += cwt_coeffs[j, i] * wavelet_conj * delta_a * delta_b/ a ** 2
        reconstructed_signal[i] = sum_over_scales / C_psi

    return reconstructed_signal   # Multiply by delta_b as part of the reconstruction integral

def generate_scales(dj, dt, N):
    s0 = 2*dt
    # Largest scale
    J = int((1 / dj) * np.log2(N * dt / s0))
    sj = s0 * 2 ** (dj * np.arange(0, J + 1))
    return sj




# Example usage
if __name__ == '__main__':
    # Create a sample signal: a sine wave
    time = np.linspace(0, 1, 400)
    signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 8 * time) * 1

    # scales = np.arange(1, 10)
    # cwt_matrix = continuous_wavelet_transform(signal, scales, morlet)
    # reconstruct = icwt(cwt_matrix,scales)

    # delta_a = scales[1] - scales[0]  # Assuming uniform scale steps for simplicity
    # delta_b = time[1] - time[0]  # Assuming uniform position steps
    # reconstruct2 = reconstruct_signal_from_cwt(cwt_matrix,scales,time*400,delta_a,1)

    dj = 0.125
    delta_b = 0.0025
    # scales = generate_scales(dj,delta_b, len(signal))
    scales = np.arange(1, 1000)
    cwt_matrix, frequencies = pywt.cwt(signal, scales, 'morl')
    # cwt_matrix = continuous_wavelet_transform(signal, scales)
    reconstruct2 = icwt(cwt_matrix,scales)
    # reconstruct2 = reconstruct_signal_from_cwt(cwt_matrix, scales, time*400 , dj, delta_b)

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(signal)
    plt.subplot(3,1,2)
    plt.plot(reconstruct2)
    plt.subplot(3,1,3)
    plt.imshow(np.abs(cwt_matrix), extent=[0, 1, 1, 31], aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.title('Continuous Wavelet Transform (CWT)')
    plt.show()
