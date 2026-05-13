import matplotlib.pyplot as plt

from Amp_correction_coefs import def_window,rectangular_window,tukey_window,normalized_cosine_similarity,dot_prod,dot_prod_weighted
import numpy as np


def convolve_same2(a, b):
    # Длины исходных массивов
    len_a, len_b = len(a), len(b)

    # Перевернем второй массив
    b = b[::-1]

    # Результат свертки
    result = []

    # Рассчитаем свертку, используя метод слайдинга окна
    for i in range(1 - len_b, len_a):
        sum_conv = 0
        for j in range(len_b):
            if 0 <= i + j < len_a:
                a_val = a[i + j]
                b_val = b[j]
                corr_cof = def_window(a_val,b_val)
                # corr_cof = rectangular_window(a_val,b_val)
                # corr_cof = tukey_window(a_val,b_val)
                # corr_cof = 1
                sum_conv += ((a[i + j]) * b[j])*corr_cof
        result.append(sum_conv)

    # Определим начало и конец среза для получения результата размером 'same'
    if len_a > len_b:
        start = (len_b - 1) // 2
        end = start + len_a
    else:
        start = (len_a - 1) // 2
        end = start + len_b

    return result[start:end]


def convolve_cosine_sim_based_mod(sig, wavelet, target_value):
    # wavelet = wavelet[::-1]
    if len(wavelet)%2 != 0:
        wavelet = wavelet[1:]
    wavle_half_pos = int(np.round(len(wavelet)/2))

    similarity_results_mass = []
    for i in range(len(sig)):
        wavelet_start_pos = max(wavle_half_pos-i, 0)
        wavelet_end_pos = min(wavle_half_pos + (len(sig) - i),len(wavelet))
        wavelet_part = wavelet[wavelet_start_pos:wavelet_end_pos]

        start_sig_point = max(i - wavle_half_pos, 0)
        stop_sig_point = min(i + wavle_half_pos , len(sig))
        compared_part = sig[start_sig_point:stop_sig_point]

        # plt.figure()
        # plt.title((str(normalized_cosine_similarity(compared_part,wavelet_part))))
        # plt.subplot(2,1,1)
        # plt.plot(compared_part)
        # plt.subplot(2,1,2)
        # plt.plot(wavelet_part)
        # plt.show()
        #
        # similarity_result = normalized_cosine_similarity(compared_part,wavelet_part)
        # plt.figure()
        # plt.subplot(3,1,1)
        # plt.plot(compared_part)
        # plt.subplot(3, 1, 2)
        # plt.plot(wavelet_part)
        # plt.subplot(3, 1, 3)
        # plt.plot(similarity_result)
        # plt.show()
        similarity_results_mass.append(normalized_cosine_similarity(compared_part,wavelet_part, target_value))
        # similarity_results_mass[i] = normalized_cosine_similarity(compared_part, np.real(wavelet_part))

    return np.array(similarity_results_mass)

def convolve_mod(sig, wavelet):
    if len(wavelet)%2 != 0:
        wavelet = wavelet[1:]
    wavle_half_pos = int(np.round(len(wavelet)/2))

    conv_mass = []
    for i in range(len(sig)):
        wavelet_start_pos = max(wavle_half_pos-i, 0)
        wavelet_end_pos = min(wavle_half_pos + (len(sig) - i),len(wavelet))
        wavelet_part = wavelet[wavelet_start_pos:wavelet_end_pos]

        start_sig_point = max(i - wavle_half_pos, 0)
        stop_sig_point = min(i + wavle_half_pos , len(sig))
        compared_part = sig[start_sig_point:stop_sig_point]

        weigh, prod = dot_prod_weighted(compared_part, wavelet_part)
        conv_mass.append(prod)
        # conv_mass.append(dot_prod(compared_part,wavelet_part))

    return np.array(conv_mass)



def fix_length(sig1, sig2):
    if len(sig1)>len(sig2):
        def_l = len(sig1) - len(sig2)
        return np.array(sig1)[0:-def_l], sig2

    if len(sig2)>len(sig1):
        def_l = len(sig2) - len(sig1)
        return sig1, np.array(sig2)[0:-def_l]

def custom_conv_with_metric(sig, wavelet_body):
    print('custom conv')
    sig = np.array(sig)
    wavelet_body = np.array(wavelet_body)
    # Create an empty output mass
    output_mass = [0] * len(sig)
    wavelet_len = len(wavelet_body)
    half_wavelet_offset = ((wavelet_len - 1) / 2).__floor__()
    for sig_point_num in range(len(sig)):
        start_sig_point = max(sig_point_num-half_wavelet_offset,0)
        stop_sig_point = min(sig_point_num+half_wavelet_offset+1,len(sig)-1)
        start_wavelet_point = max(half_wavelet_offset-sig_point_num, 0)
        stop_wavelet_point =  min(half_wavelet_offset + (len(sig)-1 - sig_point_num),wavelet_len-1)

        signal_part_mass = sig[start_sig_point:stop_sig_point+1]
        wavelet_part_mass = wavelet_body[start_wavelet_point:stop_wavelet_point+1]
        if len(signal_part_mass) == len(wavelet_part_mass):
            pass
        else:
            signal_part_mass,wavelet_part_mass = fix_length(signal_part_mass,wavelet_part_mass)
            print(len(signal_part_mass))
            print(len(wavelet_part_mass))
            print(start_sig_point,stop_sig_point,start_wavelet_point,stop_wavelet_point)
            print(wavelet_len)
            print(half_wavelet_offset)
            print ("Signal part and wavelet parts should be equal")

        # signal_part_mass = signal_part_mass - np.mean(signal_part_mass) #make something, need to test on more complicated sig

        #similarity_coef = sf.Cosine_similarity(signal_part_mass, np.real(wavelet_part_mass))
        # similarity_coef = sf.Pearson_Corr(signal_part_mass, np.real(wavelet_part_mass))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(signal_part_mass)
        # plt.subplot(2, 1, 2)
        # plt.plot(np.real(wavelet_part_mass))
        # plt.show()

        similarity_coef = normalized_cosine_similarity(signal_part_mass,np.real(wavelet_part_mass))

        output_mass[int(sig_point_num)] = similarity_coef


    return output_mass

