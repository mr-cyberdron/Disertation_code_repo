import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def def_window(val1,val2, q_factor = 1):
    if val1 == 0 and val2==0:
        return 0
    diff_cof = min(abs(val1), abs(val2)) / max(abs(val1), abs(val2))
    diff_cof = diff_cof ** q_factor
    return diff_cof

def rectangular_window(val1,val2,min_bound = 0.8, max_bound = 1.2):
    if val1 == 0 and val2==0:
        diff_cof = 0
    else:
        # diff_cof = min(abs(val1), abs(val2)) / max(abs(val1), abs(val2))
        diff_cof = abs(val1)/abs(val2)
    if diff_cof>min_bound and diff_cof<max_bound:
        return 1
    else:
        return 0

def tukey_window(val1,val2, min_bound = 0.8, max_bound = 1.2,alpha = 0.5):
    if val1 == 0 and val2==0:
        diff_cof = 0
    else:
        # diff_cof = min(abs(val1), abs(val2)) / max(abs(val1), abs(val2))
        diff_cof = abs(val1)/abs(val2)
    n_counts = 100
    shift_vector = np.arange(min_bound,max_bound, (max_bound-min_bound)/n_counts)
    window = signal.windows.tukey(n_counts, alpha=alpha)
    # print(diff_cof)
    # print(window)
    # input(shift_vector)
    try:
        window_correct_coef = window[np.where(shift_vector>diff_cof)][0]
    except:
        window_correct_coef = 0
    diff_cof = window_correct_coef
    # plt.plot(shift_vector,window)
    # plt.title("Tukey window")
    # plt.ylabel("Amplitude")
    # plt.xlabel("Sample")
    # plt.ylim([0, 1.1])
    # plt.show()
    return diff_cof

# def normalized_cosine_similarity(vector1, vector2):
#     dot_prod = 0
#     norm_v1 = 0
#     norm_v2 = 0
#     for val1, val2 in zip(vector1,vector2):
#         if val1 == 0:
#             val1 = 0.00001
#         if val2 == 0:
#             val2 = 0.00001
#
#         diff_cof = 1
#         # diff_cof = def_window(val1,val2)
#         #diff_cof = rectangular_window(val1, val2, min_bound=0.8, max_bound=1.2)
#         #diff_cof = tukey_window(val1,val2,min_bound=0.8,max_bound=1.2)
#
#         dot_prod += (val1*val2)*diff_cof
#         norm_v1 +=(val1**2)
#         norm_v2+=(val2**2)
#         cos_similarity_result = (dot_prod/(np.sqrt(norm_v1)*np.sqrt(norm_v2)))
#
#         # print(dot_prod)
#         # print(norm_v1)
#         # print(norm_v2)
#         # print(cos_similarity_result)
#         # plt.figure()
#         # plt.subplot(2, 1, 1)
#         # plt.plot(vector1)
#         # plt.subplot(2, 1, 2)
#         # plt.plot(vector2)
#         # plt.show()
#     return cos_similarity_result
#

def normalized_cosine_similarity(sig_part, wavelet_part, target_value):
    sigpart_mean = np.mean(sig_part)
    # sigpart_mean = 0

    dot_prod = 0
    norm_v1 = 0
    norm_v2 = 0

    diff_cof_mass =[]

    for val1, val2 in zip(sig_part, wavelet_part):

        diff_cof = def_window(val1-sigpart_mean, target_value, q_factor = 4)
        # diff_cof = 1
        # diff_cof = rectangular_window(val1-sigpart_mean,target_value, min_bound=0.5, max_bound= 2)

        # diff_cof = tukey_window(val1-sigpart_mean,target_value, min_bound=0.8, max_bound= 1.2)
        diff_cof_mass.append(diff_cof)

        # Use the complex conjugate of val1 (from complex scalar product)
        dot_prod += (val1 * val2)#*diff_cof
        norm_v1 += np.abs(val1) ** 2
        norm_v2 += np.abs(val2) ** 2
    # return diff_cof_mass

    # Avoid division by zero by checking norms before division
    if np.sqrt(norm_v1) == 0 or np.sqrt(norm_v2) == 0:
        return 0
    else:
        cos_similarity_result = dot_prod / (np.sqrt(norm_v1) * np.sqrt(norm_v2))

    return cos_similarity_result*np.mean(diff_cof_mass)

def dot_prod(v1,v2):
    dot_prod = 0



    for val1, val2 in zip(v1, v2):
        dot_prod += (val1 * np.conj(val2))

    return dot_prod


def dot_prod_weighted(v1, v2):
    weigh_mass = []
    dot_prod = 0

    for val1, val2 in zip(v1, v2):
        cof = def_window(val1,val2, q_factor=2)
        weigh_mass.append(cof)
        dot_prod += (val1 * np.conj(val2))*cof

    return weigh_mass, dot_prod
