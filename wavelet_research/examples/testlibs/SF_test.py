import Artef_sig
import numpy as np
import matplotlib.pyplot as plt
import SimilarityFunctionsDevelop


def compare_SF(signal1, signal2, func):
    plt.figure()
    plt.plot(signal1)
    plt.plot(signal2)
    plt.legend(['sig1','sig2'])
    print(func(signal1,signal2))
    plt.show()


fs = 1000
sig1 = Artef_sig.sin(30,amp=1,fs=fs)
sig2 = Artef_sig.break_sin(30,amp=1,fs=fs)

sin3 = Artef_sig.signal_zerofill_right(sig1,fs,0.2)
sin4 = Artef_sig.signal_zerofill_left(sig1,fs,0.2)


# compare_SF(sig1,sig1*20, SimilarityFunctionsDevelop.normalized_cosine_similarity)



# def amp_filter_test():
#     div_range = np.arange(0,2, 1/200)
#     dif_cof_mas = []
#     for div_count in div_range:
#         val1 = 1
#         val2 = val1*div_count
#         diff_cof = SimilarityFunctionsDevelop.tukey_window(val1,val2,min_bound=0.8,max_bound=1.2)
#         dif_cof_mas.append(diff_cof)
#     plt.figure()
#     plt.plot(div_range,dif_cof_mas)
#     plt.show()
#
# amp_filter_test()