import pandas as pd
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb_no_ann,write_wfdb
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import neurokit2 as nk
import numpy as np
from tools import plotmultileadSig
def calc_avg_card(signal,fs, qrs_poss):
    card_boundaries_sec = 0.5
    card_boundaries_samples = int(round(card_boundaries_sec*fs))
    fragms_mass = []
    for pos in qrs_poss:
        pos_from = pos-card_boundaries_samples
        pos_to = pos+card_boundaries_samples
        target_fragm_len = pos_to-pos_from
        signal_fragm = signal[pos_from:pos_to]
        if len(signal_fragm)== target_fragm_len:
            fragms_mass.append(signal_fragm)
    avg_card = np.array(fragms_mass).mean(axis=0)
    return avg_card

def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()

def filter_ecg2(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 100]).zerophaze().butter().filtration()

def filter50(sig,fs):
    return AnalogFilterDesign(sig, fs).zerophaze().notch(cutoff=50, quality_factor=50).filtration()

initial_db_path = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
metadata_file = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/metadata_normalised_LP_atached.csv"

signals_metadata = pd.read_csv(metadata_file)

avg_qrs_metadata_dict = {'num':[], 'fname_hr':[], 'scp_codes':[]}

counter = 0
for i,row in signals_metadata.iterrows():
    try:
        counter+=1
        print(f'{counter}/{len(signals_metadata)}')
        filename_hr = row['filename_hr']
        scp_codes = list(eval(row['scp_codes']).keys())
        wfdb_file_path = initial_db_path+filename_hr
        signals, re_format, fs, units, adc_gain, baseline, \
            coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb_no_ann(wfdb_file_path)

        sig = filter50(filter_ecg2(signals[0],fs),fs)
        # plotmultileadSig([sig])
        filtered_sig = filter_ecg(sig, fs)
        stat, sig_peaks = nk.ecg_process(filtered_sig, sampling_rate=fs, method='neurokit')  # kalidas2017
        detected_peaks = sig_peaks['ECG_R_Peaks']
        avg_card = calc_avg_card(sig,fs,detected_peaks)
    except:
        pass

    np.save(initial_db_path+'QRS_averaged/'+str(counter)+'_AVG_QRS_500_hz.npy', avg_card)

    avg_qrs_metadata_dict['num'].append(counter)
    avg_qrs_metadata_dict['fname_hr'].append(filename_hr)
    avg_qrs_metadata_dict['scp_codes'].append(scp_codes)


pd.DataFrame(data=avg_qrs_metadata_dict).to_csv(initial_db_path+'QRS_averaged/AVG_QRS_metadata.csv')

