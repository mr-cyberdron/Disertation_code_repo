import copy

import numpy as np
import pandas as pd
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb_no_ann,write_wfdb
from Frequency_tools.Filtering.AnalogFilters import AnalogFilterDesign
import random
from scipy.signal import resample
import neurokit2 as nk
from tools import plotmultileadSig

def generate_LP_signal(lp_signal_fs):
    #LP duration = 0.3 sec
    max_len = 2500
    sig_frag_csv = pd.read_csv('artif_samples.csv')

    sigpart1 = list(sig_frag_csv["1"].to_numpy()[0:max_len])
    sigpart2 = list(sig_frag_csv["2"].to_numpy()[0:max_len])
    sigpart3 = list(sig_frag_csv["3"].to_numpy()[0:max_len])
    sigpart4 = list(sig_frag_csv["4"].to_numpy()[0:max_len])

    lp_dur_sec = 0.8
    fs = 10000
    iter_num = 30

    parts_mass = [sigpart1, sigpart2, sigpart3, sigpart4]
    empty_sig = np.zeros(int(lp_dur_sec * fs))

    old_sig = empty_sig
    new_sig = []

    for i in range(iter_num):
        shift = int(np.round(random.random() * len(empty_sig)))
        part = random.choice(parts_mass)
        new_empty_sig = np.zeros(int(lp_dur_sec * fs))
        new_empty_sig[shift:min(len(empty_sig) - 1, shift + max_len)] = part[0:min(len(empty_sig) - 1,
                                                                                   shift + max_len) - shift]
        new_sig = old_sig + new_empty_sig
        old_sig = new_sig

    noize_sig = np.random.normal(0, 0.01, int(lp_dur_sec * fs))
    new_sig = new_sig + noize_sig

    new_sig = AnalogFilterDesign(new_sig, fs).lp(order=5, cutoff=250).zerophaze().bessel().filtration()
    new_sig = AnalogFilterDesign(new_sig, fs).hp(order=5, cutoff=1).zerophaze().bessel().filtration()

    new_sig = new_sig - np.mean(new_sig)
    # Target sampling frequency
    target_sampling_frequency = lp_signal_fs

    length_in_samp = lp_dur_sec * fs
    resampled_new_sig = resample(new_sig, int(length_in_samp * target_sampling_frequency / fs))

    return resampled_new_sig

def filter_LP(sig,fs):
    return AnalogFilterDesign(sig,fs).lp(order=5, cutoff=150).zerophaze().butter().filtration()

def filter_ecg(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 60]).zerophaze().butter().filtration()

def filter_ecg2(sig,fs):
    return AnalogFilterDesign(sig,fs).bp(order=5, cutoff=[1, 200]).zerophaze().butter().filtration()

def filter50(sig,fs):
    return AnalogFilterDesign(sig, fs).zerophaze().notch(cutoff=50, quality_factor=50).filtration()


def mount_LP_to_ecg(signal,fs, qrs_peaks_pos, LP_sample,LP_start_from_qrs_sec = 0.035):
    signal_to_change = copy.deepcopy(signal)
    LP_start_from_qrs_samp = np.ceil(LP_start_from_qrs_sec*fs)
    for peak_pos in qrs_peaks_pos:
        LP_start = int(peak_pos+LP_start_from_qrs_samp)
        LP_stop = int(peak_pos+LP_start_from_qrs_samp+len(LP_sample))
        if LP_stop<len(signal_to_change):
            signal_to_change[LP_start:LP_stop] = signal_to_change[LP_start:LP_stop]+(LP_sample)
        # peak_to_plot_from = max(0, peak_pos-500)
        # peak_to_plot_to = min(peak_pos + 500, len(signal))
        # plot_fragm = signal_to_change[peak_to_plot_from:peak_to_plot_to]
        # print(np.shape(plot_fragm))
        # plt.plot(plot_fragm)
        # plt.show()
    return signal_to_change

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

def generateLpSignal(fs,scale,dur_samp):
    Lp_signal = generate_LP_signal(fs) * scale  # 0.35  # 0.08
    Lp_signal = filter_LP(Lp_signal, fs)
    Lp_signal = Lp_signal[0:dur_samp]
    return Lp_signal

def dropNanfromNParray(arr):
    return [i for i in arr if not np.isnan(i)]
def add_VLP_lp_to_signal(signal,fs, Lp_signal):
    lp_startpos = 0.035
    filtered_sig = filter_ecg(signal, fs)
    stat, sig_peaks = nk.ecg_process(filtered_sig, sampling_rate=fs, method='neurokit')  # kalidas2017
    detected_peaks = sig_peaks['ECG_R_Peaks']
    detected_peaks = dropNanfromNParray(detected_peaks)
    ecg_with_lp = mount_LP_to_ecg(signal, fs, detected_peaks, Lp_signal, LP_start_from_qrs_sec=lp_startpos)
    return ecg_with_lp

def add_ALP_lp_to_signal(signal,fs,Lp_signal):
    lp_startpos = 0.027
    filtered_sig = filter_ecg(signal, fs)
    stat, sig_peaks = nk.ecg_process(filtered_sig, sampling_rate=fs, method='neurokit')  # kalidas2017
    detected_peaks = sig_peaks['ECG_P_Peaks']
    detected_peaks = dropNanfromNParray(detected_peaks)
    ecg_with_lp = mount_LP_to_ecg(signal, fs, detected_peaks, Lp_signal, LP_start_from_qrs_sec=lp_startpos)
    return ecg_with_lp

def addLPtolead(sig, fs, lap:bool, lvp:bool,  LVP_sig_part, LAP_sig_part):

    sig = filter_ecg2(sig, fs) / max(abs(filter_ecg2(sig, fs)[200:600]))
    sig = filter50(sig, fs)


    if lvp:
        sig = add_VLP_lp_to_signal(sig, fs, LVP_sig_part)
    if lap:
        sig = add_ALP_lp_to_signal(sig, fs, LAP_sig_part)

    # filtered_sig = filter_ecg(sig, fs)
    # stat, sig_peaks = nk.ecg_process(filtered_sig, sampling_rate=fs, method='neurokit')  # kalidas2017
    # detected_peaks = sig_peaks['ECG_R_Peaks']
    # avg_card = calc_avg_card(sig,fs,detected_peaks)
    # plotmultileadSig([avg_card])
    return sig

global lap_sig_part_success
global lvp_sig_part_success
def addLP(filePath, DbToAddMetadata, lap:bool, lvp:bool):
    signals, re_format, fs, units, adc_gain, baseline, \
        coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb_no_ann(filePath)
    try:
        LVP_sig_part = generateLpSignal(fs, 0.3, 30)
        global lvp_sig_part_success
        lvp_sig_part_success = LVP_sig_part
    except:
        LVP_sig_part = lvp_sig_part_success

    try:
        LAP_sig_part = generateLpSignal(fs, 0.65, 20)
        global lap_sig_part_success
        lap_sig_part_success = LAP_sig_part
    except:
        LAP_sig_part = lap_sig_part_success


    num_leads_with_lp = np.random.randint(4, 6) #4-5 leads
    leads_nums = list(range(2,13,1))
    leads_for_lp = np.random.choice(leads_nums, num_leads_with_lp, replace=False)
    leads_for_lp = np.insert(leads_for_lp, 0, 1)

    new_sigmass = []
    counter = 0
    for lead_sig in signals:
        counter+=1
        if counter in leads_for_lp:
            sig_with_lp = addLPtolead(lead_sig,fs, lap,lvp,LVP_sig_part, LAP_sig_part)
            new_sigmass.append(sig_with_lp)
        else:
            new_sigmass.append(lead_sig)


    new_fname = filePath.split('/')[-1]
    if lap:
        new_fname = new_fname+'_LAP'
    if lvp:
        new_fname = new_fname+'_LVP'

    filePath_to_index = '/'.join(filePath.split('/')[-3:])
    metadata_line = DbToAddMetadata[DbToAddMetadata['filename_hr'] == filePath_to_index]
    new_fpath_from_root = 'latePotentialsSig/'+new_fname
    full_fpath_to_save = initial_db_path + 'latePotentialsSig/'

    new_metadata_line = copy.deepcopy(metadata_line)
    new_metadata_line['filename_hr'] = new_fpath_from_root
    new_metadata_line['filename_lr'] = None
    old_scp_codes = eval(new_metadata_line['scp_codes'].to_numpy()[0])
    old_scp_codes.pop('NORM',None)
    old_scp_codes.pop('SR', None)

    if lap and not lvp:
        old_scp_codes['LAP'] = 100.0
        new_metadata_line['scp_codes'] = str(old_scp_codes)
    if lvp and not lap:
        old_scp_codes['LVP'] = 100.0
        new_metadata_line['scp_codes'] = str(old_scp_codes)
    if lap and lvp:
        old_scp_codes['LAP'] = 100.0
        old_scp_codes['LVP'] = 100.0
        new_metadata_line['scp_codes'] = str(old_scp_codes)

    try:
        write_wfdb(new_fname, np.array(new_sigmass).T, fs, re_format, units, None,
                   None, coments, lead_names, p=full_fpath_to_save)
        DbToAddMetadata = pd.concat([DbToAddMetadata, new_metadata_line], ignore_index=True)
    except:
        print('WFDB write exeption')

    return DbToAddMetadata



initial_db_path = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

metadata_file = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/Normalised_PTB_XL_metadata.csv"


metadata_without_late_potentials = pd.read_csv(metadata_file)

metadata_with_late_potentials_dump = copy.deepcopy(metadata_without_late_potentials)

lp_possible = ['PVC','WPW','LVH','ISCAL','ISCIN',
               'ISCIL','ISCAS','ISCLA','ISCAN',
               'IMI','ASMI','AMI','ALMI','ILMI',
               'IPLMI', 'IPMI','LMI','PMI','CRBBB','CLBBB']

only_norm_row_ids = []
lp_possible_row_ids = []
for i,row in metadata_without_late_potentials.iterrows():
    scp_codes = list(eval(row['scp_codes']).keys())
    doctor_validate = bool(row['validated_by_human'])
    if doctor_validate and (scp_codes == ['NORM', 'SR'] or scp_codes == ['NORM']):
        only_norm_row_ids.append(i)

    for lp_possible_code in lp_possible:
        if doctor_validate and lp_possible_code in scp_codes:
            lp_possible_row_ids.append(i)


only_norm_row_ids = np.unique(only_norm_row_ids)
lp_possible_row_ids = np.unique(lp_possible_row_ids)

random_state = np.random.RandomState(42)

norm_sigs_part = random_state.choice(only_norm_row_ids, 1500, replace=False,)
lp_possible_part = random_state.choice(lp_possible_row_ids, 500, replace=False)

ids_for_lp_attach = np.concatenate((norm_sigs_part, lp_possible_part), axis=0)

ids_for_lp_attach_LVP = ids_for_lp_attach[0:850]
ids_for_lp_attach_LAP = ids_for_lp_attach[851:1700]
ids_for_lp_attach_LVP_LAP = ids_for_lp_attach[1701:2000]

counter = 0
for id in ids_for_lp_attach_LVP:
    counter+=1
    print(f'{counter}/{len(ids_for_lp_attach)}')
    lp_to_attach_row = metadata_without_late_potentials.loc[id]
    hr_file_name_path = lp_to_attach_row['filename_hr']
    try:
        metadata_with_late_potentials_dump = addLP(initial_db_path+hr_file_name_path,metadata_with_late_potentials_dump, False,True)
    except:
        pass
for id in ids_for_lp_attach_LAP:
    counter += 1
    print(f'{counter}/{len(ids_for_lp_attach)}')
    lp_to_attach_row = metadata_without_late_potentials.loc[id]
    hr_file_name_path = lp_to_attach_row['filename_hr']
    try:
        metadata_with_late_potentials_dump = addLP(initial_db_path+hr_file_name_path,metadata_with_late_potentials_dump, True,False)
    except:
        pass

for id in ids_for_lp_attach_LVP_LAP:
    counter += 1
    print(f'{counter}/{len(ids_for_lp_attach)}')
    lp_to_attach_row = metadata_without_late_potentials.loc[id]
    hr_file_name_path = lp_to_attach_row['filename_hr']
    try:
        metadata_with_late_potentials_dump = addLP(initial_db_path+hr_file_name_path,metadata_with_late_potentials_dump, True,True)
    except:
        pass
metadata_with_late_potentials_dump.to_csv('metadata_normalised_LP_atached.csv')