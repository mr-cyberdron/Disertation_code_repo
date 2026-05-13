import FILES_processing_lib
import wfdb
from Dataformats_tools.WFDB.plot_wfdb_data import plot_events_annotated_ECG, plot_WFDB_by_parts
import numpy as np

def select_lead(signals,re_format,units,adc_gain,baseline,lead_names, lead_idx = 0):
    signals = [signals[lead_idx]]
    re_format = [re_format[lead_idx]]
    units = [units[lead_idx]]
    adc_gain = [adc_gain[lead_idx]]
    baseline = [baseline[lead_idx]]
    lead_names = [lead_names[lead_idx]]
    return signals,re_format,units,adc_gain,baseline,lead_names

#data_path ='E:/Bases/arhytmia_episodes_and_beat_classif_base/mit-bih-arrhythmia-database-1.0.0/'
#data_path = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0/'
#data_path = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/cu-ventricular-tachyarrhythmia-database-1.0.0/'
#data_path = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/fantasia-database-1.0.0/'
# data_path = 'E:/Bases/0Prepared_bases/long-term-af-database_prep/'
# data_path = 'E:/Bases/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0/files/'
#data_path = 'E:/Bases/brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0 (1)/'
#data_path = 'E:/Bases/paroxysmal-atrial-fibrillation-events-detection-from-dynamic-ecg-recordings-the-4th-china-physiological-signal-challenge-2021-1.0.0/Training_set_II/'
#data_path = 'E:/Bases/paroxysmal-atrial-fibrillation-events-detection-from-dynamic-ecg-recordings-the-4th-china-physiological-signal-challenge-2021-1.0.0/Training_set_I/'
# data_path = 'E:/Bases/mit-bih-long-term-ecg-database-1.0.0/'
data_path = 'E:/Bases/t-wave-alternans-challenge-database-1.0.0/'

#data_path = 'E:/Bases/mit-bih-noise-stress-test-database-1.0.0/mit-bih-noise-stress-test-database-1.0.0/noise/'
#data_path = 'E:/Bases/0Prepared_bases/Noise/mit-bih-arrhythmia-database/'

#data_path = 'E:/Bases/0Prepared_bases/Noise/cu-ventricular-tachyarrhythmia-database/'
#data_path ='E:/Bases/0Prepared_bases/Noise/BUT_QTB/noise_rec/'
# data_path = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/long-term-af-database-1.0.0/files/'
files = FILES_processing_lib.scandir(data_path,ext='.hea')

requered_num = None
total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    if not requered_num or total_file_num >= requered_num:
        print(file)
        file_ps = data_path+file.replace('.hea','')
        record = wfdb.rdrecord(file_ps)
        samples = wfdb.rdsamp(file_ps)
        ann = wfdb.rdann(file_ps, 'qrs')


        signals = record.p_signal
        signals = signals.T
        fs = record.fs
        lead_names = record.sig_name
        annotations_pos = ann.sample
        qrs_annotations = ann.symbol
        ecg_events = ann.aux_note


        # annotations_pos = []
        # qrs_annotations = []
        # ecg_events = []

        # print(signals)
        # print(fs)
        # print(lead_names)
        # print(annotations_pos)
        # print(qrs_annotations)
        # print(ecg_events)

        # n = [9,10,11]
        # new_sigmas = []
        # new_lead_names = []
        # for i in n:
        #     new_sigmas.append(signals[i])
        #     new_lead_names.append(lead_names[i])
        # signals = new_sigmas
        # lead_names = new_lead_names

        plot_WFDB_by_parts(signals,fs,annotations_pos,qrs_annotations,ecg_events,lead_names,part_siglen_sec=1000, onlypart=False)
        try:
            plot_events_annotated_ECG(signals,fs,annotations_pos,qrs_annotations,ecg_events,lead_names, anot_pos_scale=[1,0.85])
        except:
            pass

