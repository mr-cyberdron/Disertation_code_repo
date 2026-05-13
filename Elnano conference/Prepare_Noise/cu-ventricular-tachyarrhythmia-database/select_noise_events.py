import numpy as np

import FILES_processing_lib
import wfdb
from Dataformats_tools.WFDB.plot_wfdb_data import plot_events_annotated_ECG, plot_WFDB_by_parts

def sig_slice (signals, annotations_pos, qrs_annotations, ecg_events, from_sample,to_sample):
    signals_new = signals[:,from_sample:to_sample]
    new_pos =[]
    new_qrs =[]
    new_events = []
    for pos,qrs,events in zip(annotations_pos,qrs_annotations,ecg_events):
        if pos>=from_sample and pos <=to_sample:
            new_pos.append(pos)
            new_qrs.append(qrs)
            new_events.append(events)
    new_pos = np.array(new_pos)
    new_pos = new_pos-from_sample

    return signals_new, new_pos,new_qrs,new_events

def plot_noise_episodes(signals,fs,annotations_pos,qrs_annotations,ecg_events,lead_names, signame):
    noise_start_pos = None
    noise_end_pos = None
    for anot,qrs,event in zip(annotations_pos,qrs_annotations,ecg_events):
        if qrs == '~' and noise_start_pos != None:
            noise_end_pos = anot
        if qrs == '~' and noise_end_pos == None and noise_start_pos == None:
            noise_start_pos = anot

        if noise_start_pos != None and noise_end_pos != None:
            print(f"'{file}':[{noise_start_pos},{noise_end_pos}]")
            signals_new, new_pos,new_qrs,new_events = sig_slice(signals,annotations_pos,qrs_annotations,ecg_events,noise_start_pos,noise_end_pos)
            plot_events_annotated_ECG(signals_new, fs, new_pos, new_qrs, new_events, lead_names)
            noise_start_pos = None
            noise_end_pos = None

data_path ='D:/Bases/0Prepared_bases/mit-bih-arrhythmia-database_prepared/'

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
        ann = wfdb.rdann(file_ps, 'atr')


        signals = record.p_signal
        signals = signals.T
        fs = record.fs
        lead_names = record.sig_name
        annotations_pos = ann.sample
        qrs_annotations = ann.symbol
        ecg_events = ann.aux_note

        plot_noise_episodes(signals,fs,annotations_pos,qrs_annotations,ecg_events,lead_names,file)


        plot_events_annotated_ECG(signals,fs,annotations_pos,qrs_annotations,ecg_events,lead_names, anot_pos_scale=[1,0.85])


