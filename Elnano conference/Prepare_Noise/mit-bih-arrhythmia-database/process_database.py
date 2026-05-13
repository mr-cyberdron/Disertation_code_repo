import FILES_processing_lib
import numpy as np
from process_peculiarities import peculiarities
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb

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

def select_lead(signals,re_format,units,adc_gain,baseline,lead_names, lead_idx = 0):
    signals = [signals[lead_idx]]
    re_format = [re_format[lead_idx]]
    units = [units[lead_idx]]
    adc_gain = [adc_gain[lead_idx]]
    baseline = [baseline[lead_idx]]
    lead_names = [lead_names[lead_idx]]
    return signals,re_format,units,adc_gain,baseline,lead_names

def fix_anot_pos(old_anot):
    new_anot = []
    for anot_it in old_anot:
        if anot_it>=0:
            new_anot.append(anot_it)
        else:
            new_anot.append(0)
    return np.array(new_anot)

db_source = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/mit-bih-arrhythmia-database-1.0.0/'
storing_path = 'E:/Bases/0Prepared_bases/Noise/mit-bih-arrhythmia-database/'

files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    print(file)
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'atr')

    if file in list(peculiarities.keys()):
        noise_frags_mass = peculiarities[file]
        frag_count = 0
        for frag in noise_frags_mass:
            frag_count = frag_count+1
            signals_new, new_pos, new_qrs, new_events = sig_slice(signals, annotations_pos, qrs_annotations, ecg_events, frag[0],frag[1])
            if len(frag) == 3:
                signals_new3,re_format_3,units_3,adc_gain_3,baseline_3,lead_names_3 = select_lead(signals_new,re_format,units,adc_gain,baseline,lead_names,lead_idx=frag[2])
                write_annotated_wfdb(file.replace('.hea', '')+'_p'+str(frag_count), np.array(signals_new3).T, fs, re_format_3, units_3, adc_gain_3,
                                     baseline_3, coments, lead_names_3,
                                     fix_anot_pos(new_pos), new_qrs, new_events, p=storing_path)
            else:
                write_annotated_wfdb(file.replace('.hea', '') + '_p' + str(frag_count), np.array(signals_new).T, fs,
                                     re_format, units, adc_gain,
                                     baseline, coments, lead_names,
                                     fix_anot_pos(new_pos), new_qrs, new_events, p=storing_path)







