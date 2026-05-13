import FILES_processing_lib
import numpy as np
from process_peculiarities import peculiarities
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb

def select_lead(signals,re_format,units,adc_gain,baseline,lead_names, lead_idx = 0):
    signals = [signals[lead_idx]]
    re_format = [re_format[lead_idx]]
    units = [units[lead_idx]]
    adc_gain = [adc_gain[lead_idx]]
    baseline = [baseline[lead_idx]]
    lead_names = [lead_names[lead_idx]]
    return signals,re_format,units,adc_gain,baseline,lead_names

def remove_leads(signals,re_format,units,adc_gain,baseline,lead_names, lead_idx = 0):
    lead_indexes = list(range(len(signals)))
    lead_indexes.remove(lead_idx)
    signals = signals[lead_indexes]
    re_format = np.array(re_format)[lead_indexes]
    units = np.array(units)[lead_indexes]
    adc_gain = np.array(adc_gain)[lead_indexes]
    baseline = np.array(baseline)[lead_indexes]
    lead_names = np.array(lead_names)[lead_indexes]
    return signals, list(re_format), list(units), list(adc_gain), list(baseline), list(lead_names)





def fix_signames(signames_mas):
    seen = {}
    for i, element in enumerate(signames_mas):
        if element in seen:
            signames_mas[i] = f"{element}{seen[element]}"
            seen[element] += 1
        else:
            seen[element] = 1
    return signames_mas

def fix_anot_pos(old_anot):
    new_anot = []
    for anot_it in old_anot:
        if anot_it>=0:
            new_anot.append(anot_it)
        else:
            new_anot.append(0)
    return np.array(new_anot)


db_source = 'E:/Bases/t-wave-alternans-challenge-database-1.0.0/'
storing_path = 'E:/Bases/0Prepared_bases/t-wave-alternans-challenge-database_prepared/'

files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    print(file)
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'qrs')

    save_flag = True
    files_with_pecularities = list(peculiarities.keys())
    if file in files_with_pecularities:
        for cond in peculiarities[file]:
            if cond == 'skip_rec':
                save_flag = False
            if cond == 'skip_lead_0':
                signals, re_format, units, adc_gain, baseline, lead_names = remove_leads(
                    signals,re_format,units,adc_gain,baseline,lead_names,lead_idx=0)
            if cond == 'skip_lead_2':
                signals, re_format, units, adc_gain, baseline, lead_names = remove_leads(
                    signals,re_format,units,adc_gain,baseline,lead_names,lead_idx=2)
            if cond == 'skip_lead_4':
                signals, re_format, units, adc_gain, baseline, lead_names = remove_leads(
                    signals,re_format,units,adc_gain,baseline,lead_names,lead_idx=4)



    if save_flag:
        write_annotated_wfdb(file.replace('.hea',''),np.array(signals).T,fs,re_format,units,adc_gain,baseline,coments,lead_names,
                        fix_anot_pos(annotations_pos),qrs_annotations,ecg_events,p=storing_path)




