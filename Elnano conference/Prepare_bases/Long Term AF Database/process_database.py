import FILES_processing_lib
import numpy as np
from process_peculiarities import peculiarities
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb

#-------------------Add noise detection need to add--------------#
def select_lead(signals,re_format,units,adc_gain,baseline,lead_names, lead_idx = 0):
    signals = [signals[lead_idx]]
    re_format = [re_format[lead_idx]]
    units = [units[lead_idx]]
    adc_gain = [adc_gain[lead_idx]]
    baseline = [baseline[lead_idx]]
    lead_names = [lead_names[lead_idx]]
    return signals,re_format,units,adc_gain,baseline,lead_names

def fix_signames(signames_mas):
    seen = {}
    for i, element in enumerate(signames_mas):
        if element in seen:
            signames_mas[i] = f"{element}{seen[element]}"
            seen[element] += 1
        else:
            seen[element] = 1
    return signames_mas

db_source = 'E:/Bases/arhytmia_episodes_and_beat_classif_base/long-term-af-database-1.0.0/files/'
storing_path = 'E:/Bases/0Prepared_bases/long-term-af-database_prep/'

files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    print(file)
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'atr')

    save_flag = True
    files_with_pecularities = list(peculiarities.keys())
    if file in files_with_pecularities:
        if peculiarities[file][0] == 'only1lead':
            signals, re_format, units, adc_gain, baseline, lead_names = select_lead(
                signals,re_format,units,adc_gain,baseline,lead_names,lead_idx=0)
        if peculiarities[file][0] == 'only2lead':
            signals, re_format, units, adc_gain, baseline, lead_names = select_lead(
                signals,re_format,units,adc_gain,baseline,lead_names,lead_idx=1)
        if peculiarities[file][0] == 'skip_rec':
            save_flag = False

    if save_flag:
        write_annotated_wfdb(file.replace('.hea',''),np.array(signals).T,fs,re_format,units,adc_gain,baseline,coments,fix_signames(lead_names),
                        annotations_pos,qrs_annotations,ecg_events,p=storing_path)



