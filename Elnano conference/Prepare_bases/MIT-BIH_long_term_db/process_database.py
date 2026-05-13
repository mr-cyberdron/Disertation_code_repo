import FILES_processing_lib
import numpy as np
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb

#-------------------Add noise detection need to add--------------#

def fix_anot_pos(old_anot):
    new_anot = []
    for anot_it in old_anot:
        if anot_it>=0:
            new_anot.append(anot_it)
        else:
            new_anot.append(0)
    return np.array(new_anot)

db_source = 'E:/Bases/mit-bih-long-term-ecg-database-1.0.0/'
storing_path = 'E:/Bases/0Prepared_bases/mit-bih-long-term-ecg-dat_prep/'

files = FILES_processing_lib.scandir(db_source,ext='.hea')

total_file_num = 0
for file in files:
    total_file_num = total_file_num+1
    print(f'{total_file_num}/{len(files)}')
    print(file)
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'atr')



    write_annotated_wfdb(file.replace('.hea',''),np.array(signals).T,fs,None,units,None,None,coments,lead_names,
                    fix_anot_pos(annotations_pos),qrs_annotations,ecg_events,p=storing_path)





