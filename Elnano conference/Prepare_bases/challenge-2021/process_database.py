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

def remove_nan_annot(annotations_pos, qrs_annotations, ecg_events):
    new_qrs_pos = []
    new_qrs_an = []
    new_events = []
    for pos,qrs_an,event in zip(annotations_pos,qrs_annotations,ecg_events):
        if event == 'None':
            pass
        else:
            new_qrs_pos.append(pos)
            new_qrs_an.append(qrs_an)
            new_events.append(event)

    return new_qrs_pos,new_qrs_an,new_events


db_source1 = 'E:/Bases/paroxysmal-atrial-fibrillation-events-detection-from-dynamic-ecg-recordings-the-4th-china-physiological-signal-challenge-2021-1.0.0/Training_set_I/'
db_source2 = 'E:/Bases/paroxysmal-atrial-fibrillation-events-detection-from-dynamic-ecg-recordings-the-4th-china-physiological-signal-challenge-2021-1.0.0/Training_set_II/'
storing_path = 'E:/Bases/0Prepared_bases/2021_challange_prepared/'

sources_mass = [db_source2,db_source1]
count_source = 0
for db_source in sources_mass:
    count_source = count_source+1

    files = FILES_processing_lib.scandir(db_source,ext='.hea')

    total_file_num = 0
    for file in files:
        total_file_num = total_file_num+1
        print(f'{total_file_num}/{len(files)}')
        print(file)
        signals, re_format, fs, units, adc_gain, baseline, \
        coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb(db_source+file, ann = 'atr')

        annotations_pos, qrs_annotations, ecg_events = remove_nan_annot(annotations_pos, qrs_annotations, ecg_events)


        write_annotated_wfdb(file.replace('.hea','')+f"pp_{count_source}",np.array(signals).T,fs,re_format,units,adc_gain,baseline,coments,lead_names,
                        fix_anot_pos(annotations_pos),qrs_annotations,ecg_events,p=storing_path)





