import FILES_processing_lib
import numpy as np
import wfdb
from Dataformats_tools.WFDB.WFDB_RW import read_wfdb, write_annotated_wfdb,read_wfdb_no_ann
from Dataformats_tools.EDF import save_edf
import pandas as pd

def get_files_list(db_dir):
    files_list =list(pd.read_csv(db_dir+'RECORDS', header=None).to_numpy())
    ecg_files = []
    for fp in files_list:
        if fp[0].endswith("_ECG"):
            ecg_files.append(fp[0])
    return ecg_files

def prepare_ann_from_csv(df_path):
    annot_df_db = pd.read_csv(df_path, header=None)
    # anot_group = annot_df_db.iloc[:,0:3]
    # anot_group = annot_df_db.iloc[:, 3:6] #ok
    #anot_group = annot_df_db.iloc[:, 6:9]
    anot_group = annot_df_db.iloc[:, 9:12]
    anot_group = anot_group.dropna()
    annot_pos_mass_1 = list(anot_group.iloc[:,0].to_numpy().astype(int))
    annot_pos_mass_2 = list(anot_group.iloc[:, 1].to_numpy().astype(int))
    annot_mass = list(anot_group.iloc[:,2].to_numpy().astype(int))

    new_annotations_pos = []
    new_qrs_annotations = []
    new_ecg_events = []

    for anot_pos_1,anot_pos_2,event in zip(annot_pos_mass_1,annot_pos_mass_2,annot_mass):
        new_annotations_pos.append(anot_pos_1)
        new_annotations_pos.append(anot_pos_2)
        new_ecg_events.append(str(event)+'_start')
        new_ecg_events.append(str(event) + '_stop')
        new_qrs_annotations.append('~')
        new_qrs_annotations.append('~')
    assert (len(new_annotations_pos) == len(new_ecg_events))
    return np.array(new_annotations_pos),np.array(new_qrs_annotations), np.array(new_ecg_events)

def save_frags(signals,fs, df_path,lead_names,p = './', n = '',p2 = './'):
    annot_df_db = pd.read_csv(df_path, header=None)
    # anot_group = annot_df_db.iloc[:,0:3]
    # anot_group = annot_df_db.iloc[:, 3:6] #ok
    # anot_group = annot_df_db.iloc[:, 6:9]
    anot_group = annot_df_db.iloc[:, 9:12]
    anot_group = anot_group.dropna()
    annot_pos_mass_1 = list(anot_group.iloc[:, 0].to_numpy().astype(int))
    annot_pos_mass_2 = list(anot_group.iloc[:, 1].to_numpy().astype(int))
    annot_mass = list(anot_group.iloc[:, 2].to_numpy().astype(int))

    new_annotations_pos = []
    new_qrs_annotations = []
    new_ecg_events = []

    fragm_counter = 0
    for anot_pos_1, anot_pos_2, event in zip(annot_pos_mass_1, annot_pos_mass_2, annot_mass):
        fragm_counter = fragm_counter+1
        print(f'fragm{fragm_counter}')

        try:
            save_edf((signals[anot_pos_1:anot_pos_2,:]).T,fs,chanelnamesmas=['ECG'],dimension='mV',p=f'{p}{n}_{fragm_counter}_type_{event}.edf')
        except:
            pass
        if event == 3:
            write_annotated_wfdb(f'{n}_{fragm_counter}_type_{event}', (signals[anot_pos_1:anot_pos_2,:]/1000), fs, None, ['mV'],
                                 None, None, None, lead_names,
                                 np.array([0]), np.array(['~']), np.array(['U']), p2)

        # wfdb.wrsamp(record_name=f'{n}_type_{fragm_counter}', p_signal=signals, fs=fs,
        #             units=['mV'],comments=[str(event)],
        #             sig_name=['ECG'], write_dir=p)


but_qdb_dir = 'E:/Bases/' \
              'brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0/'

out_dir = 'E:/Bases/0Prepared_bases/Noise/BUT_QTB/'
frags_dir = 'E:/Bases/0Prepared_bases/Noise/BUT_QTB/frags_types/'
noise_dir = 'E:/Bases/0Prepared_bases/Noise/BUT_QTB/noise_rec/'

ecg_list = get_files_list(but_qdb_dir)

counrer =0
for ecg_file in ecg_list:
    counrer = counrer+1
    print(f'{counrer}/{len(ecg_list)}')
    full_ecg_path = but_qdb_dir+ecg_file
    signals, re_format, fs, units, adc_gain, baseline, \
    coments, lead_names, annotations_pos, qrs_annotations, ecg_events = read_wfdb_no_ann(full_ecg_path)

    new_annotations_pos,new_qrs_anot, new_ecg_events = prepare_ann_from_csv(full_ecg_path.replace('_ECG','_ANN')+'.csv')
    save_frags((signals/1000).T,fs, full_ecg_path.replace('_ECG','_ANN')+'.csv',lead_names,p = frags_dir, n = ecg_file.split('/')[-1], p2=noise_dir)


    write_annotated_wfdb(ecg_file.split('/')[-1].replace('.hea',''),(signals/1000).T,fs,None,['mV'],None,None,coments,lead_names,
                         new_annotations_pos,new_qrs_anot,new_ecg_events, out_dir)




