import numpy as np
import pandas as pd
from ClassificationScheme import arrhythmias_group
from ClassificationScheme import arrhythmia_warnings_group
from ClassificationScheme import arrhythmias_group_names
from ClassificationScheme import arrhythmia_warnings_group_names


dbPath = "D:/Bases/PTB XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
# db_metadata = dbPath+'ptbxl_database.csv'
db_metadata = dbPath+'metadata_normalised_LP_atached.csv'

metadata_db = pd.read_csv(db_metadata)
metadata_db_validated = metadata_db[metadata_db['validated_by_human'].astype(str).str.contains("True", na=False)]

# metadata_db = metadata_db_validated

print(f'Records number: {len(metadata_db)}')
print(f'Records validated number: {len(metadata_db_validated)}')

all_path_groups_data = arrhythmias_group+arrhythmia_warnings_group
all_path_groups_names = arrhythmias_group_names+arrhythmia_warnings_group_names

scp_codes_mass = []
groups_sizes = []
for group, group_name in zip(all_path_groups_data, all_path_groups_names):
    group_scp_codes = [list(group_code.keys())[0] for group_code in group]
    group_size = 0
    for scp_code in group_scp_codes:
        scp_code_db_slice = metadata_db[metadata_db['scp_codes'].str.contains("'" + scp_code + "'", na=False,  regex=False)]
        code_size = len(scp_code_db_slice)
        group_size = group_size+ code_size
        print(f'code: {scp_code}, size: {code_size}')
        scp_codes_mass.append(scp_code)
    print(f'group name: {group_name}, size: {group_size}')
    groups_sizes.append(group_size)

print(np.mean((groups_sizes)))
print(np.sum(groups_sizes))
print(groups_sizes)
print(scp_codes_mass)

test_slice_conf = {'NORM':200,'LAP':120,'LVP':120,'PAC':70, 'AFIB':120, 'AFLT':None, 'SARRH':120, 'STACH':100,
 'SBRAD':60, 'SVARR':70, 'SVTAC':20, 'PSVT':20, 'BIGU':20, 'TRIGU':None, 'PVC':110,
 '1AVB':100, '2AVB':None, '3AVB':None, 'CRBBB':60, 'CLBBB':70, 'IRBBB':100,
 'ILBBB':None, 'LAFB':130, 'LPFB':25, 'WPW':25, 'LPR':30, 'PRC(S)':None,
 'LVH':150, 'RVH':20, 'SEHYP':None, 'VCLVH':75, 'LAO/LAE':60, 'RAO/RAE':30,
 'ISC_':100, 'ISCAL':70, 'ISCIN':30, 'ISCIL':30, 'ISCAS':30, 'ISCLA':None, 'ISCAN':None,
 'IMI':200, 'ASMI':160, 'AMI':70, 'ALMI':50, 'ILMI':70, 'IPLMI':40, 'IPMI':20, 'LMI':25, 'PMI':None,
 'LNGQT':50, 'ANEUR':20}


