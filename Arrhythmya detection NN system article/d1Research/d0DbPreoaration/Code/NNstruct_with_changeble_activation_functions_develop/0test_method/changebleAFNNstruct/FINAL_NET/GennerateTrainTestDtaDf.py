import copy

import numpy as np
import pandas as pd

transformed_metadata_path = './FinalNetData/TransformedData/metadata.csv'
metadata_df = pd.read_csv(transformed_metadata_path)
train_df = copy.deepcopy(metadata_df)

train_df_norm_slice = train_df[train_df['pathologies'].str.contains('NORM', case=False, na=False)].drop(columns=['Unnamed: 0'], errors='ignore')
norm_rows = train_df_norm_slice.sample(n=6500, random_state=42)
train_df = train_df.drop(norm_rows.index)

train_df = train_df[~train_df['pathologies'].str.contains('AFLT', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('TRIGU', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('2AVB', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('3AVB', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('ILBBB', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('PRC(S)', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('SEHYP', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('ISCLA', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('ISCAN', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('PMI', case=False, na=False)]
train_df = train_df[~train_df['pathologies'].str.contains('IPMI', case=False, na=False)]
test_df = pd.DataFrame(columns=train_df.columns)


# test_slice_conf = {'NORM':200,'LAP':120,'LVP':120,'PAC':70, 'AFIB':120, 'AFLT':None, 'SARRH':120, 'STACH':100,
#  'SBRAD':60, 'SVARR':70, 'SVTAC':20, 'PSVT':20, 'BIGU':20, 'TRIGU':None, 'PVC':110,
#  '1AVB':100, '2AVB':None, '3AVB':None, 'CRBBB':60, 'CLBBB':70, 'IRBBB':100,
#  'ILBBB':None, 'LAFB':130, 'LPFB':25, 'WPW':25, 'LPR':30, 'PRC(S)':None,
#  'LVH':150, 'RVH':20, 'SEHYP':None, 'VCLVH':75, 'LAO/LAE':60, 'RAO/RAE':30,
#  'ISC_':100, 'ISCAL':70, 'ISCIN':30, 'ISCIL':30, 'ISCAS':30, 'ISCLA':None, 'ISCAN':None,
#  'IMI':200, 'ASMI':160, 'AMI':70, 'ALMI':50, 'ILMI':70, 'IPLMI':40, 'IPMI':20, 'LMI':25, 'PMI':None,
#  'LNGQT':50, 'ANEUR':20}

test_slice_conf_sorted = {
 'SVTAC': 20,
 'PSVT': 20,
 'BIGU': 20,
 'RVH': 20,
 'ANEUR': 20,
'LPFB': 25,
 'WPW': 25,
 'LMI': 25,
 'LPR': 30,
 'RAO/RAE': 30,
 'ISCIN': 30,
 'ISCIL': 30,
 'ISCAS': 30,
 'IPLMI': 40,
 'ALMI': 50,
 'LNGQT': 50,
 'SBRAD': 60,
 'CRBBB': 60,
 'LAO/LAE': 60,
 'PAC': 70,
 'SVARR': 70,
 'CLBBB': 70,
 'ISCAL': 70,
 'AMI': 70,
 'ILMI': 70,
 'VCLVH': 75,
 'STACH': 100,
 '1AVB': 100,
 'IRBBB': 100,
 'ISC_': 100,
 'PVC': 110,
 'LAP': 120,
 'LVP': 120,
 'AFIB': 120,
 'SARRH': 120,
 'LAFB': 130,
 'LVH': 150,
 'ASMI': 160,
 'NORM': 200,
 'IMI': 200
}

Train_data_sizes = {}

for pat_key, pat_test_size in zip(list(test_slice_conf_sorted.keys()),list(test_slice_conf_sorted.values())):
    key_df = train_df[train_df['pathologies'].str.contains(pat_key, case=False, na=False)].drop(columns=['Unnamed: 0'], errors='ignore')
    TestRows = key_df.sample(n=pat_test_size, random_state=42)
    train_df = train_df.drop(TestRows.index)
    key_left = len(train_df[train_df['pathologies'].str.contains(pat_key, case=False, na=False)].drop(columns=['Unnamed: 0'],
                                                                                                errors='ignore'))
    Train_data_sizes[pat_key] = key_left
    test_df = pd.concat([test_df,TestRows], ignore_index=True)

print(Train_data_sizes)
print(np.shape(train_df))
print(np.shape(test_df))

train_df.to_csv('train_data_slices.csv')
test_df.to_csv('test_data_slices.csv')
