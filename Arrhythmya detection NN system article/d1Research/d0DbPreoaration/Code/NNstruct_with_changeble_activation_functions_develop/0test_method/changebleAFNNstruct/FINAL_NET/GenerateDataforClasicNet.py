import numpy as np
import pandas as pd
import ast

train_df = pd.read_csv('train_data_slices.csv')
test_df = pd.read_csv('test_data_slices.csv')

Y_ref_dict = {'SVTAC': 0, 'PSVT': 0, 'BIGU': 0, 'RVH': 0, 'ANEUR': 0, 'LPFB': 0, 'WPW': 0, 'LMI': 0,
                      'LPR': 0, 'RAO/RAE': 0, 'ISCIN': 0, 'ISCIL': 0, 'ISCAS': 0, 'IPLMI': 0, 'ALMI': 0, 'LNGQT': 0,
                      'SBRAD': 0, 'CRBBB': 0, 'LAO/LAE': 0, 'PAC': 0, 'SVARR': 0, 'CLBBB': 0, 'ISCAL': 0, 'AMI': 0,
                      'ILMI': 0, 'VCLVH': 0, 'STACH': 0, '1AVB': 0, 'IRBBB': 0, 'ISC_': 0, 'PVC': 0, 'LAP': 0,
                      'LVP': 0, 'AFIB': 0, 'SARRH': 0, 'LAFB': 0, 'LVH': 0, 'ASMI': 0, 'NORM': 0, 'IMI': 0}
#
# dict_enum = {}
# counter = 0
# for i in list(Y_ref_dict):
#     dict_enum[i] = counter
#     counter+=1
# input(dict_enum)

def generate_X_Y_for_train(metadata_Df, n = 'Train'):
    X_mass = []
    Y_mass = []
    for i,row in metadata_Df.iterrows():
        # Y_ref_dict = {'SVTAC': 0, 'PSVT': 0, 'BIGU': 0, 'RVH': 0, 'ANEUR': 0, 'LPFB': 0, 'WPW': 0, 'LMI': 0,
        #               'LPR': 0, 'RAO/RAE': 0, 'ISCIN': 0, 'ISCIL': 0, 'ISCAS': 0, 'IPLMI': 0, 'ALMI': 0, 'LNGQT': 0,
        #               'SBRAD': 0, 'CRBBB': 0, 'LAO/LAE': 0, 'PAC': 0, 'SVARR': 0, 'CLBBB': 0, 'ISCAL': 0, 'AMI': 0,
        #               'ILMI': 0, 'VCLVH': 0, 'STACH': 0, '1AVB': 0, 'IRBBB': 0, 'ISC_': 0, 'PVC': 0, 'LAP': 0,
        #               'LVP': 0, 'AFIB': 0, 'SARRH': 0, 'LAFB': 0, 'LVH': 0, 'ASMI': 0, 'NORM': 0, 'IMI': 0}

        Y_ref_dict = {'SVTAC': 0, 'PSVT': 0, 'BIGU': 0, 'RVH': 0, 'ANEUR': 0, 'LPFB': 0, 'WPW': 0, 'LMI': 0,
                      'LPR': 0, 'RAO/RAE': 0, 'ISCIN': 0, 'ISCIL': 0, 'ISCAS': 0, 'IPLMI': 0, 'ALMI': 0, 'LNGQT': 0,
                      'SBRAD': 0, 'CRBBB': 0, 'LAO/LAE': 0, 'PAC': 0, 'SVARR': 0, 'CLBBB': 0, 'ISCAL': 0, 'AMI': 0,
                      'ILMI': 0, 'VCLVH': 0, 'STACH': 0, '1AVB': 0, 'IRBBB': 0, 'ISC_': 0, 'PVC': 0, 'LAP': 0,
                      'LVP': 0, 'AFIB': 0, 'SARRH': 0, 'LAFB': 0, 'LVH': 0, 'ASMI': 0, 'NORM': 0, 'IMI': 0}

        pathologies = ast.literal_eval(row['pathologies'])
        patkeys = list(pathologies.keys())
        for patkey in patkeys:
            if patkey in list(Y_ref_dict.keys()):
                Y_ref_dict[patkey] = 1

        X_mass.append(np.load(row['fpath']))
        Y_mass.append(list(Y_ref_dict.values()))



    X_mass = np.array(X_mass)
    Y_mass = np.array(Y_mass)
    print(np.shape(X_mass))
    print(np.shape(Y_mass))
    np.save('./FinalNetData/TrainDataForClassicNET/X_'+n+'.npy', X_mass)
    np.save('./FinalNetData/TrainDataForClassicNET/Y_' + n + '.npy', Y_mass)

    # test2data
    #
    # np.save('./FinalNetData/TrainDataForClassicNET/X_' + n + '.npy', X_mass)
    # np.save('./FinalNetData/TrainDataForClassicNET/Y_' + n + '.npy', Y_mass)



generate_X_Y_for_train(train_df, n='Train')
generate_X_Y_for_train(test_df, n='Test')

