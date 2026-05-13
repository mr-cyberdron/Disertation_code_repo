arrhythmias = [
    {"PAC": "Передсердна екстрасистолія (Atrial premature complex)"},
    {"AFIB": "Фібриляція передсердь (Atrial fibrillation)"},
    {"AFLT": "Тріпотіння передсердь (Atrial flutter)"},
    {"SARRH": "Синусова аритмія (Sinus arrhythmia)"},
    {"STACH": "Синусова тахікардія (Sinus tachycardia)"},
    {"SBRAD": "Синусова брадикардія (Sinus bradycardia)"},
    {"SVARR": "Надшлуночкова аритмія (Supraventricular arrhythmia)"},
    {"SVTAC": "Надшлуночкова тахікардія (Supraventricular tachycardia)"},
    {"PSVT": "Пароксизмальна надшлуночкова тахікардія (Paroxysmal supraventricular tachycardia)"},
    {"BIGU": "Бігемінія, невідомого походження (Bigeminal pattern)"},
    {"TRIGU": "Тригемінія, невідомого походження (Trigeminal pattern)"},
    {"PVC": "Шлуночкова екстрасистолія (Ventricular premature complex)"},
    {"1AVB": "Блокада AV-вузла першого ступеня (First degree AV block)"},
    {"2AVB": "Блокада AV-вузла другого ступеня (Second degree AV block)"},
    {"3AVB": "Блокада AV-вузла третього ступеня (Third degree AV block)"},
    {"CRBBB": "Повна блокада правої ніжки пучка Гіса (Complete right bundle branch block)"},
    {"CLBBB": "Повна блокада лівої ніжки пучка Гіса (Complete left bundle branch block)"},
    {"IRBBB": "Неповна блокада правої ніжки пучка Гіса (Incomplete right bundle branch block)"},
    {"ILBBB": "Неповна блокада лівої ніжки пучка Гіса (Incomplete left bundle branch block)"},
    {"LAFB": "Блокада лівої передньої гілки (Left anterior fascicular block)"},
    {"LPFB": "Блокада лівої задньої гілки (Left posterior fascicular block)"},
    {"WPW": "Синдром Вольфа-Паркінсона-Уайта (Wolf-Parkinson-White syndrome)"},
    {"LPR": "Подовжений PR інтервал (Prolonged PR interval)"},
    {"PRC(S)": "Передчасні комплекси (Premature complex(es))"}
]

arrhythmia_warnings =[
    {"LVH": "Гіпертрофія лівого шлуночка (Left ventricular hypertrophy)"},
    {"RVH": "Гіпертрофія правого шлуночка (Right ventricular hypertrophy)"},
    {"SEHYP": "Гіпертрофія міжшлуночкової перегородки (Septal hypertrophy)"},
    {"VCLVH": "Вольтажні критерії гіпертрофії лівого шлуночка (Voltage criteria for left ventricular hypertrophy)"},
    {"LAO/LAE": "Перевантаження/збільшення лівого передсердя (Left atrial overload/enlargement)"},
    {"RAO/RAE": "Перевантаження/збільшення правого передсердя (Right atrial overload/enlargement)"},
    {"ISC_": "Неспецифічна ішемія (Non-specific ischemic)"},
    {"ISCAL": "Ішемія в антеролатеральних відведеннях (Ischemic in anterolateral leads)"},
    {"ISCIN": "Ішемія в нижніх відведеннях (Ischemic in inferior leads)"},
    {"ISCIL": "Ішемія в інферолатеральних відведеннях (Ischemic in inferolateral leads)"},
    {"ISCAS": "Ішемія в антеросептальних відведеннях (Ischemic in anteroseptal leads)"},
    {"ISCLA": "Ішемія в латеральних відведеннях (Ischemic in lateral leads)"},
    {"ISCAN": "Ішемія в передніх відведеннях (Ischemic in anterior leads)"},
    {"IMI": "Нижній інфаркт міокарда (Inferior myocardial infarction)"},
    {"ASMI": "Антеросептальний інфаркт міокарда (Anteroseptal myocardial infarction)"},
    {"AMI": "Передній інфаркт міокарда (Anterior myocardial infarction)"},
    {"ALMI": "Антеролатеральний інфаркт міокарда (Anterolateral myocardial infarction)"},
    {"ILMI": "Інферолатеральний інфаркт міокарда (Inferolateral myocardial infarction)"},
    {"IPLMI": "Інферопостеролатеральний інфаркт міокарда (Inferoposterolateral myocardial infarction)"},
    {"IPMI": "Інферопостеріорний інфаркт міокарда (Inferoposterior myocardial infarction)"},
    {"LMI": "Латеральний інфаркт міокарда (Lateral myocardial infarction)"},
    {"PMI": "Задній інфаркт міокарда (Posterior myocardial infarction)"},
    {"LNGQT": "Подовження QT-інтервалу (Long QT-interval)"},
    {"ANEUR": "Зміни ST-T, пов'язані з аневризмою шлуночка (ST-T changes compatible with ventricular aneurysm)"}
]

normal_ecg = [
    {'NORM': 'ЕКГ в межах норми'}
]

# Аритмії
atrial_arrhythmias = [arrhythmias[0], arrhythmias[1], arrhythmias[2]]  # PAC, AFIB, AFLT
sinus_arrhythmias = [arrhythmias[3], arrhythmias[4], arrhythmias[5]]  # SARRH, STACH, SBRAD
supraventricular_arrhythmias = [arrhythmias[6], arrhythmias[7], arrhythmias[8],arrhythmias[9],arrhythmias[10]]  # SVARR, SVTAC, PSVT
ventricular_arrhythmias = [arrhythmias[11]]  # PVC
heart_blocks = [arrhythmias[12], arrhythmias[13], arrhythmias[14], arrhythmias[15], arrhythmias[16], arrhythmias[17], arrhythmias[18], arrhythmias[19],arrhythmias[20]]  # 1AVB, 2AVB, 3AVB, CRBBB, CLBBB, IRBBB, ILBBB, LAFB, LPFB
premature_complexes = [arrhythmias[21],arrhythmias[22],arrhythmias[23]]  # WPW, LPR, PRC(S)

# Патології, пов’язані з аритміями
ventricular_hypertrophy = [arrhythmia_warnings[0], arrhythmia_warnings[1], arrhythmia_warnings[2], arrhythmia_warnings[3]]  # LVH, RVH, SEHYP, VCLVH
atrial_enlargement = [arrhythmia_warnings[4], arrhythmia_warnings[5]]  # LAO/LAE, RAO/RAE
ischemia = arrhythmia_warnings[6:13]  # ISC_, ISCAL, ISCIN, ..., PMI
infarction = arrhythmia_warnings[13:22]
long_qt = [arrhythmia_warnings[22]]  # LNGQT,  ANEUR, ABQRS
ventricular_st_changes =[arrhythmia_warnings[23]]

# ЕКГ в межах норми
ecg_norm = [normal_ecg[0]]

arrhythmias_group = [atrial_arrhythmias,
                     sinus_arrhythmias,
                     supraventricular_arrhythmias,
                     ventricular_arrhythmias,
                     heart_blocks,
                     premature_complexes]

arrhythmia_warnings_group = [ventricular_hypertrophy,
                             atrial_enlargement,
                             ischemia,
                             infarction,
                             long_qt,
                             ventricular_st_changes]

normal_ecg_group = [ecg_norm]

arrhythmias_group_names = ["atrial_arrhythmias",
                     "sinus_arrhythmias",
                     "supraventricular_arrhythmias",
                     "ventricular_arrhythmias",
                     "heart_blocks",
                     "premature_complexes"]

arrhythmia_warnings_group_names = ["ventricular_hypertrophy",
                             "atrial_enlargement",
                             "ischemia",
                             "infarction",
                             "long_qt",
                             "ventricular_st_changes"]

normal_ecg_group_names = [normal_ecg_group]


data_normalisation_scheme = {
    #supraventricular_arrhythmias
    'SVARR':{'mode':'expand', 'degree':2}, #157
    'SVTAC':{'mode':'expand', 'degree':4},# 27
    'PSVT':{'mode':'expand', 'degree':4}, #24
    'BIGU':{'mode':'expand', 'degree':2},#82
    'TRIGU':{'mode':'expand', 'degree':4},#20

    #heart_blocks
    'IRBBB':{'mode':'reduce', 'degree':None}, #1118
'LAFB':{'mode':'reduce', 'degree':None}, # 1623
'2AVB':{'mode':'expand', 'degree':4}, #14
'3AVB':{'mode':'expand', 'degree':4}, #16

    #premature_complexes
'WPW':{'mode':'expand', 'degree':2}, #79
'PRC(S)':{'mode':'expand', 'degree':5}, #10

    #ventricular_hypertrophy
'LVH':{'mode':'reduce', 'degree':None}, #2132
'SEHYP':{'mode':'expand', 'degree':3}, #29

    #atrial_enlargement
'RAO/RAE':{'mode':'expand', 'degree':2}, #99

    #ischemia
'ISCAN':{'mode':'expand', 'degree':2}, #44

    #infarction
'IMI':{'mode':'reduce', 'degree':None}, #2676
'ASMI':{'mode':'reduce', 'degree':None}, #2357
'PMI':{'mode':'expand', 'degree':5}, #17
'IPMI':{'mode':'expand', 'degree':3}, #33
'IPLMI':{'mode':'expand', 'degree':3}, #51

    #long_qt
'LNGQT':{'mode':'expand', 'degree':2}, #117

    #ventricular_st_changes
'ANEUR':{'mode':'expand', 'degree':2}, #104


}




