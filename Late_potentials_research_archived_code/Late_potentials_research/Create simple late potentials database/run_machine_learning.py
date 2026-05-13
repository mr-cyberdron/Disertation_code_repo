import pandas as pd
import numpy as np
import copy
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from numpy import reshape
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import RobustScaler
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def clasif_flow(clasif_prepared_db, label):
    target = (clasif_prepared_db['class'].to_numpy())
    data = (clasif_prepared_db.drop(columns='class').to_numpy())

    data, target = scaling_training_data(data, target)
    data, target = remove_outliers(data, target)
    # tsne_plot(data,target)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # -------------------cross validation------------------#
    # Create a logistic regression model
    clf = SVC(kernel='linear')
    clf = RandomForestClassifier()
    clf = LogisticRegression()
    # clf = KNeighborsClassifier(n_neighbors=10)
    # Compute the cross-validation scores
    scores = cross_val_score(clf, data, target, cv=5)
    # Print the cross-validation scores
    print("Cross-validation scores: {}".format(scores))
    # Compute the mean cross-validation score
    mean_score = np.mean(scores)
    # Print the mean cross-validation score
    print("Mean cross-validation score: {:.2f}".format(mean_score))

    # --------------------svm-------------------------#
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrixx(y_test, y_pred, label='-----------------------------SVM' + label + '--------------------------')
    # ----------------------Random forest---------------#
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrixx(y_test, y_pred,
                      label='-----------------------------Random forest' + label + '--------------------------')
    # ----------------------Logistic regression---------#
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrixx(y_test, y_pred,
                      label='-----------------------------Logistic regression' + label + '--------------------------')
    # ----------------------Kneibours-------------------#
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrixx(y_test, y_pred,
                      label='-----------------------------Kneibours' + label + '--------------------------')


def confusion_matrixx(y_test, y_pred, label='', plot=False):
    cm = confusion_matrix(y_test, y_pred)
    print(label)
    print("Confusion matrix:")
    print(cm)
    # Compute the sensitivity, specificity, and accuracy
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr = fp/(fp+tn)
    fnr = fn/(fn+tp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    proc_100 = tn+fp+fn+tp
    tn_proc = (tn/proc_100)*100
    fp_proc = (fp/proc_100)*100
    fn_proc = (fn/proc_100)*100
    tp_proc = (tp/proc_100)*100
    # Print the sensitivity, specificity, and accuracy
    print('tn: {:.2f}%'.format(tn_proc))
    print('fp: {:.2f}%'.format(fp_proc))
    print('fn: {:.2f}%'.format(fn_proc))
    print('tp: {:.2f}%'.format(tp_proc))
    print("Sensitivity (true positive rate): {:.2f}".format(sensitivity))
    print("Specificity (true negative rate): {:.2f}".format(specificity))
    print('FPR: {:.2f}'.format(fpr))
    print('FNR: {:.2f}'.format(fnr))
    print("Accuracy: {:.2f}".format(accuracy))

    if plot:
        # Define the labels for the plot
        labels = ['Negative', 'Positive']

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap='Blues')
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        # Show all ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))

        # Label the ticks
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # Set font size for labels
        ax.tick_params(axis='both', labelsize=12)

        # Set the axis labels
        ax.set_xlabel('Predicted label', fontsize=14)
        ax.set_ylabel('True label', fontsize=14)

        # Add the values in the cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, cm[i, j], ha='center', va='center', color='w', fontsize=16)

        # Set the title
        ax.set_title('Confusion matrix', fontsize=16)

        # Display the plot
        plt.show()



def scaling_training_data(data, target):
    if True:
        scaler = RobustScaler()
        # scaler = StandardScaler() +
        data = scaler.fit_transform(data)
    return data, target

def tsne_plot(data, target):
    target = np.array(target)
    tsne = TSNE(n_components=3, verbose=1, random_state=None)
    data_tsne = tsne.fit_transform(np.array(data))
    tsne_normal = data_tsne[np.where(target == 0)]
    tsne_pat = data_tsne[np.where(target == 1)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter1 = ax.scatter(tsne_normal[:, 0], tsne_normal[:, 1], tsne_normal[:, 2], c='gray', alpha=0.8, label = 'Normal')
    scatter2 = ax.scatter(tsne_pat[:, 0], tsne_pat[:, 1], tsne_pat[:, 2], c='red', alpha=0.8, label = 'Patology')
    # Set the axis labels
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    ax.set_title('TSNE')
    plt.legend(loc="upper right")
    plt.show()

def remove_outliers(data, target):
    target = np.array(target)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    clf.fit(data)
    outliers = clf.predict(data)
    clean_data = data[outliers != -1]
    clean_targeg = target[outliers != -1]
    return clean_data,clean_targeg

def code_targets(df, tag, code):
    dff = copy.deepcopy(df)
    dff.loc[dff['class'] == tag, 'class'] = code
    return dff

def clean_df(df):
    dfff = copy.deepcopy(df)
    #remove_file_paths
    dfff = dfff.drop(columns='file')
    #remove nan rows
    full_empty_row_mass = []
    for index, row in df.iterrows():
        empty_row_check = list(row.isnull().to_numpy())
        empty_val_number = empty_row_check.count(True)
        if empty_val_number >2:
            full_empty_row_mass.append(index)
    dfff = dfff.drop(full_empty_row_mass)
    # replace nan by median
    columns_name = list(dfff.columns)
    for column in columns_name:
        column_median = dfff[column].median()
        dfff[column] = dfff[column].fillna(column_median)

    #replace outliers by median
    if True:
        median = dfff.median()
        std = dfff.std()
        lower_bound = median - (3 * std)
        upper_bound = median + (3 * std)
        for column in dfff.columns:
            dfff[column] = dfff[column].clip(lower_bound[column], upper_bound[column])
            dfff[column] = dfff[column].fillna(median[column])

    return dfff

def make_clasif_db(norm_df, pat_df, norm_code = 0, pat_code = 1):
    norm_tag = 'Normal'
    pat_tag = 'Patology'
    norm_df_coded = code_targets(norm_df,norm_tag, norm_code)
    pat_df_coded = code_targets(pat_df,pat_tag, pat_code)
    norm_df_coded_cleaned = clean_df(norm_df_coded)
    pat_df_coded_cleaned = clean_df(pat_df_coded)

    #classes_equalisation
    if True:
        pat_len = len(pat_df_coded_cleaned)
        norm_df_coded_cleaned = norm_df_coded_cleaned.sample(n = pat_len, random_state=None)
    clasif_df = pd.concat([norm_df_coded_cleaned, pat_df_coded_cleaned], ignore_index=True).reset_index(drop=True)

    print(len(norm_df_coded_cleaned))
    print(len(pat_df_coded_cleaned))
    # input(len(clasif_df))
    return clasif_df


#-----------------LVP_AVG-----------------
# norm_df = pd.read_csv('features_LVP_avg_avg_Normal__.csv')
# pat_df = pd.read_csv('features_LVP_avg_avg_Patology__.csv')
# label = '_LVP_avg_avg'
# clasif_prepared_db = make_clasif_db(norm_df, pat_df, norm_code=0, pat_code=1)
# clasif_flow(clasif_prepared_db,label)
# #-----------------LVP_custom______________
# norm_df = pd.read_csv('features_LVP_avg_custom_Normal__.csv')
# pat_df = pd.read_csv('features_LVP_avg_custom_Patology__.csv')
# label = '_LVP_avg_custom'
# clasif_prepared_db = make_clasif_db(norm_df, pat_df, norm_code=0, pat_code=1)
# clasif_flow(clasif_prepared_db,label)
#-----------------LAP_AVG-----------------
# norm_df = pd.read_csv('features_LAP_avg_avg_Normal__.csv')
# pat_df = pd.read_csv('features_LAP_avg_avg_Patology__.csv')
# label = '_LAP_avg_avg'
# clasif_prepared_db = make_clasif_db(norm_df, pat_df, norm_code=0, pat_code=1)
# clasif_flow(clasif_prepared_db,label)
#-----------------LAP_custom______________
norm_df = pd.read_csv('features_LAP_avg_custom_Normal__.csv')
pat_df = pd.read_csv('features_LAP_avg_custom_Patology__.csv')
label = '_LAP_avg_custom'
clasif_prepared_db = make_clasif_db(norm_df, pat_df, norm_code=0, pat_code=1)
clasif_flow(clasif_prepared_db,label)


