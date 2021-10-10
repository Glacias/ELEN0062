# ! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
Notes :
-------
This code contains parts taken directly from the toy_example.py provided.

Some parts of the code are in comment blocks. These are parts that we have tried
during the projects but that are not used for the 'final' model that we computed.

The code provided is the one used to compute the model that has been used for 
the private leaderbord on Kaggle.
'''

import os
import time
import datetime
import argparse
from contextlib import contextmanager
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from vecstack import stacking
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools




@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    print("{}...".format(label))
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    if not os.path.exists(path):
        raise FileNotFoundError("File '{}' does not exists.".format(path))
    return pd.read_csv(path, delimiter=delimiter)




def create_fingerprints(chemical_compounds):
    """
    Create a learning matrix `X` with (Morgan) fingerprints
    from the `chemical_compounds` molecular structures.

    Parameters
    ----------
    chemical_compounds: array [n_chem, 1] or list [n_chem,]
        chemical_compounds[i] is a string describing the ith chemical
        compound.

    Return
    ------
    X: array [n_chem, nBits]
        Generated Morgan or Avalon or Daylightfingerprints for each chemical 
        compound, which represent presence or absence of substructures.
    """
    n_chem = chemical_compounds.shape[0]

    nBits = 3200

    '''
    #Morgan
    X = np.zeros((n_chem, nBits))


    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        X[i,:] = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=nBits)
    '''

    
    #Avalon
    X = np.zeros((n_chem, nBits))
    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        X[i,:] = pyAvalonTools.GetAvalonFP(m, nBits=nBits)

    '''
    #Daylight
    X = np.zeros((n_chem, 2048))
    generator = rdFingerprintGenerator.GetRDKitFPGenerator()
    for i in range(n_chem):
        m = Chem.MolFromSmiles(chemical_compounds[i])
        X[i,:] = generator.GetFingerprint(m)
    '''

    return X


def make_submission(y_predicted, auc_predicted, file_name="submission", date=True, indexes=None):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predicted: array [n_predictions, 1]
        if `y_predict[i]` is the prediction
        for chemical compound `i` (or indexes[i] if given).
    auc_predicted: float [1]
        The estimated ROCAUC of y_predicted.
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Creating default indexes if not given
    if indexes is None:
        indexes = np.arange(len(y_predicted))+1

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"Chem_ID","Prediction"\n')
        handle.write('Chem_{:d},{}\n'.format(0,auc_predicted))

        for n,idx in enumerate(indexes):

            if np.isnan(y_predicted[n]):
                raise ValueError('The prediction cannot be NaN')
            line = 'Chem_{:d},{}\n'.format(idx, y_predicted[n])
            handle.write(line)
    return file_name


def auc(y_true, y_pred):
    '''
    Computes the ROC AUC.
    
    Parameters
    ----------
    y_true : True classes
    y_pred : Predicted probabilities

    Return
    ------
    auc
    '''

    auc = roc_auc_score(y_true, y_pred[:,1])
    return auc


def hamming_mat(X1, X2):
    '''
    Computes the hamming mat (or custom) kernel for the SVC.
    '''
    # Common substructures
    my_mat = np.linalg.multi_dot((X1, np.transpose(X2)))
    # Common missing substructures
    '''
    X1_compl = (X1==0).astype(np.integer)
    X2_compl = (X2==0).astype(np.integer)
    my_mat += np.linalg.multi_dot((X1_compl, np.transpose(X2_compl)))
    '''
    return my_mat





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make a toy submission")
    parser.add_argument("--ls", default="data/learning_set.csv",
                        help="Path to the learning set as CSV file")
    parser.add_argument("--ts", default="data/test_set.csv",
                        help="Path to the test set as CSV file")

    args = parser.parse_args()

    # Load training data
    LS = load_from_csv(args.ls)
    # Create fingerprint features and output
    x_LS = create_fingerprints(LS["SMILES"].values)
    y_LS = LS["ACTIVE"].values
    # Load test data
    TS = load_from_csv(args.ts)
    # Create fingerprint features and output
    x_TS = create_fingerprints(TS["SMILES"].values)

    
    #Models with best parameters found
    lr = LogisticRegression(class_weight='balanced', solver='newton-cg', C = 0.001, max_iter=10000, n_jobs=-1)
    gnb = GaussianNB()
    gbc = GradientBoostingClassifier(n_estimators=250)
    sgd = SGDClassifier(loss='log', class_weight='balanced')
    knn = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)
    lda = LinearDiscriminantAnalysis()
    rfc = RandomForestClassifier(class_weight='balanced', n_estimators=400, min_impurity_decrease=0.0005, criterion='entropy')
    xgb = XGBClassifier(learning_rate=0.1, n_estimators=250, max_depth=3, n_jobs=-1)
    svc = SVC(gamma='scale', kernel = hamming_mat, probability=True, class_weight='balanced')
    
    
    '''
    #Block to try different models with different parameters
    models={
        AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight='balanced', n_estimators=350, min_impurity_decrease=0.0005, criterion='entropy'), n_estimators=10),
        AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight='balanced', n_estimators=350, min_impurity_decrease=0.0005, criterion='entropy'), n_estimators=100),
        AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight='balanced', n_estimators=350, min_impurity_decrease=0.0005, criterion='entropy'), n_estimators=300),
        }

    #selector = SelectFromModel(estimator=LogisticRegression(class_weight='balanced', solver='newton-cg', C = 0.01, max_iter=10000), threshold=0.25).fit(x_LS, y_LS)
    #x_LS = selector.transform(x_LS)
    #print('Number of features kept : ' + str(x_LS.shape[1]))


    # Divide the LS to predict the auc
    x_LS_train, x_LS_test, y_LS_train, y_LS_test = train_test_split(x_LS, y_LS, test_size=0.33, random_state=42)

    for model in models:
        model.fit(x_LS_train, y_LS_train)
        y_LS_pred = model.predict_proba(x_LS_test)
        print('\n------------------')
        print(str(model) + '\n\nauc : ' + str(auc(y_LS_test, y_LS_pred)))
    exit()
    '''

  
    # Models used for the stacking
    estimators = [
        #xgb,
        #gbc,
        knn,
        #sgd,
        #lda,
        rfc,
        #lr,
        #gnb
        svc
    ]


    '''
    # Under/over sampling
    X_LS_false = X_LS[:15504]
    y_LS_false = y_LS[:15504]

    X_LS_false, y_LS_false = shuffle(X_LS_false, y_LS_false)

    X_LS_false = X_LS_false[:12000]
    y_LS_false = y_LS_false[:12000]

    X_LS = np.append(X_LS[15505:], X_LS_false, axis=0)
    y_LS =np.append(y_LS[15505:], y_LS_false, axis=0)
    '''



    '''
    # Feature selection
    selector = SelectFromModel(estimator=LogisticRegression(class_weight='balanced', solver='newton-cg', C = 0.01, max_iter=10000), threshold=0.25).fit(x_LS, y_LS)
    x_LS = selector.transform(x_LS)
    x_TS = selector.transform(x_TS)
    print('Number of features kept : ' + str(x_LS.shape[1]))
    '''

    # Divide the LS several times in different ways to predict the auc score.
    auc_all = np.array([])
    for i in range(10):
        x_LS_train, x_LS_test, y_LS_train, y_LS_test = train_test_split(x_LS, y_LS, test_size=0.33)


        s_train, s_test = stacking(estimators, x_LS_train, y_LS_train, x_LS_test, regression=False, mode='oof_pred_bag', needs_proba=True, metric=auc, shuffle=True, n_folds=3, stratified=True, verbose=2)
        #model = xgb
        model = lr
        #model = rfc
        #model = svc
        model.fit(s_train, y_LS_train)
        y_LS_pred = model.predict_proba(s_test)



        auc_predicted = auc(y_LS_test, y_LS_pred)
        print("  >" + str(i) + " Accuracy = " + str(auc_predicted))
        auc_all = np.append(auc_all, auc_predicted)

    print("############################################################")
    auc_mean = auc_all.mean()
    print("Mean of AUCs = {}".format(auc_mean))
    print("std of AUCs = {}".format(auc_all.std()))
    auc_predicted = auc_mean


    # Predicting on test set

    s_train, s_test = stacking(estimators, x_LS, y_LS, x_TS, regression=False, mode='oof_pred_bag', needs_proba=True, metric=auc, shuffle=True, n_folds=3, stratified=True, verbose=2)
    model = lr
    model.fit(s_train, y_LS)
    y_pred = model.predict_proba(s_test)

    # Making the submission file
    fname = make_submission(y_pred[:,1], auc_predicted, 'RF_20NN_LS')
    print('Submission file "{}" successfully written'.format(fname))
