import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, ensemble, neighbors, metrics, grid_search
import time
import sys, os
import pickle
called_version = sys.version_info
if called_version[0] == 2:
    from Tkinter import Tk
    from tkFileDialog import askopenfilename
elif called_version[0] == 3:
    from tkinter import Tk,filedialog

"""This file performs all functions necessary to extract and process data, then
make predictions for presence of heart disease based on Cleveland dataset.

Author:        Brandon Veber
Date Created:  6/12/2015
Last Modified: 6/12/2015

Dependencies:
    sklearn-0.16
    numpy
    pandas

Tested in Python 2.7 and 3.4
"""

def main(StringData):
    """This checks to see if a SVM model exists. If not it calls makeModel.  Then
    it encodes and normalizes the input data, then makes a prediction
    Inputs:
      StringData: string, required.  This contains all required patient information
                  separated by whitespace.
                  (age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal)
                  See heart-disease.names file for more details regarding attributes
   Outputs:
      The probability of the presence of heart disease
    """
    if not os.path.isfile('/'.join((os.getcwd(),'HeartPrediction.p'))): makeModel()
    modeller = pickle.load(open('HeartPrediction.p','rb'))
    inputData = StringData.split(' ')
    inputData = [float(elem) for elem in inputData]
    scaledData = normalizeTest(inputData,modeller['scaler'],modeller['encoder'])
    prediction = modeller['model'].predict_proba(scaledData)
    print(str(prediction[0][1]))

def makeModel(filename=None):
    """This makes a SVM model from the saved data and saves the model
    in a a pickle file in the same folder where this Python file is saved.
    Inputs:
      filename: string or None optional (default = None), this points to the location of the data file.
                if None, then the file selection GUI will run
    """
    if not filename: filename = getFilename()
    rawData = fetchData(filename)
    data = extractFeaturesAndTarget(rawData)
    trainingScaled,scaler,encoder = normalizeTrain(data['X'])
    model = svm.SVC(probability=True).fit(trainingScaled,data['y'])
    pickle.dump({'model':model,'data':data,'scaler':scaler,'encoder':encoder},
                open('HeartPrediction.p','wb'))

def analysis(filename=None,verbose=0):
    """This runs the entire analysis and returns scores
    Inputs:
      filename: string or None optional (default = None), this points to the location of the data file.
                if None, then the file selection GUI will run
      verbose: int, optional (default = 0). Controls verbosity of output statements
    Outputs:
      scores: numpy array. Contains the individual scores for all 10 folds.
              Row 1 is classification accuracy, Row 2 is Recall,
              Row 3 is precision, Row 4 is the ROC_AUC"""    
    t0 = time.time()
    if not filename: filename = getFilename()
    rawData = fetchData(filename)
    data = extractFeaturesAndTarget(rawData)
    score = crossValidate(data,verbose)
    if verbose >= 0:
        print("%-15s%-15s%-15s%-15s"%('Avg Accuracy', 'Avg Recall', 'Avg Precision','Avg ROC_AUC'))
        print("%-15.3f%-15.3f%-15.3f%-15.3f"%(np.mean(score[0]),np.mean(score[1]),np.mean(score[2]),np.mean(score[3])))
    print('Runtime is ',time.time()-t0,'(s)')
    return(score)

def crossValidate(data,verbose=0):
    """This performs a 10 fold cross-validation on the data
    Inputs:
      data: dict, required. This has the 'X' and 'y' values of all the data
      verbose: int, optional (default=0). This controls the verbosity of print statements
    Outputs:
      scores: numpy array. Contains the individual scores for all 10 folds.
              Row 1 is classification accuracy, Row 2 is Recall,
              Row 3 is precision, Row 4 is the ROC_AUC"""
    folds = cross_validation.KFold(len(data['y']),10,random_state=0)
    accuracies = np.array([])
    recalls = np.array([])
    precisions = np.array([])
    rocs = np.array([])
    nFold = 1
    for train_index,test_index in folds:
        X_train,X_test = data['X'][train_index],data['X'][test_index]
        y_train,y_test = data['y'][train_index],data['y'][test_index]
        X_train,X_test = normalize(X_train,X_test)
        clf = grid_search.GridSearchCV(
            svm.SVC(probability=True),{'C':[1],'kernel':['rbf']}
##            ensemble.RandomForestClassifier(),{'min_samples_split':[2,5,10,25,50],'n_estimators':[10,50,100]}
##            neighbors.KNeighborsClassifier(),{'n_neighbors':[2,5,10,25,50],'leaf_size':[15,30,45,60]}
            ).fit(X_train,y_train)
        if verbose>3:print(clf.best_estimator_)
        preds = clf.predict(X_test)
        predsProb = clf.predict_proba(X_test)
        if verbose>2:
            comp=[[y_test[i],predsProb[i,1]] for i in range(len(y_test))]
            for row in comp: print(row)
        roc = metrics.roc_auc_score(y_test,predsProb[:,1])
        acc = metrics.accuracy_score(y_test,preds)
        recall = metrics.recall_score(y_test,preds)
        precision = metrics.precision_score(y_test,preds)
        rocs=np.append(rocs,roc)
        accuracies=np.append(accuracies,acc)
        recalls=np.append(recalls,recall)
        precisions=np.append(precisions,precision)
        if verbose>1:
            print('%-15s%-15s%-15s%-15s%-15s'%('Fold','Accuracy','Recall','Precision','ROC_AUC'))
            print('%-15.3f%-15.3f%-15.3f%-15.3f%-15.3f'%(nFold,acc,recall,precision,roc))
        nFold += 1
    return(np.array((accuracies,recalls,precisions,roc)))

def normalize(train,test,scalerType='Standard'):
    """Return the scaled data.  Only performs scaling on real values
    Inputs:
      train: numpy array, required.  This is the training data to scale
      test: numpy array, required.  This is the test data to scale
      scalerType: string, optional (default = 'Standard'). This defines
      the type of input scaling to be used. Options are 'MinMax' or 'Standard'
    Outputs:
      scaledData: numpy array. This is the scaled output data"""
    trainScaled,scaler,encoder = normalizeTrain(train,scalerType)
    testScaled = normalizeTest(test,scaler,encoder)
    return(trainScaled,testScaled)


def normalizeTrain(train,scalerType='Standard'):
    """Return the scaled data and scaler.  Only performs scaling on real values
    Inputs:
      train: numpy array, required.  This is the training data to scale
      scalerType: string, optional (default = 'Standard'). This defines
      the type of input scaling to be used. Options are 'MinMax', 'Standard' or None
    Outputs:
      scaledData: numpy array. This is the scaled training data
      scaler: sklearn scaler. This is the scaler used to transform training set
      encoder: sklearn OneHotEncoder.  This encodes categorical features to binary"""
    #Categorical Indices = [2,6,10,12], convert with one hot encoder
    encoder = preprocessing.OneHotEncoder('auto',[2,6,10,12],sparse=False)
    features = encoder.fit_transform(train)
    #Start of real value index. Everything before is one-hot encoded
    realValueIndex = 13
    realValuedData = features[:,realValueIndex:]
    if scalerType == 'MinMax':
        scaler = preprocessing.MinMaxScaler()
        scaledData = scaler.fit_transform(realValuedData)
    elif scalerType == 'Standard':
        scaler = preprocessing.StandardScaler()
        scaledData = scaler.fit_transform(realValuedData)
    return(np.hstack((features[:,:realValueIndex],scaledData)),scaler,encoder)

def normalizeTest(test,scaler,encoder):
    """Return the scaled test data.  Only performs scaling on real values
    Inputs:
      test: numpy array, required.  This is the training data to scale
      scaler: sklearn scaler object. This is the scaler used to transform training set
      encoder: sklearn OneHotEncoder.  This encodes categorical features to binary
    Outputs:
      scaledData: numpy array. This is the scaled test data"""
    #Start of real value index. Everything before is one-hot encoded
    features = encoder.transform(test)
    realValueIndex = 13
    realValuedData = features[:,realValueIndex:]
    scaledData = scaler.transform(realValuedData)
    return(np.hstack((features[:,:realValueIndex],scaledData)))

def extractFeaturesAndTarget(inputData):
    """This seperates the input features and target labels
    Inputs:
      inputData: numpy array, required.  This is the entire dataset with features plus target
    Outputs:
      outputData: dictionary.  Dictionary with 2 keys: 'X' and 'y'.  'X' is the
                  input feature, output is binary target"""
    
    #Make copy or original data
    inputDataWork = inputData.copy()
    #Replace missing values (?'s) with NaN's then convert to float
    inputDataWork[inputDataWork == '?'] = np.nan
    #Impute missing data to most frequent value per column, then convert to float
    inputDataWork.astype(np.float)
    features = preprocessing.Imputer(strategy='most_frequent').fit_transform(inputDataWork[:,:-1])
    #Binarize target labels, anything > 1 indicates presence of heart disease
    inputDataWork[:,-1][inputDataWork[:,-1] >= 1]=1
    return({'X':features,'y':np.array(inputDataWork[:,-1],dtype='f')})

def fetchData(filename='C:/Users/Brandon/Downloads/processed.cleveland.data'):
    """ Return data from preprocessed Cleveland dataset as numpy array
    Inputs:
      filename: string, optional.  This points to the data file.  Input not
      necessary if being run on Brandon's system
    Outputs:
      Numpy array.  This is the 303x14 dataset in numpy array format"""
    return(np.array(pd.read_csv(filename,header=None)))

def getFilename():
    """GUI for selecting the proper CSV files
    Inputs: None
    Outputs:
      fname: str.  The filename that is output based on the user selection
    """
    ###File Selection GUI###
    root = Tk()
    if called_version[0] == 3:fname = filedialog.askopenfilename(title='Find Cleveland Heart Data',filetypes=(('DATA Files','.data'),('All Files','.*')))
    elif called_version[0] == 2:fname = askopenfilename(title='Find Cleveland Heart Data',filetypes=(('DATA Files','.data'),('All Files','.*')))
    root.destroy()
    #################
    return(fname)
