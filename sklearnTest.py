import sklearn
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import math
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from joblib import dump, load

from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

import parse
import tls
import globs

fn = 0

def parseForSklearn(directory):
    """
    parse vehicles and divide into samples, labels(true class from manual classification) and alabels(class from java 8+1 classification)
    """
    vehicles = parse.parseVehicleFiles(directory)
    #TODO
    global fn 
    fn = [key for key,value in vehicles[0].items() if key not in ['aclass', 'class']]
    samples = []
    labels = []
    aLabels = []
    for v in vehicles:
        sample = []
        def inner():
            for key, value in v.items():
                if (key == 'class'):
                    if (v[key] in [vc.name for vc in globs.vClass][0:9]):
                        labels.append(globs.vClass[v[key]].value)
                    else:
                        print(v[key])
                        return
                elif (key == 'aclass'):
                    aLabels.append(globs.vClass[v[key]].value)
                else:
                    sample.append(v[key])

            samples.append(sample) 
        inner()

    return samples, labels, aLabels

def visualizeCMx2(clf, stest, ltest, identifier):
    disp = plot_confusion_matrix(clf, stest, ltest, display_labels=[vc.name for vc in globs.vClass][1:9], cmap=plt.cm.Blues, normalize='true') 
    plt.title(identifier + ' rn')
    disp = plot_confusion_matrix(clf, stest, ltest, display_labels=[vc.name for vc in globs.vClass][1:9], cmap=plt.cm.Blues, normalize='pred') 
    plt.title(identifier + ' cn')

def bghTest(labels, aLabels):
    print("\nbgh========================================")
    cm = confusion_matrix(labels, aLabels, labels=[vc.value for vc in globs.vClass][1:9], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[vc.name for vc in globs.vClass][1:9]) 
    disp = disp.plot(include_values=True, cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal') 
    plt.title('bgh rn')
    cm = confusion_matrix(labels, aLabels, normalize='pred', labels=[vc.value for vc in globs.vClass][1:9])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[vc.name for vc in globs.vClass][1:9]) 
    disp = disp.plot(include_values=True, cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal') 
    plt.title('bgh cn')
    printClfRes(labels, aLabels)

def DTCheck():
    """
    finding solutions to PKW vs Kleintransporter
    """
    directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
    samples, labels, aLabels = parseForSklearn(directory)
    s = []
    l = []
    for i in range(0,len(labels)):
        if (labels[i] in [globs.vClass['PKW'].value, globs.vClass['Kleintransporter'].value]):
            s.append(samples[i])
            l.append(labels[i])

    samples = s
    labels = l
    strain, stest, ltrain, ltest = train_test_split(samples, labels, test_size = 0.5, random_state = 42)
    clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=20)#min_weight_fraction_leaf = 0.01)#, max_depth = 2)
    #clf = tree.DecisionTreeClassifier(class_weight = 'balanced', max_depth=3, min_samples_split=20)#min_weight_fraction_leaf = 0.01)#, max_depth = 2)
    #fit on the first half of the data
    clf.fit(strain, ltrain)

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
    #fn = [key for key,value in vehicles[0].items() if key not in ['aclass', 'class']]
    #print(fn)
    tree.plot_tree(clf, feature_names = fn, class_names = [vc.name for vc in globs.vClass][2:9])
    fig.savefig('pvk.png')

    #predict all data
    predicted = clf.predict(stest)
    #but only use second half for evaluation

    #print(sklearn.metrics.classification_report(labels[half:len(labels)], secondHalfPredicted, target_names = [vc.name for vc in globs.vClass][0:9]))
    #print(sklearn.metrics.classification_report(secondHalfLabels, secondHalfPredicted, target_names = [vc.name for vc in globs.vClass][0:9]))
    #print(sklearn.metrics.classification_report(labels, aLabels, target_names = [vc.name for vc in globs.vClass][0:9]))

    visualizeCMx2(clf, stest, ltest, 'DT')
    printClfRes(ltest, predicted)

def RFTest(strain, stest, ltrain, ltest):
    print("\nRF========================================")
    clf = RandomForestClassifier(max_features=int(math.sqrt(len(strain[0]))), n_estimators=30, min_samples_split=2, n_jobs=3, class_weight = 'balanced')
    clf.fit(strain, ltrain)
    dump(clf, 'rf.joblib')
    predicted = clf.predict(stest)
    visualizeCMx2(clf, stest, ltest, 'RF')
    printClfRes(ltest, predicted)

def SVMRBF(strain, stest, ltrain, ltest):
    """
    playing around with
    """
    print("\nSVMRBF========================================")
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf', class_weight = 'balanced'))
    clf.fit(strain,ltrain)
    predicted = clf.predict(stest)
    visualizeCMx2(clf, stest, ltest, 'SVMRBF')
    printClfRes(ltest, predicted)


def SVMLinearTest(strain, stest, ltrain, ltest):
    """
    playing around with
    """
    print("\nSVCLinear========================================")
    clf = make_pipeline(StandardScaler(), svm.LinearSVC(class_weight = 'balanced'))
    clf.fit(strain,ltrain)
    predicted = clf.predict(stest)
    visualizeCMx2(clf, stest, ltest, 'SVMLinear')
    printClfRes(ltest, predicted)


def pmmlTest(strain, stest, ltrain, ltest):
    pipeline = PMMLPipeline([
    	("classifier", tree.DecisionTreeClassifier())
    ])
    pipeline.fit(strain, ltrain)
    
    dump(pipeline, "model.pkl.z", protocol=2)
    clf = load("model.pkl.z")
    print(clf.predict(stest[:1]))
    #pipeline.verify(strain[:10])
    #sklearn2pmml(pipeline, "model.pmml")#, with_repr = False)

def DTTest(strain, stest, ltrain, ltest):
    """
    playing around with sklearn decision trees, comparing them with current classification
    and output in tls conform format
    """
    print("\nDT========================================")
    directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
    clf = tree.DecisionTreeClassifier(class_weight = 'balanced')#, max_depth = 3)#min_weight_fraction_leaf = 0.01)#, max_depth = 2)
    #fit on the first half of the data
    clf.fit(strain, ltrain)

    #fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
    #fn = [key for key,value in vehicles[0].items() if key not in ['aclass', 'class']]
    #print(fn)
    #tree.plot_tree(clf, feature_names = fn, class_names = [vc.name for vc in globs.vClass][0:9])
    #fig.savefig('barna.png')

    #predict all data
    predicted = clf.predict(stest)
    #but only use second half for evaluation

    #print(sklearn.metrics.classification_report(labels[half:len(labels)], secondHalfPredicted, target_names = [vc.name for vc in globs.vClass][0:9]))
    #print(sklearn.metrics.classification_report(secondHalfLabels, secondHalfPredicted, target_names = [vc.name for vc in globs.vClass][0:9]))
    #print(sklearn.metrics.classification_report(labels, aLabels, target_names = [vc.name for vc in globs.vClass][0:9]))

    visualizeCMx2(clf, stest, ltest, 'DT')
    printClfRes(ltest, predicted)

def printClfRes(labels, predicted):
    tls.printTLSRes(confusionMatrix(labels, predicted))
    #for vc in list(globs.vClass)[0:9]:
    #    aRes[vc.value].insert(0, vc.name)
    #print(tabulate(aRes, headers = [vc.name for vc in globs.vClass][0:9]))

def confusionMatrix(labels, predicted):
    cm = []
    for i in range(0,9):
        cm.append([])
        for j in range(0,9):
            cm[i].append(0)
    for i in range(0, len(labels)):
        cm[labels[i]][predicted[i]] += 1
    return cm

def kNeighborsTest(strain, stest, ltrain, ltest):
    dump(stest[:1], 'oneVehicle.joblib')
    #print("\nKNeighbors uniform========================================")
    #clf = neighbors.KNeighborsClassifier()
    #clf.fit(strain,ltrain)
    #predicted = clf.predict(stest)
    #printClfRes(ltest, predicted)
    #visualizeCMx2(clf, stest, ltest, 'knn uniform')
    #print("\nKNeighbors uniform n_neighbors = 1 ========================================")
    #clf = neighbors.KNeighborsClassifier(1)
    #clf.fit(strain,ltrain)
    #predicted = clf.predict(stest)
    #visualizeCMx2(clf, stest, ltest, 'knn uniform 1 neighbor')
    #printClfRes(ltest, predicted)
    print("\nKNeighbors distance========================================")
    clf = make_pipeline(StandardScaler(), neighbors.KNeighborsClassifier(weights = 'distance'))
    clf.fit(strain,ltrain)
    dump(clf, 'knn.joblib')
    predicted = clf.predict(stest)
    visualizeCMx2(clf, stest, ltest, 'knn distance')
    printClfRes(ltest, predicted)
    print(sklearn.metrics.classification_report(ltest, predicted, target_names = [vc.name for vc in globs.vClass][1:9]))
    #print("\nKNeighbors distance n_neighbors = 1========================================")
    #clf = neighbors.KNeighborsClassifier(1, weights = 'distance')
    #clf.fit(strain,ltrain)
    #predicted = clf.predict(stest)
    #visualizeCMx2(clf, stest, ltest, 'knn distance 1 neighbor')
    #printClfRes(ltest, predicted)

def knnGSTest(strain, stest, ltrain, ltest):
    #List Hyperparameters that we want to tune.
    n_neighbors = list(range(1,30))
    #metrics = ["euclidean",  "manhattan",  "chebyshev",  "minkowski",  "wminkowski", "seuclidean", "mahalanobis"]
    p = [1,2]
    weights = ['uniform', 'distance']
    #hyperparameters = dict(n_neighbors=n_neighbors, weights=weights, metric=metrics)
    tuned_parameters = [{'kneighborsclassifier__n_neighbors': n_neighbors, 'kneighborsclassifier__p': p, 'kneighborsclassifier__weights': weights}]
    clf = GridSearchCV(make_pipeline(StandardScaler(), neighbors.KNeighborsClassifier()), tuned_parameters)
    clf.fit(strain,ltrain)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()

    visualizeCMx2(clf, stest, ltest, 'knn gridsearchcv')
    predicted = clf.predict(stest)
    printClfRes(ltest, predicted)
    print(sklearn.metrics.classification_report(ltest, predicted, target_names = [vc.name for vc in globs.vClass][1:9]))

def rNeighborsTest(strain, stest, ltrain, ltest):
    print("\nRNeighbors========================================")
    clf = neighbors.RadiusNeighborsClassifier()
    clf.fit(strain,ltrain)
    predicted = clf.predict(stest)
    printClfRes(ltest, predicted)

def featureCheck():
    directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
    samples, labels, aLabels = parseForSklearn(directory)
    #data = pd.read_csv("D://Blogs//train.csv")
    #X = data.iloc[:,0:20]  #independent columns
    #y = data.iloc[:,-1]    #target column i.e price range#apply SelectKBest class to extract top 10 best features
    X = pd.DataFrame(samples)
    y = pd.DataFrame(labels)
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features
    print(fn)

def fc2():
    directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
    #samples, labels, aLabels = parseForSklearn(directory)
    #samples = pd.DataFrame(samples)
    #labels = pd.DataFrame(labels)
    vehicles = parse.parseVehicleFiles(directory)
    df = pd.DataFrame(vehicles)
    #df = df[df['class'].isin(['PKW', 'Kleintransporter'])]
    df['axleWeightDiff0'] = df['axleWeight0'] - df['axleWeight1']
    samples = df.drop(columns=['class', 'aclass'])
    labels = df['class']
    #print(df.columns)
    clf = RandomForestClassifier(max_features=int(math.sqrt(len(samples.iloc[0]))), n_estimators=30, min_samples_split=2, n_jobs=3, class_weight = 'balanced')
    clf.fit(samples, labels)
    print(clf.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(clf.feature_importances_, index=samples.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

def overview():
    #directory = r'D:\Rohdaten\FREII12020-07-16-clean\Daten'
    directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
    samples, labels, aLabels = parseForSklearn(directory)
    strain, stest, ltrain, ltest = train_test_split(samples, labels, test_size = 0.5, random_state = 42)
    #bghTest(labels, aLabels)
    #DTTest(strain, stest, ltrain, ltest)
    #SVMLinearTest(strain, stest, ltrain, ltest)
    #SVMRBF(strain, stest, ltrain, ltest)
    #RFTest(strain, stest, ltrain, ltest)
    #kNeighborsTest(strain, stest, ltrain, ltest)
    #knnGSTest(strain, stest, ltrain, ltest)
    #rNeighborsTest(strain, stest, ltrain, ltest)
    pmmlTest(strain, stest, ltrain, ltest)
    plt.show()

overview()
#DTCheck()
#featureCheck()
#fc2()

