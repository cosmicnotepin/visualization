import sklearn
from sklearn import tree
import math
from tabulate import tabulate

import parse
import tls
import globs

def decisionTreeTest():
    """
    playing around with sklearn decision trees, comparing them with current classification
    and output in tls conform format
    """
    directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
    vehicles = parse.parseVehicleFiles(directory)
    samples = []
    labels = []
    aLabels = []
    #divide into samples, labels(true class from manual classification) and alabels(class from java 8+1 classification)
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

    half = math.floor(len(labels)/2)

    clf = tree.DecisionTreeClassifier(class_weight = 'balanced')#, max_depth = 3)#min_weight_fraction_leaf = 0.01)#, max_depth = 2)
    firstHalfSamples = samples[0:half]
    firstHalfLabels = labels[0:half]
    secondHalfSamples = samples[half:len(samples)]
    secondHalfLabels = labels[half:len(labels)]
    #fit on the first half of the data
    clf.fit(samples[0:half], labels[0:half])

    #fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
    #fn = [key for key,value in vehicles[0].items() if key not in ['aclass', 'class']]
    #print(fn)
    #tree.plot_tree(clf, feature_names = fn, class_names = [vc.name for vc in globs.vClass][0:9])
    #fig.savefig('barna.png')

    #predict all data
    predicted = clf.predict(samples)
    firstHalfPredicted = predicted[0:half]
    #but only use second half for evaluation
    secondHalfPredicted = predicted[half:len(predicted)]

    #print(sklearn.metrics.classification_report(labels[half:len(labels)], secondHalfPredicted, target_names = [vc.name for vc in globs.vClass][0:9]))
    #print(sklearn.metrics.classification_report(secondHalfLabels, secondHalfPredicted, target_names = [vc.name for vc in globs.vClass][0:9]))
    #print(sklearn.metrics.classification_report(labels, aLabels, target_names = [vc.name for vc in globs.vClass][0:9]))

    print("\n\nDT results =======================================================================================")
    #build results table for DecisionTree
    dtRes = []
    for i in range(0,9):
        dtRes.append([])
        for j in range(0,9):
            dtRes[i].append(0)
    for i in range(0, len(secondHalfPredicted)):
        dtRes[labels[half + i]][secondHalfPredicted[i]] += 1
    tls.printTLSRes(dtRes)
    for vc in list(globs.vClass)[0:9]:
        dtRes[vc.value].insert(0, vc.name)
    print(tabulate(dtRes, headers = [vc.name for vc in globs.vClass][0:9]))

    print("\n\njava results =====================================================================================")
    #build results table for java classification
    aRes = []
    for i in range(0,9):
        aRes.append([])
        for j in range(0,9):
            aRes[i].append(0)
    for i in range(0, len(labels)):
        aRes[labels[i]][aLabels[i]] += 1
    tls.printTLSRes(aRes)
    for vc in list(globs.vClass)[0:9]:
        aRes[vc.value].insert(0, vc.name)
    print(tabulate(aRes, headers = [vc.name for vc in globs.vClass][0:9]))
