from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys
from pandas import *
from datetime import datetime
import contextlib
from enum import Enum
import sklearn
from sklearn import tree
import math
from tabulate import tabulate

#textitext
maxAxles = 10
#directory = r'C:\Users\borgro\svn\java\SensorFusion\Daten\Daten'
#directory = r'D:\Rohdaten\FRV\Daten'
#directory = r'D:\Rohdaten\FREII1\Daten'
directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
class2Marker = { "nicht_klass_Kfz":dict(marker='x', c='black'), "Motorrad":dict(marker='4', c='yellow'), "PKW":dict(marker='o', c='blue'), "Kleintransporter":dict(marker='>', c='green'), "PKW_Anhaenger":dict(marker='s', c='cyan'), "LKW":dict(marker='+', c='red'), "LKW_Anhaenger":dict(marker='P', c='magenta'), "Sattel_Kfz":dict(marker='*', c='brown'), "Bus":dict(marker='_', c='DarkGoldenRod'), "ungueltig":dict(marker='None', c='DarkGoldenRod')  }

class vClass(Enum):
    nicht_klass_Kfz = 0
    Motorrad = 1
    PKW = 2
    Kleintransporter = 3
    PKW_Anhaenger = 4
    LKW = 5
    LKW_Anhaenger = 6
    Sattel_Kfz = 7
    Bus = 8
    ungueltig = 9
    fraglich = 10

def parseWimFile(path):
    axleSpacings = []
    pattern = r'.*?\\Daten(?P<relPath>.*)'
    match = re.match(pattern, path)
    with open(directory + match.group('relPath'), 'r') as wf:
        line = wf.readline()
        pattern = 'start=(?P<start>.*?) stop=(?P<stop>.*?) speed=(?P<speed>.*?) .*? NumScaleAxleIdWeight=(?P<NumScaleAxleIdWeight>.*?) NumAxleCountPerScale=(?P<NumAxleCountPerScale>.*?) NumAxleSpacing=(?P<NumAxleSpacing>.*?) .*'
        match = re.match(pattern, line)
        startTime = datetime.strptime(match.group('start'), '%Y-%m-%d %H:%M:%S.%f')
        stopTime = datetime.strptime(match.group('stop'), '%Y-%m-%d %H:%M:%S.%f')
        speed = float(match.group('speed').replace(',','.'))
        vehicleTimeSeconds = (stopTime - startTime).total_seconds()
        vehicleLoopLength = (speed/3.6) * vehicleTimeSeconds
        NumScaleAxleIdWeight = int(match.group('NumScaleAxleIdWeight'))
        NumAxleCountPerScale = int(match.group('NumAxleCountPerScale'))
        NumAxleSpacing = int(match.group('NumAxleSpacing'))
        for i in range(NumScaleAxleIdWeight + NumAxleCountPerScale):
            wf.readline()
        for i in range(NumAxleSpacing):
            axleSpacings.append(float(wf.readline().replace(',', '.')))

    return axleSpacings, vehicleLoopLength

def parseVehicleFiles(directory = directory):
    vehicles = []
    #'\d to differentiate from wim.txt files
    for entry in filter(lambda x: re.search(r'\d.txt', x.path), os.scandir(directory)):
        vehicle = {}
        try:
            with open(entry.path, 'r') as vf:
                line = vf.readline()
                pattern = '.*? length=(?P<length>.*?) weight=(?P<weight>.*?) maxAxleWeight=(?P<maweight>.*?) axles=(?P<axles>.*?) axleWeights=(?P<axleWeights>.*?) .*? numWimFiles=(?P<wim>[0-9]+) numAnprPics=(?P<anpr>[0-9]+) numIpCamPics=(?P<cam>[0-9]+) numScannerPics=(?P<scanner>[0-9]+)'
                match = re.match(pattern, line)
                if(match.group('length') == '-1,00'):
                    continue #invalid vehicle
                vehicle['length'] = float(match.group('length').replace(',', '.'))
                vehicle['axles'] = float(match.group('axles').replace(',', '.'))
                vehicle['weight'] = float(match.group('weight').replace(',', '.'))
                axleWeights = [float(x.replace(',', '.')) for x in match.group('axleWeights').split(';')]
                for i in range(maxAxles):
                    if(i<len(axleWeights)):
                        vehicle['axleWeight' + str(i)] = axleWeights[i]
                    else:
                        vehicle['axleWeight' + str(i)] = 0

                with open(entry.path.replace(".txt", ".cls"), 'r') as cf:
                    vehicle['class'] = cf.readline()
                with open(entry.path.replace(".txt", ".acls"), 'r') as cf:
                    vehicle['aclass'] = cf.readline()

                vf.readline() #skip empty line
                axleSpacings, vehicleLoopLength = parseWimFile(vf.readline())
                vehicle['vehicleLoopLength'] = vehicleLoopLength
                vehicle['axleSpacingsSum'] = 0
                for i in range(maxAxles - 1):
                    if(i<len( axleSpacings )):
                        vehicle['axleSpacing' + str(i)] = axleSpacings[i]
                        vehicle['axleSpacingsSum'] = vehicle['axleSpacingsSum'] + axleSpacings[i]
                    else:
                        vehicle['axleSpacing' + str(i)] = 0
                vehicle['overhang'] = vehicleLoopLength - vehicle['axleSpacingsSum']
                vehicles.append(vehicle)
        except FileNotFoundError:
            continue #no cls-file: skip

    return vehicles

def plotVehicles(vehicles, xDim='length', yDim='weight', zDim=None, classes=class2Marker.keys()):
    df = DataFrame(vehicles)
    fig = plt.figure()
    if (zDim):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel(zDim)
    else:
        ax = fig.add_subplot(111)

    ax.set_xlabel(xDim)
    ax.set_ylabel(yDim)

    for key in classes:
        filteredDf = df[df['class'] == key]
        if (zDim):
            ax.scatter(filteredDf[xDim], filteredDf[yDim], filteredDf[zDim],  **class2Marker[key])
        else:
            ax.scatter(filteredDf[xDim], filteredDf[yDim], **class2Marker[key])
    fig.show()


def plotVehiclesTest(vehicles, xDim='length', yDim='weight', zDim=None, classes=class2Marker.keys()):
    minAxleDistance = 2 
    #vehicles = vehicles[19:20]

    #Gewicht der zweiten Hauptachse
    for vehicle in vehicles:
        foundSecondMain = False
        for i in range(0, maxAxles - 2):
            if (not foundSecondMain):
                if (vehicle['axleSpacing' + str(i)] < minAxleDistance):
                    continue
                else:
                    foundSecondMain = True
                    vehicle['secondMainAxleWeight'] = vehicle['axleWeight' + str(i + 1)]
                    continue

            if (foundSecondMain and vehicle['axleSpacing' + str(i)] < minAxleDistance):
                vehicle['secondMainAxleWeight'] += vehicle['axleWeight' + str(i + 1)]
            else:
                vehicle['secondMainSpacing'] = vehicle['axleSpacing' + str(i)]
                break

    df = DataFrame(vehicles)

    #df['custom'] = abs(df['axleWeight0'] - df['axleWeight1']) # Differenz Achsgewichte

    #Hauptachsabstaende
    df['custom'] = (df['axleSpacing0'] > minAxleDistance).astype(int)
    for i in range(1,maxAxles-1):
        df['custom'] += (df['axleSpacing' + str(i)] > minAxleDistance).astype(int)

    df = df[df['axles'] > 2]
    #df = df[df['axleSpacing0'] > 5.7]

    #df['i0'] = (df['axleSpacing0'] < minAxleDistance).astype(int)
    #df['marker'] = (df['axleSpacing0'] > minAxleDistance).astype(int)
    #for i in range(1,maxAxles-1):
    #    df['i0'] += (df['axleSpacing' + str(i)] < minAxleDistance && df['marker'] == 0).astype(int)
    #    df['marker'] = (df['axleSpacing' + str(i)] > minAxleDistance || df['marker'] == 1).astype(int)

    #df['i1'] = df['i0']
    #for i in range(1,maxAxles-1):
    #    df['i1'] += (df['axleSpacing' + str(i)] < minAxleDistance && df['marker'] == 1).astype(int)
    #    df['marker'] = (df['axleSpacing' + str(i)] < minAxleDistance && df['marker'] == 1).astype(int)

    fig = plt.figure()
    if (zDim):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel(zDim)
    else:
        ax = fig.add_subplot(111)

    ax.set_xlabel(xDim)
    ax.set_ylabel(yDim)

    for key in classes:
        filteredDf = df[df['class'] == key]
        if (zDim):
            ax.scatter(filteredDf[xDim], filteredDf[yDim], filteredDf[zDim],  **class2Marker[key])
        else:
            ax.scatter(filteredDf[xDim], filteredDf[yDim], **class2Marker[key])
    fig.show()

def explainPKWVsKleintransporter():
    plotVehicles(parseVehicleFiles(), classes=['PKW', 'Kleintransporter'], xDim = 'axleSpacing0')

def fixData():
    vehicles = parseVehicleFiles()

def cleanupData():
    directory = r'D:\Rohdaten\FRV_FREII1clean'
    vehicles = []
    noVDel = 0
    noPDel = 0
    #'\d to differentiate from wim.txt files
    for entry in filter(lambda x: re.search(r'\d.txt', x.path), os.scandir(directory + r'\Daten')):
        delete = False
        with open(entry.path, 'r') as vf:
            line = vf.readline()
            pattern = '.*? length=(?P<length>.*?) weight=(?P<weight>.*?) maxAxleWeight=(?P<maweight>.*?) axles=(?P<axles>.*?) axleWeights=(?P<axleWeights>.*?) .*? numWimFiles=(?P<wim>[0-9]+) numAnprPics=(?P<anpr>[0-9]+) numIpCamPics=(?P<cam>[0-9]+) numScannerPics=(?P<scanner>[0-9]+)'
            match = re.match(pattern, line)

            #no wim data or no pictures means it is useless
            if(match.group('wim') == '0' or match.group('scanner') == '0'): 
                delete = True
                with contextlib.suppress(FileNotFoundError):
                    os.remove(entry.path.replace(".txt", ".cls")) 
                    vf.readline() #skip empty line
                    pattern = r'.*?(?P<relPath>\\Daten.*)'
                    for i in range(0, int(match.group('wim'))):
                        line = vf.readline()
                        wimMatch = re.match(pattern, line)
                        print(entry)
                        os.remove(directory + wimMatch.group('relPath'))
                        noPDel += 1
                    noPics = int(match.group('anpr')) + int(match.group('cam')) + int(match.group('scanner')) 
                    pattern = r'.*?(?P<relPath>\\Bilder.*)'
                    for i in range(0, noPics):
                        line = vf.readline()
                        picMatch = re.match(pattern, line)
                        os.remove(directory + picMatch.group('relPath'))
                        noPDel += 1

        if (delete):
            os.remove(entry)
            noVDel += 1

    print('removed vehicles: ' + str(noVDel) + ' removed pictures: ' + str(noPDel))

def decisionTreeTest():
    directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
    vehicles = parseVehicleFiles(directory)
    samples = []
    labels = []
    aLabels = []
    for v in vehicles:
        sample = []
        def inner():
            for key, value in v.items():
                if (key == 'class'):
                    if (v[key] in [vc.name for vc in vClass][0:9]):
                        labels.append(vClass[v[key]].value)
                    else:
                        print(v[key])
                        return
                elif (key == 'aclass'):
                    aLabels.append(vClass[v[key]].value)
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
    clf.fit(samples[0:half], labels[0:half])
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=600)
    fn = [key for key,value in vehicles[0].items() if key not in ['aclass', 'class']]
    print(fn)
    tree.plot_tree(clf, feature_names = fn, class_names = [vc.name for vc in vClass][0:9])
    fig.savefig('barna.png')
    predicted = clf.predict(samples)
    firstHalfPredicted = predicted[0:half]
    secondHalfPredicted = predicted[half:len(predicted)]
    #print(sklearn.metrics.classification_report(labels[half:len(labels)], secondHalfPredicted, target_names = [vc.name for vc in vClass][0:9]))
    print(sklearn.metrics.classification_report(secondHalfLabels, secondHalfPredicted, target_names = [vc.name for vc in vClass][0:9]))
    print(sklearn.metrics.classification_report(labels, aLabels, target_names = [vc.name for vc in vClass][0:9]))
    res = []
    for i in range(0,9):
        res.append([])
        for j in range(0,9):
            res[i].append(0)
    for i in range(0, len(secondHalfPredicted)):
        res[labels[half + i]][secondHalfPredicted[i]] += 1
    print(res)
    for vc in list(vClass)[0:9]:
        res[vc.value].insert(0, vc.name)
    print(res)
    print(tabulate(res, headers = [vc.name for vc in vClass][0:9]))



