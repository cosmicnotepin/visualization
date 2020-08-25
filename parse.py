import re
import os
from datetime import datetime
import contextlib
import itertools
from collections import defaultdict

import globs

def parseWimFile(directory, path):
    """
    helper function to parse the file that contains the wim message
    every vehicle log file points to one of these
    """
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


def parseVehicleFiles(directory):
    """
    parses a TrafficDetect "Daten" directory. Returns a list of vehicle dictionaries.
    each vehicle dictionary contains some data from the vehicle txt file and some from the WIM message file that is linked to in the vehicle file
    this function also parses the *.cls file (which contains the manual classification) that the manual classification GUI creates and 
    the *.acls file (which contains the classification by the current implementation of the 8+1 classifier in vehicle.java) that the ClassificationTester creates.
    """
    vehicles = []
    #'\d to differentiate from wim.txt files
    for entry in filter(lambda x: re.search(r'\d.txt', x.path), os.scandir(directory)):
        vehicle = {}
        try:
            with open(entry.path, 'r') as vf:
                line = vf.readline()
                pattern = '.*? height=(?P<height>.*?) width=(?P<width>.*?) length=(?P<length>.*?) weight=(?P<weight>.*?) maxAxleWeight=(?P<maweight>.*?) axles=(?P<axles>.*?) axleWeights=(?P<axleWeights>.*?) .*? numWimFiles=(?P<wim>[0-9]+) numAnprPics=(?P<anpr>[0-9]+) numIpCamPics=(?P<cam>[0-9]+) numScannerPics=(?P<scanner>[0-9]+)'
                match = re.match(pattern, line)
                if(match.group('length') == '-1,00'):
                    continue #invalid vehicle
                vehicle['length'] = float(match.group('length').replace(',', '.'))
                vehicle['axles'] = float(match.group('axles').replace(',', '.'))
                vehicle['weight'] = float(match.group('weight').replace(',', '.'))
                vehicle['height'] = float(match.group('height').replace(',', '.'))
                vehicle['width'] = float(match.group('width').replace(',', '.'))
                axleWeights = [float(x.replace(',', '.')) for x in match.group('axleWeights').split(';')]
                for i in range(globs.maxAxles):
                    if(i<len(axleWeights)):
                        vehicle['axleWeight' + str(i)] = axleWeights[i]
                    else:
                        vehicle['axleWeight' + str(i)] = 0

                with open(entry.path.replace(".txt", ".cls"), 'r') as cf:
                    vehicle['class'] = cf.readline()
                with open(entry.path.replace(".txt", ".acls"), 'r') as cf:
                    vehicle['aclass'] = cf.readline()

                vf.readline() #skip empty line
                axleSpacings, vehicleLoopLength = parseWimFile(directory, vf.readline())
                #loopLength makes no difference to knn
                #vehicle['vehicleLoopLength'] = vehicleLoopLength
                vehicle['axleSpacingsSum'] = 0
                for i in range(globs.maxAxles - 1):
                    if(i<len( axleSpacings )):
                        vehicle['axleSpacing' + str(i)] = axleSpacings[i]
                        vehicle['axleSpacingsSum'] = vehicle['axleSpacingsSum'] + axleSpacings[i]
                    else:
                        vehicle['axleSpacing' + str(i)] = 0
                #overhang worsens knn result
                #vehicle['overhang'] = vehicleLoopLength - vehicle['axleSpacingsSum']
                #numMainAxleSpacings makes no difference to knn
                #vehicle['numMainAxleSpacings'] = 0
                #for spacing in axleSpacings:
                #    if (spacing > 2):
                #        vehicle['numMainAxleSpacings'] += 1
                vehicles.append(vehicle)
        except FileNotFoundError:
            continue #no cls-file: skip

    return vehicles

def parseDebugLog(path):
    pattern = '.*? VHCL (?P<id>\d*?): (?P<mergeOrData>.*?) .*'
    dataPattern = '.*? VHCL (?P<id>\d*?): adding scanner data .*? min_y=(?P<min_y>.*?) max_y=(?P<max_y>.*?) height=(?P<height>.*?) width=(?P<width>.*?) scantime=(?P<scantime>.*?) .*'
    mergePattern = '.*? VHCL (?P<id>\d*?): combining vehicles (?P<vhcl1>.*?) and (?P<vhcl2>.*)'
    scanObjs = defaultdict(list) 
    merges = defaultdict(list)
    with open(path, 'r') as dl:
        i = itertools.count()
        for line in dl:
            if(next(i) > 100):
                break
            match = re.match(pattern, line)
            if(match):
                if(match.group('mergeOrData') == 'adding'):
                    dataMatch = re.match(dataPattern, line)
                    id = int(dataMatch.group('id')) 
                    scanObj = [
                            float(dataMatch.group('min_y').replace(',', '.')),
                            float(dataMatch.group('max_y').replace(',', '.')),
                            float(dataMatch.group('height').replace(',', '.')),
                            float(dataMatch.group('width').replace(',', '.')),
                            datetime.strptime(dataMatch.group('scantime'), '%H:%M:%S.%f')
                            ]
                    scanObjs[id].append(scanObj)
                elif(match.group('mergeOrData') == 'combining'):
                    mergeMatch = re.match(mergePattern, line)
                    remainingVhcl = int(mergeMatch.group('vhcl1'))
                    mergedVhcl = int(mergeMatch.group('vhcl2'))
                    merges[remainingVhcl].append(mergedVhcl)

    #scanObjs = merge(scanObjs, merges) 
    #vehicle['length'] = float(match.group('length').replace(',', '.'))

    print('wtf: ' + str(scanObjs))
    print('wtf2: ' + str(merges))
    return scanObjs

parseDebugLog(r'D:\Rohdaten\FRV_FREII1clean\Logs\2020-05-25_00-00-00_NKP-FREII1-debug.log')
