import re
import os
from datetime import datetime, timedelta
import contextlib
import itertools
from collections import defaultdict

import globs

def addWimFileData(vhcl, directory, path):
    """
    helper function to parse the file that contains the wim message and add the contained data to the vehicle
    every vehicle log file points to one of these
    """
    axleSpacings = []
    pattern = r'.*?\\Daten(?P<relPath>.*)'
    match = re.match(pattern, path)
    with open(directory + match.group('relPath'), 'r') as wf:
        line = wf.readline()
        pattern = 'start=(?P<start>.*?) stop=(?P<stop>.*?) speed=(?P<speed>.*?) .*? NumScaleAxleIdWeight=(?P<NumScaleAxleIdWeight>.*?) NumAxleCountPerScale=(?P<NumAxleCountPerScale>.*?) NumAxleSpacing=(?P<NumAxleSpacing>.*?) .*'
        match = re.match(pattern, line)
        #vhcl['wimStartTime'] = datetime.strptime(match.group('start'), '%Y-%m-%d %H:%M:%S.%f').total_seconds()
        #vhcl['wimStopTime'] = datetime.strptime(match.group('stop'), '%Y-%m-%d %H:%M:%S.%f').total_seconds()
        #vhcl['wimSpeed'] = float(match.group('speed').replace(',','.'))
        #vehicleTimeSeconds = (stopTime - startTime).total_seconds()
        #vehicleLoopLength = (speed/3.6) * vehicleTimeSeconds
        NumScaleAxleIdWeight = int(match.group('NumScaleAxleIdWeight'))
        NumAxleCountPerScale = int(match.group('NumAxleCountPerScale'))
        NumAxleSpacing = int(match.group('NumAxleSpacing'))
        for i in range(NumScaleAxleIdWeight + NumAxleCountPerScale):
            wf.readline()
        for i in range(globs.maxAxles):
            if(i < NumAxleSpacing):
                vhcl['axleSpacing' + str(i)] = float(wf.readline().replace(',', '.'))
            else:
                vhcl['axleSpacing' + str(i)] = 0

def parseVehicleFiles(directory):
    """
    parses all data that is parsable for a vehicle, the vehicle data file, the wim log file and the debugLog for the scanObjs
    Returns a list of vehicle dictionaries.
    each vehicle dictionary contains some data from the vehicle txt file and some from the WIM message file that is linked to in the vehicle file
    this function also parses the *.cls file (which contains the manual classification) that the manual classification GUI creates and 
    the *.acls file (which contains the classification by the current implementation of the 8+1 classifier in vehicle.java) that the ClassificationTester creates.
    """

    pattern = '.*? id=(?P<id>.*?) .*? height=(?P<height>.*?) width=(?P<width>.*?) length=(?P<length>.*?) weight=(?P<weight>.*?) maxAxleWeight=(?P<maweight>.*?) axles=(?P<axles>.*?) axleWeights=(?P<axleWeights>.*?) .*? numWimFiles=(?P<wim>[0-9]+) numAnprPics=(?P<anpr>[0-9]+) numIpCamPics=(?P<cam>[0-9]+) numScannerPics=(?P<scanner>[0-9]+)'
    patternRE = re.compile(pattern)
    vehicles = []
    #'\d to differentiate from wim.txt files
    for entry in filter(lambda x: re.search(r'\d.txt', x.path), os.scandir(directory + r'\Daten')):
        vehicle = {}
        try:
            with open(entry.path, 'r') as vf:
                line = vf.readline()
                match = patternRE.match(line)
                if(match.group('length') == '-1,00'):
                    continue #invalid vehicle
                vehicle['id'] = match.group('id')
                vehicle['filename'] = entry.path
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
                    vehicle['aClass'] = cf.readline()

                vf.readline() #skip empty line
                addWimFileData(vehicle, directory + r'\Daten', vf.readline())
                vehicles.append(vehicle)
        except FileNotFoundError:
            continue #no cls-file: skip

    addDebugLogData(vehicles, directory)
    return vehicles

def addDebugLogData(vehicles, directory):
    path = next(os.scandir(directory + r'\Logs'))
    pattern = '.{24}VHCL (?P<id>\d*?): (?P<mergeOrData>.*? .*?) .*'
    dataPattern = '.{24}VHCL (?P<id>\d*?): adding scanner data .*? min_y=(?P<min_y>.*?) max_y=(?P<max_y>.*?) height=(?P<height>.*?) width=(?P<width>.*?) scantime=(?P<scth>\d{2}):(?P<sctm>\d{2}):(?P<scts>\d{2})\.(?P<sctms>\d{3}) .*'
    dataRE = re.compile(dataPattern)
    mergePattern = '.{24}VHCL (?P<id>\d*?): combining vehicles (?P<vhcl1>.*?) and (?P<vhcl2>.*)'
    mergeRE = re.compile(mergePattern)
    scanObjs = defaultdict(list) 
    merges = defaultdict(list)

    with open(path, 'r') as dl:
        for line in dl:
            match = dataRE.match(line)
            if(match):
                dataMatch = match
                id = dataMatch.group('id') 
                scanObj = {}
                scanObj['min_y'] = float(dataMatch.group('min_y').replace(',', '.'))
                scanObj['max_y'] = float(dataMatch.group('max_y').replace(',', '.'))
                scanObj['height'] = float(dataMatch.group('height').replace(',', '.'))
                scanObj['width'] = float(dataMatch.group('width').replace(',', '.'))
                #scanObj['scantime'] = (datetime.strptime(dataMatch.group('scantime'), '%H:%M:%S.%f') + timedelta(days=100000)).timestamp() #+10000 because of timestamp() bug for years close to epoch and then replace with a steak right before consumption
                scanObj['scantime'] = int(dataMatch.group('scth'))*3600 + int(dataMatch.group('sctm'))*60 + int(dataMatch.group('scts')) + int(dataMatch.group('sctms'))*0.001
                scanObjs[id].append(scanObj)
                continue

            match = mergeRE.match(line)
            if(match):
                mergeMatch = match
                remainingVhcl = int(mergeMatch.group('vhcl1'))
                mergedVhcl = int(mergeMatch.group('vhcl2'))
                scanObjs[remainingVhcl] += scanObjs[mergedVhcl]
                del scanObjs[mergedVhcl]

    for key, scanObj in scanObjs.items():
        scanObj.sort(key=lambda x: x['scantime'])

    #remove vehicles without scannerData
    vehicles[:] = [vhcl for vhcl in vehicles if scanObjs[vhcl['id']] != []]
    #actually add the scannerdata now
    for vhcl in vehicles:
        vhcl['scanObjs'] = scanObjs[vhcl['id']]

def parseVehicleFilesInSubDirs(directory=globs.laserScannerOnlyDir):
    """
    expects directory to contains directories parsable with parseVehicleFiles - 
    that is: the usual Daten/ Bilder/ Logs Folders containing the necessary files
    """
    vhcls = []
    for subDir in os.scandir(directory):
        print("reading vehicles in " + subDir.path)
        vhcls += parseVehicleFiles(subDir.path)

    return vhcls

#zentriertes Binomialfilter dritter Ordnung
def binomFilter(scanObjs):
    if(len(scanObjs) == 0):
        return scanObjs
    if(len(scanObjs) == 1):
        scanObjs[0]['smoothHeight'] = scanObjs[0]['height']
        return scanObjs

    for i in range(1,len(scanObjs) - 1):
        scanObjs[i]['smoothHeight'] = 0.25 * scanObjs[i-1]['height'] + 0.5 * scanObjs[i]['height'] + 0.25 * scanObjs[i+1]['height']
    scanObjs[0]['smoothHeight'] = 0.5 * scanObjs[0]['height'] + 0.5 * scanObjs[1]['height']
    scanObjs[-1]['smoothHeight'] = 0.5 * scanObjs[-1]['height'] + 0.5 * scanObjs[-2]['height']
    return scanObjs

def findGap(scanObjs):
    gapThreshold = 0.6 
    gapCandidateIndex = 0
    gapCandidateSmoothHeight = 10000
    gapFound = False
    gapCandidateFound = False
    firstGapPos = 0
    firstGapRelPos = 0
    maxHeightSoFar = 0

    for i in range(len(scanObjs)):
        so = scanObjs[i]

        if(so['smoothHeight'] > maxHeightSoFar):
            maxHeightSoFar = so['smoothHeight']

        if(so['smoothHeight'] + gapThreshold < maxHeightSoFar and so['smoothHeight'] < gapCandidateSmoothHeight and not gapFound):
            gapCandidateIndex = i
            gapCandidateSmoothHeight = so['smoothHeight']
            gapCandidateFound = True

        if(gapCandidateFound and so['smoothHeight'] > gapCandidateSmoothHeight and not gapFound):
            gapFound = True 
            return gapFound

    return gapFound

def addExtractedFeatures(vehicles):
    """
    all features that have to be calculated from parsed data are added here
    for example: sums or differences of other features like axleSpacingsSum
    """

    for vhcl in vehicles: 
        vhcl['axleSpacingsSum'] = 0
        for i in range(int(vhcl['axles']) -1):
            vhcl['axleSpacingsSum'] += vhcl['axleSpacing' + str(i)]
        #scanObjs stuff:
        minTimestamp = vhcl['scanObjs'][0]['scantime']
        maxTimestamp = vhcl['scanObjs'][-1]['scantime']
        duration = maxTimestamp - minTimestamp
        if(maxTimestamp == minTimestamp):
            duration = 1
        #vhcl['minWidth'] = 10000
        vhcl['minSmoothHeight'] = 10000
        vhcl['relPosMinWidth'] = 0
        vhcl['relPosMinSmoothHeight'] = 0
        #vhcl['maxWidth'] = 0
        vhcl['maxSmoothHeight'] = 0
        #vhcl['relPosMaxWidth'] = 0
        vhcl['relPosMaxSmoothHeight'] = 0
        volume = 0

        binomFilter(vhcl['scanObjs'])

        third = int(len(vhcl['scanObjs'])/3)
        vhcl['gapInFirstThird'] = int(findGap(vhcl['scanObjs'][:third]))
        vhcl['gapInSecondThird'] = int(findGap(vhcl['scanObjs'][third:2*third]))


        for i in range(len(vhcl['scanObjs'])):
            so = vhcl['scanObjs'][i]

            volume += so['smoothHeight'] * so['width'] 

            if(so['smoothHeight'] < vhcl['minSmoothHeight']):
                vhcl['relPosMinSmoothHeight'] = (so['scantime'] - minTimestamp)/duration
                vhcl['minSmoothHeight'] = so['smoothHeight']
            #if(so['width'] < vhcl['minWidth']):
            #    vhcl['relPosMinWidth'] = (so['scantime'] - minTimestamp)/duration 
            #    vhcl['minWidth'] = so['width']
            if(so['smoothHeight'] > vhcl['maxSmoothHeight']):
                vhcl['relPosMaxSmoothHeight'] = (so['scantime'] - minTimestamp)/duration
                vhcl['maxSmoothHeight'] = so['smoothHeight']
            #if(so['width'] > vhcl['maxWidth']):
            #    vhcl['relPosMaxWidth'] = (so['scantime'] - minTimestamp)/duration 
            #    vhcl['maxWidth'] = so['width']

        vhcl['volume'] = volume/len(vhcl['scanObjs'])
        vhcl['noScanObjs'] = len(vhcl['scanObjs'])
        vhcl['frontHeight'] = sum([so['height'] for so in vhcl['scanObjs'][:3]])/len(vhcl['scanObjs'][:3])

    return vehicles

        #smoothing
        #Gaps/consolidated gaps
        #consolidated gaps positions
        #consolidated gaps count

        #axleSpacings, vehicleLoopLength = parseWimFile(directory, vf.readline())
        #loopLength makes no difference to knn
        #vehicle['vehicleLoopLength'] = vehicleLoopLength

        #vehicle['axleSpacingsSum'] = 0
        #for i in range(globs.maxAxles - 1):
        #    if(i<len( axleSpacings )):
        #        vehicle['axleSpacing' + str(i)] = axleSpacings[i]
        #        vehicle['axleSpacingsSum'] = vehicle['axleSpacingsSum'] + axleSpacings[i]
        #    else:
        #        vehicle['axleSpacing' + str(i)] = 0
        #overhang worsens knn result
        #vehicle['overhang'] = vehicleLoopLength - vehicle['axleSpacingsSum']
        #numMainAxleSpacings makes no difference to knn
        #vehicle['numMainAxleSpacings'] = 0
        #for spacing in axleSpacings:
        #    if (spacing > 2):
        #        vehicle['numMainAxleSpacings'] += 1
