import re
import os
from datetime import datetime, timedelta
import contextlib
import itertools
from collections import defaultdict
import struct
import time
import pickle

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
    #TODO parse date from filename
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
                scanObj['scantime'] = 1000 * (int(dataMatch.group('scth'))*3600 + int(dataMatch.group('sctm'))*60 + int(dataMatch.group('scts'))) + int(dataMatch.group('sctms'))# + datetime(2020, 5, 25).timestamp()
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
    expects directory to contain directories parsable with parseVehicleFiles - 
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

    for i, so in enumerate(scanObjs):
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

MAX_ENTRIES = 10
MAX_DELAY = 10000

#4-Byte Markierung am Begin der Datei
VTDFILE_MAGIC = 2754628164
#4-Byte Markierung von Dateien der Version 0
VTDFILE_VERSION_0 = 0x5F005F00
#4-Byte Markierung von Dateien der Version 1
VTDFILE_VERSION_1 = 0x5F005F01
#4-Byte Markierung von Dateien der Version 2
VTDFILE_VERSION_2 = 0x5F005F02
#4-Byte Markierung vor jedem Objekt
VTDOBJ_MAGIC = 569825642

#Maximale Anzahl von Meta-Informationen
NTDFILE_META_MAX = 20
#Maximale Länge eines Schlüsselnames
NTDFILE_META_KEY_MAX = 50
#Maximale Länge eines Wertes
NTDFILE_META_VALUE_MAX = 100

VTDFILE_OBJHEADER_SIZE = 20
VTDFILE_SCANNERDATA_SIZE = 28 + (274 * 2)

#extracts on Scannerdata set from the vtdFile and returns it as a dictionary
def readScannerData(vtdFile):
    scannerData = {}
    objHeader = vtdFile.read(VTDFILE_OBJHEADER_SIZE)
    magic = int.from_bytes(objHeader[:4], 'little', signed=False)
    if(magic != VTDOBJ_MAGIC):
        return None
    objId = int.from_bytes(objHeader[4:8], 'little', signed=False)
    #[8:12] #vtdWriter.addScannerData time, not scantime
    #[12:16]
    length = int.from_bytes(objHeader[16:20], 'little', signed=False)
    if(length != VTDFILE_OBJHEADER_SIZE + VTDFILE_SCANNERDATA_SIZE - 4):
        return None
    data = vtdFile.read(VTDFILE_SCANNERDATA_SIZE)
    scannerId = int.from_bytes(data[:4], 'little', signed=False)
    scannerData['id'] = scannerId
    dataItems = int.from_bytes(data[4:8], 'little', signed=False)
    dists = []
    for i in range(dataItems):
        dists.append(int.from_bytes(data[8+(i*2):10+(i*2)], 'little', signed=False))
    scannerData['dists'] = dists
    index = 10 + (dataItems-1)*2
    startAngle = int.from_bytes(data[index:index+4], 'little', signed=False)
    index += 4
    endAngle = int.from_bytes(data[index:index+4], 'little', signed=False)
    index += 4
    vertAngle, = struct.unpack('<f', data[index:index+4])
    scannerData['vertAngle'] = vertAngle
    index += 4
    time1 = int.from_bytes(data[index:index+4], 'little', signed=False)
    index += 4
    time2 = int.from_bytes(data[index:index+4], 'little', signed=False)
    index += 4
    timestamp = int(time1*1000 + time2/1000)
    dt = datetime.fromtimestamp(timestamp/1000) #this is not 'utcfromtimestamp' for a reason!
    scannerData['scantime'] =  1000*(dt.hour*3600 + dt.minute*60 + dt.second) + dt.microsecond/1000
    return scannerData


def addScannerRawData(vhcls, vtdFilePath):
    scanTime2Vhcl = defaultdict(lambda : list([1]))
    counter = 0
    allCounter = 0
    minScanTime = 99999999
    maxScanTime = 0
    for vhcl in vhcls:
        for so in vhcl['scanObjs']:
            minScanTime = min(minScanTime, so['scantime'])
            maxScanTime = max(maxScanTime, so['scantime'])
            allCounter += 1
            if(so['scantime'] in scanTime2Vhcl):
                counter += 1
            scanTime2Vhcl[so['scantime']].append(so)

    start = time.time()
    with open(vtdFilePath, 'rb') as vtd:
        fileHeader = vtd.read(12 + NTDFILE_META_MAX * (NTDFILE_META_KEY_MAX + NTDFILE_META_VALUE_MAX))
        magic = int.from_bytes(fileHeader[:4], 'little', signed=False)
        version = int.from_bytes(fileHeader[4:8], 'little', signed=False)
        scannerData = None
        noScanObjsForRaw = 0
        while(scannerData := readScannerData(vtd)):
            if(scannerData['scantime'] > maxScanTime):
                break
            allSos = scanTime2Vhcl[scannerData['scantime']]
            if(allSos[0] == len(allSos)):
                noScanObjsForRaw += 1
                continue
            allSos[allSos[0]]['rawData'] = scannerData
            allSos[0] += 1
        print(f"noScanObjsForRaw: {noScanObjsForRaw}")

    print("it took: " + str(int(((time.time() - start)/60))) + ':' + str((time.time() - start)%60))
    counter = 0
    allCounter = 0
    for vhcl in vhcls:
        for so in vhcl['scanObjs']:
            allCounter +=1
            if(not 'rawData' in so):
                counter +=1
    print(f"no rawData counter = {counter} of {allCounter}")
    return vhcls

def pickleDumpSome():
    vhcls = addScannerRawData(parseVehicleFiles(globs.laserScannerOnlyDir + r'\1'), globs.laserScannerOnlyDir + r'\1' + r'\Daten\2020-05-25_00-00-00_NKP-FREII1.vtd')
    interestingIds = ['314970','314901', '315234', '318647', '320420']
    #314970 ist normaler pkw, rest ist seltsame lkw
    toPickle = {}
    for vhcl in vhcls:
        if(vhcl['id'] in interestingIds):
            toPickle[vhcl['id']] = vhcl
    with open('InterVhcls.pickle', 'wb') as f:
        pickle.dump(toPickle, f, pickle.HIGHEST_PROTOCOL)

def pickleDumpAll():
    #vhcls = addExtractedFeatures(parseVehicleFiles(globs.laserScannerOnlyDir + r'\1'))
    #vhcls = addScannerRawData(vhcls, globs.laserScannerOnlyDir + r'\1' + r'\Daten\2020-05-25_00-00-00_NKP-FREII1.vtd')
    vhcls = addExtractedFeatures(parseVehicleFiles(globs.laserScannerOnlyDir + r'\3'))
    vhcls = addScannerRawData(vhcls, globs.laserScannerOnlyDir + r'\3' + r'\Daten\2020-06-02_00-15-07_NKP-FRV.vtd')
    with open('all2.pickle', 'wb') as f:
        pickle.dump(vhcls, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    pickleDumpAll()
