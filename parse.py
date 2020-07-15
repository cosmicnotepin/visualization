import re
import os
from datetime import datetime
import contextlib

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
                pattern = '.*? length=(?P<length>.*?) weight=(?P<weight>.*?) maxAxleWeight=(?P<maweight>.*?) axles=(?P<axles>.*?) axleWeights=(?P<axleWeights>.*?) .*? numWimFiles=(?P<wim>[0-9]+) numAnprPics=(?P<anpr>[0-9]+) numIpCamPics=(?P<cam>[0-9]+) numScannerPics=(?P<scanner>[0-9]+)'
                match = re.match(pattern, line)
                if(match.group('length') == '-1,00'):
                    continue #invalid vehicle
                vehicle['length'] = float(match.group('length').replace(',', '.'))
                vehicle['axles'] = float(match.group('axles').replace(',', '.'))
                vehicle['weight'] = float(match.group('weight').replace(',', '.'))
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
                vehicle['vehicleLoopLength'] = vehicleLoopLength
                vehicle['axleSpacingsSum'] = 0
                for i in range(globs.maxAxles - 1):
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

