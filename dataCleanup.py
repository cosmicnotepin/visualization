import re
import os

def cleanupData():
    directory = r'D:\Rohdaten\FRV_FREII1clean'
    #directory = r'D:\Rohdaten\FRKO120_07_07'
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
            if(not match):
                delete = True

            #no wim data or no pictures means it is useless
            if(match and ( match.group('wim') == '0' or match.group('scanner') == '0') or match.group('axles') == '1' ): 
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

