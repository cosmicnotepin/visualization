from enum import Enum
from collections import defaultdict

#directory = r'C:\Users\borgro\svn\java\SensorFusion\Daten\Daten'
#directory = r'D:\Rohdaten\FRV\Daten'
#directory = r'D:\Rohdaten\FREII1\Daten'
directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'
laserScannerOnlyDir = r'D:\Rohdaten\LaserScannerOnly'

maxAxles = 10
averageSpeed = 11/1000 # 40 km/h in m/ms roughly

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

labels = [cls.name for cls in vClass if cls.value in range(1, 9)]

#map 8+1 to 8+1 (Identity)
def AchtPlus1To8Plus1(cls):
    return cls

#map 8+1 to 2
def AchtPlus1To2(cls):
    retDict = defaultdict(lambda : 'PKW', {
        'nicht_klass_Kfz' : 'PKW',
        'Motorrad' : 'PKW',
        'PKW' : 'PKW',
        'Kleintransporter' : 'PKW',
        'PKW_Anhaenger' : 'LKW',
        'LKW' : 'LKW',
        'LKW_Anhaenger' : 'LKW',
        'Sattel_Kfz' : 'LKW',
        'Bus' : 'LKW',
        'ungueltig' : 'PKW',
        'fraglich' : 'PKW'
        })

    return retDict[cls]
