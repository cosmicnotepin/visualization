from enum import Enum

#directory = r'C:\Users\borgro\svn\java\SensorFusion\Daten\Daten'
#directory = r'D:\Rohdaten\FRV\Daten'
#directory = r'D:\Rohdaten\FREII1\Daten'
directory = r'D:\Rohdaten\FRV_FREII1clean\Daten'

maxAxles = 10

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
