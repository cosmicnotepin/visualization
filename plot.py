from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
from pandas import *

import globs
import parse

class2Marker = { "nicht_klass_Kfz":dict(marker='x', c='black'), "Motorrad":dict(marker='4', c='yellow'), "PKW":dict(marker='o', c='blue'), "Kleintransporter":dict(marker='>', c='green'), "PKW_Anhaenger":dict(marker='s', c='cyan'), "LKW":dict(marker='+', c='red'), "LKW_Anhaenger":dict(marker='P', c='magenta'), "Sattel_Kfz":dict(marker='*', c='brown'), "Bus":dict(marker='_', c='DarkGoldenRod'), "ungueltig":dict(marker='None', c='DarkGoldenRod')  }

def plotVehiclesTest(vehicles, xDim='length', yDim='weight', zDim=None, classes=class2Marker.keys()):
    """
    uses matplotlib to plot vehicles
    x/y/zDim are the mappings from vehicle fields to axis of the plot
    classes allows to only show specific classes
    you can define new features in this function an set it to be one of the plot axis of course
    the vehicles are parsed into pandas Dataframes to enable some comfortable data manipulation.
    """

    minAxleDistance = 2 
    #vehicles = vehicles[19:20]

    #weight of second main axis (all axis between the first and second main axis spacing)
    #length of second main axis spacing
    for vehicle in vehicles:
        foundSecondMain = False
        for i in range(0, globs.maxAxles - 2):
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

    #difference axle weights
    #df['custom'] = abs(df['axleWeight0'] - df['axleWeight1'])

    #count main axle spacings (spacings that are not between double-axis
    df['mainSpacingCount'] = (df['axleSpacing0'] > minAxleDistance).astype(int)
    for i in range(1,globs.maxAxles-1):
        df['mainSpacingCount'] += (df['axleSpacing' + str(i)] > minAxleDistance).astype(int)

    #filter by axis count
    #df = df[df['axles'] > 2]

    #filter by first spacing
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
