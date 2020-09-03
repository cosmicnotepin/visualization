from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

import parse
import globs

def plotDebugLog(path=r'D:\Rohdaten\FRV_FREII1clean\Logs\2020-05-25_00-00-00_NKP-FREII1-debug.log'):
    scanObjs = parse.parseDebugLog(path)
    request = int(input('id to display: '))
    while(request != 0):
        vhcl = scanObjs[request]
        plotVehicle(vhcl)
        request = int(input('id to display: '))

def plotVehicle(vhcl):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_ylim(bottom=0, top=30)
    ax.set_zlim(bottom=0, top=30)
    ax.set_xlim(left=0, right=30)
    for scanObj in vhcl:
        rect = Rectangle([scanObj['min_y'], 0], scanObj['width'], scanObj['height'], fill=False)
        ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=-1*(scanObj['scantime'] - vhcl[0]['scantime'])*11, zdir="x") # 11 ~ 40km/h in m/s
    plt.show()
    #plt.draw()
    #plt.pause(0.001)

def plotProfileAt(ax, vhcl, x, y):
    stretch = 11 # assuming ~ 40 km/h ~ 11 m/s
    startTime = vhcl['scanObjs'][0]['scantime']
    endTime = vhcl['scanObjs'][-1]['scantime']
    startWidth = 1000
    for scanObj in vhcl['scanObjs']:
        if(scanObj['min_y'] < startWidth):
            startWidth = scanObj['min_y']

    for scanObj in vhcl['scanObjs']:
        lineX = x + (scanObj['scantime'] - startTime)*stretch
        lineH = mlines.Line2D([lineX, lineX], [y, y + scanObj['height']], lw=1., alpha=0.5)
        widthOffset = scanObj['min_y'] - startWidth
        lineW = mlines.Line2D([lineX, lineX], [y + 6 + widthOffset, y + 6 + widthOffset + scanObj['width']], lw=1., alpha=0.5, color='green')
        ax.add_line(lineH)
        ax.add_line(lineW)
    minHLineX = x + (endTime - startTime) * vhcl['relPosMinHeight'] * stretch
    minWLineX = x + (endTime - startTime) * vhcl['relPosMinWidth'] * stretch
    lineMH = mlines.Line2D([minHLineX, minHLineX], [y + 4, y + 5], lw=1., alpha=1, color='purple')
    lineMW = mlines.Line2D([minWLineX, minWLineX], [y + 4 + 6, y + 5 + 6], lw=1., alpha=1, color='red')
    ax.add_line(lineMH)
    ax.add_line(lineMW)

def plotStuff():
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(bottom=0, top=96)
    ax.set_xlim(left=0, right=140)
    ax.set_yticks(range(6,102,12))
    ax.set_yticklabels(reversed([cls.name for cls in globs.vClass if cls.value in range(1,9)]))

    yPositions = [-1000, 7*12, 6*12, 5*12, 4*12, 3*12, 2*12, 1*12, 0*12, -1000, -1000]
    vhcls = parse.parseVehicleFiles(globs.laserScannerOnlyDir + r'\1')
    parse.addExtractedFeatures(vhcls)
    vhclsByClass = []
    for cls in globs.vClass:
        vhclsByClass.append([])
    for vhcl in vhcls:
        vhclsByClass[globs.vClass[vhcl['class']].value].append(vhcl)
    for cls in globs.vClass:
        for i in range(min(30, len(vhclsByClass[cls.value]))):
            plotProfileAt(ax,vhclsByClass[cls.value][i], i*30, yPositions[cls.value])

    plt.show()

plotStuff()
