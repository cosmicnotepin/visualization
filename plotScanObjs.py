from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import time
import math
import pickle

import parse
import globs

yPositions = [-1000, 7*12, 6*12, 5*12, 4*12, 3*12, 2*12, 1*12, 0*12, -1000, -1000]
allArtists = []

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
        art3d.pathpatch_2d_to_3d(rect, z=-1*(scanObj['scantime'] - vhcl[0]['scantime'])*globs.averageSpeed, zdir="x")
    plt.show()

def plotProfileAt(ax, vhcl, x, y):
    startTime = vhcl['scanObjs'][0]['scantime']
    endTime = vhcl['scanObjs'][-1]['scantime']
    startWidth = 1000
    for scanObj in vhcl['scanObjs']:
        startWidth = min(startWidth, scanObj['min_y'])

    allArtists.append(ax.text(x, y, vhcl['aClass']))
    for scanObj in vhcl['scanObjs']:
        lineX = x + (scanObj['scantime'] - startTime)*globs.averageSpeed
        lineH = mlines.Line2D([lineX, lineX], [y, y + scanObj['height']], lw=1., alpha=0.5)
        widthOffset = scanObj['min_y'] - startWidth
        lineW = mlines.Line2D([lineX, lineX], [y + 6 + widthOffset, y + 6 + widthOffset + scanObj['width']], lw=1., alpha=0.5, color='green')
        #lineW = mlines.Line2D([lineX, lineX], [y + 6, y + 6 + scanObj['smoothHeight']], lw=1., alpha=0.5, color='red')
        allArtists.append(lineH)
        allArtists.append(lineW)
        ax.add_line(lineH)
        ax.add_line(lineW)
    #minHLineX = x + (endTime - startTime) * vhcl['relPosMinHeight'] * globs.averageSpeed
    #lineMH = mlines.Line2D([minHLineX, minHLineX], [y + 4, y + 5], lw=1., alpha=1, color='purple')
    #ax.add_line(lineMH)
    #minWLineX = x + (endTime - startTime) * vhcl['firstGapRelPos'] * globs.averageSpeed
    #lineMW = mlines.Line2D([minWLineX, minWLineX], [y + 4 + 6, y + 5 + 6], lw=1., alpha=1, color='red')
    #ax.add_line(lineMW)
    minWLineX = x
    lineMW = mlines.Line2D([minWLineX, minWLineX], [y + 4 + 6, y + 5 + 6], lw=1., alpha=1, color='red')
    if(vhcl['gapInFirstThird']):
        allArtists.append(lineMW)
        ax.add_line(lineMW)
    minWLineX += 1
    lineMW = mlines.Line2D([minWLineX, minWLineX], [y + 4 + 6, y + 5 + 6], lw=1., alpha=1, color='green')
    if(vhcl['gapInSecondThird']):
        allArtists.append(lineMW)
        ax.add_line(lineMW)

def plotVhclsScanObjs(vhcls, brokenOnly=False):
    global allArtists
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_ylim(bottom=0, top=96)
    ax1.set_xlim(left=0, right=140)
    ax1.set_yticks(range(6,102,12))
    ax1.set_yticklabels(reversed([cls.name for cls in globs.vClass if cls.value in range(1,9)]))
    for artist in allArtists:
        artist.remove()
    allArtists = []
    vhclsByClass = []
    for cls in globs.vClass:
        vhclsByClass.append([])
    for vhcl in vhcls:
        vhclsByClass[globs.vClass[vhcl['class']].value].append(vhcl)
    for cls in globs.vClass:
        plotted = 0
        for vhcl in vhclsByClass[cls.value]:
            if(vhcl['class'] != vhcl['aClass'] or not brokenOnly): 
                plotProfileAt(ax1, vhcl, plotted*30, yPositions[cls.value])
                plotted += 1
            if(plotted > 30):
                break

    plt.draw()
    plt.pause(0.1)

SCAN_MAX_NUM_VALUES = 274
deltaAngle = math.radians(96.3281/(SCAN_MAX_NUM_VALUES-1))
scannerXOffset = 550
#
#scannerDist = 3550
#scannerHeight0 = 5170
#scannerHorizAngle0 = math.radians(-1)
#scannerHeight1 = 5260
#scannerHorizAngle1 = math.radians(-5)
scannerDataArtists = []

def plotScannerData(scannerData, fig, ax, scannerDist, scannerHeight0, scannerHeight1, scannerHorizAngle0, scannerHorizAngle1):
    global scannerDataArtists
    scannerSin0 = [math.sin(i*deltaAngle + scannerHorizAngle0) for i in range(SCAN_MAX_NUM_VALUES)]
    scannerCos0 = [math.cos(i*deltaAngle + scannerHorizAngle0) for i in range(SCAN_MAX_NUM_VALUES)]
    scannerSin1 = [math.sin(i*deltaAngle + scannerHorizAngle1) for i in range(SCAN_MAX_NUM_VALUES)]
    scannerCos1 = [math.cos(i*deltaAngle + scannerHorizAngle1) for i in range(SCAN_MAX_NUM_VALUES)]

    for artist in scannerDataArtists:
        artist.remove()
    scannerDataArtists = []
    x = []
    y = []
    color = 'red'
    for i in range(SCAN_MAX_NUM_VALUES):
        if('rawData' in scannerData):
            rawData = scannerData['rawData']
            if(rawData['id'] == 0):
                color = 'blue'
                x.append(scannerXOffset + rawData['dists'][i] * scannerSin0[i])
                y.append(scannerHeight0 - (rawData['dists'][i] * scannerCos0[i]))
            else:
                x.append(scannerXOffset + scannerDist - rawData['dists'][i] * scannerSin1[i])
                y.append(scannerHeight1 - (rawData['dists'][i] * scannerCos1[i]))

    factor = 1000
    rect = Rectangle([factor * scannerData['min_y'], 0], factor * scannerData['width'], factor * scannerData['height'], fill=False)
    ax.add_patch(rect)
    scannerDataArtists.append(rect)
    scannerDataArtists.append(ax.scatter(x,y, s=0.5, c=color))
    fig.canvas.draw()

keyPressed = False
lastKey = ''
keyToIndexChange = {'left' : -1, 'right' : 1}
def onKey(event):
    global keyPressed, lastKey
    keyPressed = True
    lastKey = event.key

def initPlt():
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', onKey)
    ax.set_ylim(bottom=-15000, top=15000)
    ax.set_xlim(left=-15000, right=15000)
    return fig, ax

currentId = None
def setVhclIdToPlot(id):
    global currentId
    currentId = id

def navigateCurrentScanObjs(pickleFile, scannerDist, scannerHeight0, scannerHeight1, scannerHorizAngle0, scannerHorizAngle1):
    global currentId
    fig, ax = initPlt()
    allVhcls = None 
    with open(pickleFile, 'rb') as f:
        allVhcls = pickle.load(f)
    while(1):
        for v in allVhcls:
            if(v['id'] == currentId):
                navigateScanObjsAndWaitForSignal(v, fig, ax, scannerDist, scannerHeight0, scannerHeight1, scannerHorizAngle0, scannerHorizAngle1)
                break

def plotFullProfiles(vhcl, i, fig, ax):
    global allArtists
    x = -14000
    y = -14000
    scaling = 1000
    for artist in allArtists:
        artist.remove()
    allArtists = []
    startTime = vhcl['scanObjs'][0]['scantime']
    endTime = vhcl['scanObjs'][-1]['scantime']
    widthStart = 1000
    for scanObj in vhcl['scanObjs']:
        widthStart = min(widthStart, scanObj['min_y'])

    for ci, scanObj in enumerate(vhcl['scanObjs']):
        colorH = 'red' if ci == i else 'blue'
        colorW = 'red' if ci == i else 'green'
        lwH = 2. if ci == i else 1
        lwW = 2. if ci == i else 1
        lineX = x + (scanObj['scantime'] - startTime) * globs.averageSpeed * scaling
        lineH = mlines.Line2D([lineX, lineX], [y, y + scanObj['height'] * scaling], lw=lwH, alpha=0.5, color=colorH)
        widthOffset = (scanObj['min_y'] - widthStart) * scaling
        lineW = mlines.Line2D([lineX, lineX], [y + 6 * scaling + widthOffset, y + 6 * scaling + widthOffset + scanObj['width'] * scaling], lw=lwW, alpha=0.5, color=colorW)
        allArtists.append(lineH)
        allArtists.append(lineW)
        ax.add_line(lineH)
        ax.add_line(lineW)

    #in first third
    gapXs = [-10000, -9000]
    gapY = -1500
    colorGapFt = 'red' if vhcl['gapInFirstThird'] else 'blue'
    colorGapSt = 'red' if vhcl['gapInSecondThird'] else 'blue'
    lineFt = mlines.Line2D([gapXs[0], gapXs[0]], [gapY, gapY + 1000], lw=lwH, alpha=0.5, color=colorGapFt)
    ax.add_line(lineFt)
    lineSt = mlines.Line2D([gapXs[1], gapXs[1]], [gapY, gapY + 1000], lw=lwH, alpha=0.5, color=colorGapSt)
    ax.add_line(lineSt)
    allArtists.append(lineFt)
    allArtists.append(lineSt)

def plotVhclsScanObjs(vhcls, brokenOnly=False):
    global allArtists
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_ylim(bottom=0, top=96)
    ax1.set_xlim(left=0, right=140)
    ax1.set_yticks(range(6,102,12))
    ax1.set_yticklabels(reversed([cls.name for cls in globs.vClass if cls.value in range(1,9)]))
    for artist in allArtists:
        artist.remove()
    allArtists = []
    vhclsByClass = []
    for cls in globs.vClass:
        vhclsByClass.append([])
    for vhcl in vhcls:
        vhclsByClass[globs.vClass[vhcl['class']].value].append(vhcl)
    for cls in globs.vClass:
        plotted = 0
        for vhcl in vhclsByClass[cls.value]:
            if(vhcl['class'] != vhcl['aClass'] or not brokenOnly): 
                plotProfileAt(ax1, vhcl, plotted*30, yPositions[cls.value])
                plotted += 1
            if(plotted > 30):
                break

    plt.draw()
    plt.pause(0.1)

def navigateScanObjsAndWaitForSignal(vhcl, fig, ax, scannerDist, scannerHeight0, scannerHeight1, scannerHorizAngle0, scannerHorizAngle1):
    global keyPressed, lastKey, currentId
    scanObjs = vhcl['scanObjs']
    i = 0
    lastId = currentId
    while(lastId == currentId):
        so = scanObjs[i]
        plotScannerData(so, fig, ax, scannerDist, scannerHeight0, scannerHeight1, scannerHorizAngle0, scannerHorizAngle1)
        plotFullProfiles(vhcl, i, fig, ax)
        while(not keyPressed and lastId == currentId):
            plt.pause(0.1)
        keyPressed = False
        try:
            i = min(len(scanObjs) - 1, max(0, i + keyToIndexChange[lastKey]))
        except KeyError:
            i = i

def navigateScanObjs(scanObjs, fig, ax):
    global keyPressed, lastKey
    i = 0
    while(True):
        so = scanObjs[i]
        plotScannerData(so, fig, ax)
        while(not keyPressed):
            plt.pause(0.1)
        keyPressed = False
        try:
            i = min(len(scanObjs) - 1, max(0, i + keyToIndexChange[lastKey]))
        except KeyError:
            return

def plotScannerRawOfVhcl(request):
    fig, ax = initPlt()
    with open('InterVhcls.pickle', 'rb') as f:
        interVhcls = pickle.load(f)
    if(not request):
        request = input('id to display: ')
    while(request != '0'):
        vhcl = interVhcls[request]
        navigateScanObjs(vhcl['scanObjs'], fig, ax)
        request = input('id to display: ')

def justPlotAll():
    fig, ax = initPlt()
    with open('all.pickle', 'rb') as f:
        allVhcls = pickle.load(f)
    for vhcl in allVhcls:
        navigateScanObjs(vhcl['scanObjs'], fig, ax)

def scannerIdTest():
    with open('InterVhcls.pickle', 'rb') as f:
        interVhcls = pickle.load(f)
        counts = [0,0,0,0,0]
        for id, vhcl in interVhcls.items():
            for so in vhcl['scanObjs']: 
                if('rawData' in so):
                    counts[so['rawData']['id']] += 1 
        print(counts)

