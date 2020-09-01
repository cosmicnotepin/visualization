from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import parse

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

plotDebugLog()
