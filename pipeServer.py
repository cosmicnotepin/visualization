import time
import sys
import win32pipe, win32file, pywintypes
import parse
import plotScanObjs
import globs
import matplotlib.pyplot as plt
import threading
import math

def pipeListener():
    pipe = win32pipe.CreateNamedPipe(
        r'\\.\pipe\Foo',
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
        1, 65536, 65536,
        0,
        None)

    try:
        print("waiting for client")
        win32pipe.ConnectNamedPipe(pipe, None)
        print("got client")
        while(1):
            resp = win32file.ReadFile(pipe, 64*1024)
            id = resp[1].decode()
            print(id)
            plotScanObjs.setVhclIdToPlot(id)
        print("finished now")
    finally:
        win32file.CloseHandle(pipe)

def pipeServerRaw(pickleFile, scannerDist, scannerHeight0, scannerHeight1, scannerHorizAngle0, scannerHorizAngle1):
    print("pipe server")
    listenerThread = threading.Thread(target=pipeListener)
    listenerThread.start()
    print(f"reading vhcl info from pickle ({pickleFile})")
    plotScanObjs.navigateCurrentScanObjs(pickleFile, scannerDist, scannerHeight0, scannerHeight1, scannerHorizAngle0, scannerHorizAngle1)

def pipeServerFrontSide():
    print("pipe server")
    vhcls = parse.parseVehicleFiles(globs.laserScannerOnlyDir + r'\1')
    vhcls = parse.addExtractedFeatures(vhcls)
    count = 0
    pipe = win32pipe.CreateNamedPipe(
        r'\\.\pipe\Foo',
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
        1, 65536, 65536,
        0,
        None)
    try:
        print("waiting for client")
        win32pipe.ConnectNamedPipe(pipe, None)
        print("got client")
        while(1):
            while(1):
                plt.pause(0.1)
                respP = win32pipe.PeekNamedPipe(pipe, 64*1024)
                if(respP[1] != 0):
                    break
            resp = win32file.ReadFile(pipe, 64*1024)
            id = resp[1].decode()
            print(id)
            for i, vhcl in enumerate(vhcls):
                if(vhcl['id'] == id):
                    plotScanObjs.plotVhclScanObjs(vhcls[i:i+1])
                    break

        #while count < 10:
        #    print(f"writing message {count}")
        #    # convert to bytes
        #    some_data = str.encode(f"{count}\n")
        #    #win32file.WriteFile(pipe, some_data)
        #    time.sleep(1)
        #    count += 1

        print("finished now")
    finally:
        win32file.CloseHandle(pipe)

if __name__ == '__main__':
#    pipeServerRaw(pickleFile='all.pickle', scannerDist=3550, scannerHeight0=5170, scannerHeight1=5260, scannerHorizAngle0 = math.radians(-1), scannerHorizAngle1=math.radians(-5))
    pipeServerRaw(pickleFile='all2.pickle', scannerDist=3550, scannerHeight0=5170, scannerHeight1=5260, scannerHorizAngle0 = math.radians(-1), scannerHorizAngle1=math.radians(-5))

#def pipe_client():
#    print("pipe client")
#    quit = False
#
#    while not quit:
#        try:
#            handle = win32file.CreateFile(
#                r'\\.\pipe\Foo',
#                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
#                0,
#                None,
#                win32file.OPEN_EXISTING,
#                0,
#                None
#            )
#            res = win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
#            if res == 0:
#                print(f"SetNamedPipeHandleState return code: {res}")
#            while True:
#                resp = win32file.ReadFile(handle, 64*1024)
#                print(f"message: {resp}")
#        except pywintypes.error as e:
#            if e.args[0] == 2:
#                print("no pipe, trying again in a sec")
#                time.sleep(1)
#            elif e.args[0] == 109:
#                print("broken pipe, bye bye")
#                quit = True


