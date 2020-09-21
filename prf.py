import cProfile

import parse
import globs

#cProfile.run(r"parse.parseVehicleFilesInSubDirs(globs.laserScannerOnlyDir)", 'running')
cProfile.run(r"parse.addScannerRawData(parse.parseVehicleFiles(globs.laserScannerOnlyDir + r'\1'), globs.laserScannerOnlyDir + r'\1' + r'\Daten\2020-05-25_00-00-00_NKP-FREII1.vtd')", 'running')

import pstats
from pstats import SortKey
p = pstats.Stats('running')
p.sort_stats('tottime').print_stats(30)

