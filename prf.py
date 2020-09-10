import cProfile

import parse
import globs

cProfile.run(r"parse.parseVehicleFilesInSubDirs(globs.laserScannerOnlyDir)", 'running')

import pstats
from pstats import SortKey
p = pstats.Stats('running')
p.sort_stats('tottime').print_stats(30)

