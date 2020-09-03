import cProfile

import parse
import globs

cProfile.run(r"parse.parseVehicleFiles(globs.laserScannerOnlyDir + r'\1')", 'running')

import pstats
from pstats import SortKey
p = pstats.Stats('running')
p.sort_stats(SortKey.CUMULATIVE).print_stats(10)

