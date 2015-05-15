# -*- coding: utf-8 -*-
# direct summation: serial
import sys
import numpy
import time
from treecode_helper import *

assert (len(sys.argv) == 2), "The format should be \n [script.py] [filename]"

filename = sys.argv[1]
particles = read_particle(filename)

tic = time.time()
direct_sum(particles)
toc = time.time()
t_direct = toc - tic

# print info
print(filename + '-serial' + '-direct-summation')
print(len(filename + '-serial' + '-direct-summation')*'-')
print("N = %i" % len(particles))
print(28*'-')
print("time elapsed: %f s" % t_direct)

# option to write the result
#phi_direct = numpy.asarray([particle.phi for particle in particles])
#write_result(phi_direct, filename + '_result')