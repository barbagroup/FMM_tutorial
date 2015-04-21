# -*- coding: utf-8 -*-

import numpy
from treecode_helper import *

filename = 'ellipsoid10000'
particles = read_particle(filename)

direct_sum(particles)

phi_direct = numpy.asarray([particle.phi for particle in particles])

write_result(phi_direct, filename + '_result')