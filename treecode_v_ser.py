# -*- coding: utf-8 -*-
import numpy
import time
from treecode_helper import *

filename = 'cube1000'
particles = read_particle(filename)

n_crit = 10
theta = 0.2

# build tree
tic = time.time()
root = Cell(n_crit)
root.x, root.y, root.z = 0.5, 0.5, 0.5
root.r = 0.5
cells = build_tree(particles, root, n_crit)
toc = time.time()
t_src = toc - tic

# P2M: particle to multipole
tic = time.time()
leaves = []
get_multipole(particles, 0, cells, leaves, n_crit)
toc = time.time()
t_P2M = toc - tic

# M2M
tic = time.time()
upward_sweep(cells)
toc = time.time()
t_M2M = toc - tic

# evaluate the potentials
tic = time.time()
eval_potential_v(particles, cells, leaves, n_crit, theta)
toc = time.time()
t_eval = toc -tic

# print info
print filename + '-serial' + '-vectorized'
print "N = %i" % len(particles)
print "n_crit = %i" % n_crit
print "theta = %.2f" % theta

# calculate the error
phi_direct = numpy.loadtxt('test/'+filename+'_result')
phi_tree = numpy.asarray([particle.phi for particle in particles])
l2_err(phi_direct, phi_tree)

# print benchmark
t_tree = t_src + t_P2M + t_M2M + t_eval
print "time elapsed: %f s" % t_tree
print "build tree: %f, %.2f %%" % (t_src, t_src/t_tree)
print "P2M       : %f, %.2f %%" % (t_P2M, t_P2M/t_tree)
print "M2M       : %f, %.2f %%" % (t_M2M, t_M2M/t_tree)
print "eval phi  : %f. %.2f %%" % (t_eval, t_eval/t_tree)
