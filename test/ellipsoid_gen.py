from numpy import random

n = 10000

m = 1./n

idx = 0
fid = open('ellipsoid'+str(n),'w')
while idx < n:
    coords = random.rand(3)
    if (coords[0]-0.5)**2/0.25 + (coords[1]-0.5)**2/0.09 + (coords[2]-0.5)**2/0.04 <= 1:
        fid.write(str(idx) + str(coords[:])[1:-1] + ' ' + str(m) + '\n')
        idx += 1    
    
fid.close()