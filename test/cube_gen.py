from numpy import random

n = 10000

coords = random.rand(n,3)
m = 1./n

fid = open('cube'+str(n),'w')
for idx in range(n):
    fid.write(str(idx) + str(coords[idx,:])[1:-1] + ' ' + str(m) + '\n')
    
fid.close()
