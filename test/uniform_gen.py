import numpy

p = 10

start = 1./(p+1)
end = 1 - start
coords = numpy.linspace(start, end, p)

m = 1./(p**3)

file = open('uniform'+str(p**3), 'w')
idx = 0
for x in coords:
	for y in coords:
		for z in coords:
			file.write(str(idx) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(m) + '\n')
			idx += 1
			
file.close()