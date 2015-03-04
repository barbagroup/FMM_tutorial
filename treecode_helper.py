import numpy
from matplotlib import pyplot, rcParams
from mpl_toolkits.mplot3d import Axes3D

# customizing plot parameters
rcParams['figure.dpi'] = 100
rcParams['font.size'] = 14
rcParams['font.family'] = 'StixGeneral'


class Particle():
    """The class for a particle.
    
    Arguments:
        n_source: the total number of source particles
        range: the range of random coordinates x,y,z, default=1
    
    Attributes:
        x, y, z: coordinates of the particle
        m: mass of the particle
        phi: the gravitational potential of the particle
        
    """
    def __init__(self, n_source, range=1):
        """Initialize the particle with random coordinates in (0, range) or
        (range, 0), a uniform mass depending on n_source, and a zero potential.
        
        """
        self.x = range * numpy.random.random()
        self.y = range * numpy.random.random()
        self.z = range * numpy.random.random()
        self.m = 1.0/n_source
        self.phi = 0.
        
    def distance(self, other):
        """Return the distance between two particles.
        
        """
        return numpy.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 
                                              + (self.z-other.z)**2)


def direct_sum(sources, targets):
    """Calculate the gravitational potential (target.phi) at each target 
    particle using direct summation method.

    Arguments:
        sources: the list of source objects in 'Particle' class
        targets: the list of target objects in 'Particle' class

    """
    for target in targets:
        for source in sources:
            r = target.distance(source)
            target.phi += source.m/r


def distance(array, point):
    """Return the distance array between each element in the array and
    the point.
    
    Arguments:
        array: an array of n points' xyz coordinates with a shape of (3, n)
        point: a xyz-coordinate triplet of the point 
        
    Returns:
        the distance array
        
    """
    return numpy.sqrt((array[0]-point.x)**2 + (array[1]-point.y)**2
                                            + (array[2]-point.z)**2)


def eval_potential(targets, multipole, center):
    """Given targets list, multipole and expansion center, return
    the array of target's potentials.
    
    Arguments:
        targets: the list of target objects in 'Particle' class
        multipole: the multipole array of the cell
        center: the point object of expansion center
    
    Returns:
        phi: the potential array of targets
        
    """
    # prepare for array operation
    target_x = numpy.array([target.x for target in targets])
    target_y = numpy.array([target.y for target in targets])
    target_z = numpy.array([target.z for target in targets])
    target_array = [target_x, target_y, target_z]
    
    # calculate the distance between each target and center
    r = distance(target_array, center)
    
    # prearrange some constants for weight
    dx, dy, dz = target_x-center.x, target_y-center.y, target_z-center.z
    r3 = r**3
    r5 = r3*r**2
    
    # calculate the weight for each multipole
    weight = [1/r, -dx/r3, -dy/r3, -dz/r3, 3*dx**2/r5 - 1/r3, \
              3*dy**2/r5 - 1/r3, 3*dz**2/r5 - 1/r3, 3*dx*dy/r5, \
              3*dy*dz/r5, 3*dz*dx/r5]
    
    # evaluate potential
    phi = numpy.dot(multipole, weight)
    return phi


def l2_err(phi_direct, phi_multi):
    """Print out the relative err in l2 norm.
    
    """
    err = numpy.sqrt(sum((phi_direct-phi_multi)**2)/sum(phi_direct**2))
    print 'L2 Norm error: {}'.format(err)


def plot_err(phi_direct, phi_multi): 
    """Plot the relative error band. 
    
    """
    # plotting the relative error band
    n = len(phi_direct)
    err_rel = abs((phi_multi - phi_direct) / phi_direct)
    pyplot.figure(figsize=(10,4))
    ax = pyplot.gca()
    pyplot.plot(range(n), err_rel, 'bo', alpha=0.5)
    pyplot.xlim(0,n-1)
    pyplot.ylim(1e-6, 1e-1)
    ax.yaxis.grid()
    pyplot.xlabel('target particle index')
    pyplot.ylabel(r'$e_{\phi rel}$')
    ax.set_yscale('log')