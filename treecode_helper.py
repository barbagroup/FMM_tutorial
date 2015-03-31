import numpy
from matplotlib import pyplot, rcParams
from mpl_toolkits.mplot3d import Axes3D

class Point():
    """The class for a point.
    
    Arguments:
        coords: a three-element list, containing the 3d coordinates of the point
        domain: the domain of random generated coordinates x,y,z, default=1.0
    
    Attributes:
        x, y, z: coordinates of the point
    """
    def __init__(self, coords=[], domain=1.0):
        if coords:
            assert len(coords) == 3, "the size of coords should be 3."
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
        else:
            self.x = domain * numpy.random.random()
            self.y = domain * numpy.random.random()
            self.z = domain * numpy.random.random()
            
    def distance(self, other):
        return numpy.sqrt((self.x-other.x)**2 + (self.y-other.y)**2
                                          + (self.z-other.z)**2)


class Particle(Point):
    """The derived class for a particle, inheriting the base class "Point".
    
    Attributes:
        m: mass of the particle
        phi: the gravitational potential of the particle
    """
    
    def __init__(self, coords=[], domain=1.0, m=1.0):
        Point.__init__(self, coords, domain)
        self.m = m
        self.phi = 0.

class Cell():
    """The class for a cell.
    
    Arguments:
      n_crit: maximum number of particles in a leaf cell
    
    Attributes:
      nleaf (int): number of leaves in the cell
      leaf (array of int): array of leaf index
      nchild (int):  an integer whose last 8 bits is used to keep track of the empty child cells
      child (array of int): array of child index
      parent (int): index of parent cell
      x, y, z (float): coordinates of the cell's center
      r (float): radius of the cell (half of the side length for cubic cell)
      multipole (array of float): multipole array of the cell
      
    """
    def __init__(self, n_crit):
        self.nleaf = 0        # number of leaves
        self.leaf = numpy.zeros(n_crit, dtype=numpy.int)     # array of leaf index
        self.nchild = 0       # binary counter to keep track of empty cells
        self.child = numpy.zeros(8, dtype=numpy.int)         # array of child index
        self.parent = 0       # index of parent cell
        self.x = self.y = self.z = 0.                    # center of the cell
        self.r = 0.           # radius of the cell
        self.multipole = numpy.zeros(10, dtype=numpy.float)  # multipole array


def add_child(octant, p, cells, n_crit):
    """Add a cell to the end of cells list as a child of p, initialize the center and radius of the child cell c, and establish mutual reference between child c and parent p.
    
    Arguments:
      octant: reference to one of the eight divisions in three dimensions
      p: parent cell index in cells list
      cells: the list of cells
      n_crit: maximum number of leaves in a single cell
 
    """
    # create a new cell instance
    cells.append(Cell(n_crit))
    # the last element of the cells list is the new child c
    c = len(cells) - 1
    # geometry relationship between parent and child
    cells[c].r = cells[p].r / 2
    cells[c].x = cells[p].x + cells[c].r * ((octant & 1) * 2 - 1)
    cells[c].y = cells[p].y + cells[c].r * ((octant & 2) - 1    )
    cells[c].z = cells[p].z + cells[c].r * ((octant & 4) / 2 - 1)
    # establish mutual reference in the cells list
    cells[c].parent = p
    cells[p].child[octant] = c
    cells[p].nchild = (cells[p].nchild | (1 << octant))


def split_cell(particles, p, cells, n_crit):
    """Loop in parent p's leafs and reallocate the particles to subcells. If a subcell has not been created in that octant, create one using add_child. If the subcell c's leaf number exceeds n_crit, split the subcell c recursively.
    
    Arguments: 
      particles: the list of particles
      p: parent cell index in cells list
      cells: the list of cells
      n_crit: maximum number of leaves in a single cell
    
    """
    # loop in the particles stored in the parent cell that you want to split
    for l in cells[p].leaf:
        octant = (particles[l].x > cells[p].x) + ((particles[l].y > cells[p].y) << 1) \
               + ((particles[l].z > cells[p].z) << 2)   # find the particle's octant
        # if there is not a child cell in the particles octant, then create one
        if not cells[p].nchild & (1 << octant):
            add_child(octant, p, cells, n_crit)
        # reallocate the particle in the child cell
        c = cells[p].child[octant]
        cells[c].leaf[cells[c].nleaf] = l
        cells[c].nleaf += 1
        # check if the child reach n_crit
        if cells[c].nleaf >= n_crit:
            split_cell(particles, c, cells, n_crit)


def build_tree(particles, root, n_crit):
    """Construct a hierarchical octree to store the particles and return the tree (list) of cells.
    
    Arguments:
      particles: the list of particles
      root: the root cell
      n_crit: maximum number of leaves in a single cell
    
    Returns:
      cells: the list of cells
    
    """
    # set root cell
    cells = [root]       # initialize the cells list
    # build tree
    n = len(particles)
    for i in range(n):
        # traverse from the root down to a leaf cell
        curr = 0
        while cells[curr].nleaf >= n_crit:
            cells[curr].nleaf += 1
            octant = (particles[i].x > cells[curr].x) + ((particles[i].y > cells[curr].y) << 1) \
                   + ((particles[i].z > cells[curr].z) << 2)
            # if there is no child cell in the particles octant, then create one
            if not cells[curr].nchild & (1 << octant):
                add_child(octant, curr, cells, n_crit)
            curr = cells[curr].child[octant]
        # allocate the particle in the leaf cell
        cells[curr].leaf[cells[curr].nleaf] = i
        cells[curr].nleaf += 1
        # check whether to split or not
        if cells[curr].nleaf >= n_crit:
            split_cell(particles, curr, cells, n_crit)
    
    return cells


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