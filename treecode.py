import numpy

class Cell():
    """The class for a cell."""
    def __init__(self, n_crit):
        self.nleaf = 0        # number of leaves
        self.nchild = 0       # binary counter to keep track of empty cells
        self.leaf = numpy.zeros(n_crit, dtype=numpy.int)     # array of leaf index
        self.x_c = self.y_c = self.z_c = 0.                    # center of the cell
        self.r = 0.           # radius of the cell
        self.parent = 0       # index of parent cell
        self.child = numpy.zeros(8, dtype=numpy.int)         # array of child index
        self.multipole = numpy.zeros(10, dtype=numpy.float64)  # multipole array

def direct_sum(n, x, y, z, m):
    """Calculate the gravitational potential at each target particle i using direct
       summation method.

    Arguments:
        n: total number of the particles
        x, y, z: x, y, z coordinate array of the particles
        m: mass array of the particles

    Returns:
        phi: array of the gravitational potential at target points 
    """
    phi = numpy.zeros(n)
    epsilon = 0.01
    for i in range(n):
        phi[i] = -m[i]/epsilon
        for j in range(n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            r = numpy.sqrt(dx**2 + dy**2 + dz**2 + epsilon**2)
            phi[i] += m[j]/r
    return phi


def add_child(octant, p, cells, n_crit):
    """Add a cell to the end of cells list as a child of p, initialize the center and
       radius of the child cell c, and establish mutual reference between child c and
       parent p.
    
    Arguments:
        octant: reference to one of the eight divisions in three dimensions
        p: parent cell index in cells list
        cells: the list of cells
        n_crit: maximum number of leaves in a single cell
    """
    cells.append(Cell(n_crit))
    c = len(cells) - 1
    cells[c].r = cells[p].r / 2
    cells[c].x_c = cells[p].x_c + cells[c].r * ((octant & 1) * 2 - 1)
    cells[c].y_c = cells[p].y_c + cells[c].r * ((octant & 2) - 1    )
    cells[c].z_c = cells[p].z_c + cells[c].r * ((octant & 4) / 2 - 1)
    cells[c].parent = p
    cells[p].child[octant] = c
    cells[p].nchild = (cells[p].nchild | (1 << octant))

def split_cell(x, y, z, p, cells, n_crit):
    """Loop in parent p's leafs and reallocate the particles to subcells. If a subcell
       has not been created in that octant, create one using add_child. If the subcell
       c's leaf number exceeds n_crit, split the subcell c recursively.
    
    Arguments: 
    x, y, z: x, y, z coordinate array of the particles
    p: parent cell index in cells list
    cells: the list of cells
    n_crit: maximum number of leaves in a single cell
    """
    print '==start split cell {}=='.format(p)
    for i in range(n_crit):
        l = cells[p].leaf[i]
        octant = (x[l] > cells[p].x_c) + ((y[l] > cells[p].y_c) << 1) \
               + ((z[l] > cells[p].z_c) << 2)
        
        if not cells[p].nchild & (1 << octant):
            add_child(octant, p, cells, n_crit)
        
        c = cells[p].child[octant]
        cells[c].leaf[cells[c].nleaf] = l
        cells[c].nleaf += 1
        print '>>>particle {} is reallocated in cell {}'.format(i, c)
        if cells[c].nleaf >= n_crit:
            split_cell(x, y, z, c, cells, n_crit)
    print '==end split cell {}=='.format(p)