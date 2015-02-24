import numpy

class Cell():
    """The class for a cell.
    
    Arguments:
      n_crit: maximum number of particles in a single cell
    
    Attributes:
      nleaf (int): number of leaves in the cell
      leaf (array of int): array of leaf index
      nchild (int):  an integer whose last 8 bits is used to keep track of the empty child cells
      child (array of int): array of child index
      parent (int): index of parent cell
      x_c, y_c, z_c (float): coordinates of the cell's center
      r (float): radius of the cell (half of the side length for cubic cell)
      multipole (array of float): multipole array of the cell
      
    """
    def __init__(self, n_crit):
        self.nleaf = 0        # number of leaves
        self.leaf = numpy.zeros(n_crit, dtype=numpy.int)     # array of leaf index
        self.nchild = 0       # binary counter to keep track of empty cells
        self.child = numpy.zeros(8, dtype=numpy.int)         # array of child index
        self.parent = 0       # index of parent cell
        self.x_c = self.y_c = self.z_c = 0.                    # center of the cell
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
    cells[c].x_c = cells[p].x_c + cells[c].r * ((octant & 1) * 2 - 1)
    cells[c].y_c = cells[p].y_c + cells[c].r * ((octant & 2) - 1    )
    cells[c].z_c = cells[p].z_c + cells[c].r * ((octant & 4) / 2 - 1)
    # establish mutual reference in the cells list
    cells[c].parent = p
    cells[p].child[octant] = c
    cells[p].nchild = (cells[p].nchild | (1 << octant))


def split_cell(x, y, z, p, cells, n_crit):
    """Loop in parent p's leafs and reallocate the particles to subcells. If a subcell has not been created in that octant, create one using add_child. If the subcell c's leaf number exceeds n_crit, split the subcell c recursively.
    
    Arguments: 
      x, y, z: coordinates array of the particles
      p: parent cell index in cells list
      cells: the list of cells
      n_crit: maximum number of leaves in a single cell
    
    """
    # loop in the particles stored in the parent cell that you want to split
    for i in range(n_crit):
        l = cells[p].leaf[i]   # find the particle index
        octant = (x[l] > cells[p].x_c) + ((y[l] > cells[p].y_c) << 1) \
               + ((z[l] > cells[p].z_c) << 2)   # find the particle's octant
        # if there is not a child cell in the particles octant, then create one
        if not cells[p].nchild & (1 << octant):
            add_child(octant, p, cells, n_crit)
        # reallocate the particle in the child cell
        c = cells[p].child[octant]
        cells[c].leaf[cells[c].nleaf] = l
        cells[c].nleaf += 1
        # check if the child reach n_crit
        if cells[c].nleaf >= n_crit:
            split_cell(x, y, z, c, cells, n_crit)


def build_tree(n, x, y, z, x_r, y_r, z_r, r_r, n_crit):
    """Construct a hierarchical octree to store the particles and return the tree (list) of cells.
    
    Arguments:
      n: number of particles
      x, y, z: coordinates array of the particles
      x_r, y_r, z_r: coordinates of the root cell
      r_r: radius of the root cell
      n_crit: maximum number of leaves in a single cell
    
    Returns:
      cells: the list of cells
    
    """
    # set root cell
    cells = []       # initialize the cells list
    cells.append(Cell(n_crit))
    cells[0].x_c, cells[0].y_c, cells[0].z_c = x_r, y_r, z_r
    cells[0].r = r_r

    # build tree
    for i in range(n):
        # traverse from the root down to a leaf cell
        curr = 0
        while cells[curr].nleaf >= n_crit:
            cells[curr].nleaf += 1
            octant = (x[i] > cells[curr].x_c) + ((y[i] > cells[curr].y_c) << 1) \
                   + ((z[i] > cells[curr].z_c) << 2)
            # if there is no child cell in the particles octant, then create one
            if not cells[curr].nchild & (1 << octant):
                add_child(octant, curr, cells, n_crit)
            curr = cells[curr].child[octant]
        # allocate the particle in the leaf cell
        cells[curr].leaf[cells[curr].nleaf] = i
        cells[curr].nleaf += 1
        # check whether to split or not
        if cells[curr].nleaf >= n_crit:
            split_cell(x, y, z, curr, cells, n_crit)
    
    return cells


def get_multipole(x, y, z, m, p, cells, twigs, n_crit):
    """Calculate multipole arrays for all twig cells under cell p. If leaf number of 
       cell p is bigger than n_crit (non-twig), traverse down recursively. Otherwise
       (twig), calculate the multipole arrays for twig cell p.
    
    Arguments:
    p:       cell index in cells list
    """
    if cells[p].nleaf >= n_crit:
        for c in range(8):
            if cells[p].nchild & (1 << c):
                get_multipole(x, y, z, m, cells[p].child[c], cells, twigs, n_crit)
    else:
        for l in range(cells[p].nleaf):
            j = cells[p].leaf[l]
            dx = cells[p].x_c - x[j]
            dy = cells[p].y_c - y[j]
            dz = cells[p].z_c - z[j]
            cells[p].multipole[0] += m[j]
            cells[p].multipole[1] += m[j] * dx
            cells[p].multipole[2] += m[j] * dy
            cells[p].multipole[3] += m[j] * dz
            cells[p].multipole[4] += m[j] * dx * dx / 2
            cells[p].multipole[5] += m[j] * dy * dy / 2
            cells[p].multipole[6] += m[j] * dz * dz / 2
            cells[p].multipole[7] += m[j] * dx * dy / 2
            cells[p].multipole[8] += m[j] * dy * dz / 2
            cells[p].multipole[9] += m[j] * dz * dx / 2   

        twigs.append(p)

def upward_sweep(p, c, cells):
    """Calculate parent cell p's multipole array based on child cell c's multipoles
    
    Arguments:
      cells: the list of cells
      p:       parent cell index in cells list
      c:       child cell index in cells list

    """
    dx = cells[p].x_c - cells[c].x_c
    dy = cells[p].y_c - cells[c].y_c
    dz = cells[p].z_c - cells[c].z_c
    cells[p].multipole[0] += cells[c].multipole[0]
    cells[p].multipole[1] += cells[c].multipole[1] + dx * cells[c].multipole[0]
    cells[p].multipole[2] += cells[c].multipole[2] + dy * cells[c].multipole[0]
    cells[p].multipole[3] += cells[c].multipole[3] + dz * cells[c].multipole[0]
    cells[p].multipole[4] += cells[c].multipole[4] + dx * cells[c].multipole[1] \
                                                   + dx * dx * cells[c].multipole[0] / 2
    cells[p].multipole[5] += cells[c].multipole[5] + dy * cells[c].multipole[2] \
                                                   + dy * dy * cells[c].multipole[0] / 2
    cells[p].multipole[6] += cells[c].multipole[6] + dz * cells[c].multipole[3] \
                                                   + dz * dz * cells[c].multipole[0] / 2
    cells[p].multipole[7] += cells[c].multipole[7] + (dx * cells[c].multipole[2] \
                                                   +  dy * cells[c].multipole[1] \
                                                   +  dx * dy * cells[c].multipole[0]) / 2
    cells[p].multipole[8] += cells[c].multipole[8] + (dy * cells[c].multipole[3] \
                                                   +  dz * cells[c].multipole[2] \
                                                   +  dy * dz * cells[c].multipole[0]) / 2
    cells[p].multipole[9] += cells[c].multipole[9] + (dz * cells[c].multipole[1] \
                                                   +  dx * cells[c].multipole[3] \
                                                   +  dz * dx * cells[c].multipole[0]) / 2


def direct_sum(n, x, y, z, m, epsilon):
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
    phi = -m/epsilon
    eps2 = epsilon**2
    for i in range(n):
        for j in range(n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dz = z[i] - z[j]
            r = numpy.sqrt(dx**2 + dy**2 + dz**2 + eps2)
            phi[i] += m[j]/r
    return phi