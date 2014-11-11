import numpy

class Cell():
    """The class for a cell"""
    def __init__(self, n_crit):
        self.nleaf = 0        # number of leaves
        self.nchild = 0       # binary counter to keep track of empty cells
        self.leaf = numpy.zeros(n_crit, dtype=int32)     # array of leaf index
        self.x_c = self.y_c = self.z_c = 0.                    # center of the cell
        self.r = 0.           # radius of the cell
        self.parent = 0       # index of parent cell
        self.child = numpy.zeros(8, dtype=int32)         # array of child index
        self.multipole = numpy.zeros(10, dtype=float64)  # multipole array

def add_child(octant, p, cells, n_crit):
    """Add a cell to the end of cells list as a child of p, initialize the center and
       radius of the child cell c, and establish mutual reference between child c and
       parent p
    
    Arguments:
      octant:  reference to one of the eight divisions in three dimensions
      p:       parent cell index in cells list
    """
    cells.append(Cell())
    c = len(cells)
    cells[c].r = cells[p].r / 2
    cells[c].x_c = cells[p].x_c + cells[c].r * ((octant & 1) * 2 - 1)
    cells[c].y_c = cells[p].y_c + cells[c].r * ((octant & 2) - 1    )
    cells[c].z_c = cells[p].z_c + cells[c].r * ((octant & 4) / 2 - 1)
    cells[c].parent = p
    cells[p].child[octant] = c
    cells[p].nchild = (cells[p].nchild | (1 << octant))