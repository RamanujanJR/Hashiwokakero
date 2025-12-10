import numpy as np 

class HashiPuzzle:
    """Class to represent Hashiwokakero puzzle"""

    def __init__(self, grid):
        self.grid = np.array(grid)
        self.rows, self.cols = self.grid.shape
        self.islands = self._find_islands()
        self.bridges = {}

    def _find_islands(self):
        """Find all islands in grid"""
        islands = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i][j] > 0:
                    islands.append((i, j, self.grid[i][j]))
        return islands

    def get_neighbors(self, island):
        """Find all islands that can connect to this island"""
        row, col, _ = island
        neighbors = []

        directions = [(-1, 0, 'vertical'), (1, 0, 'vertical'),
                     (0, -1, 'horizontal'), (0, 1, 'horizontal')]

        for dr, dc, dir_type in directions:
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols:
                if self.grid[r][c] > 0:
                    neighbors.append(((r, c), dir_type))
                    break
                r += dr
                c += dc

        return neighbors

    @staticmethod
    def read_from_file(filename):
        """Read puzzle from file"""
        with open(filename, 'r') as f:
            lines = f.readlines()

        grid = []
        for line in lines:
            row = [int(x.strip()) for x in line.split(',')]
            grid.append(row)

        return HashiPuzzle(grid)


class BridgeVariable:
    """Class to manage logic variables for bridges"""

    def __init__(self, puzzle: HashiPuzzle):
        self.puzzle = puzzle
        self.var_counter = 1
        self.bridge_to_var = {}
        self.var_to_bridge = {}
        self._initialize_variables()

    def _initialize_variables(self):
        """Create logic variables for all possible bridges"""
        for island1 in self.puzzle.islands:
            neighbors = self.puzzle.get_neighbors(island1)
            r1, c1, _ = island1

            for (r2, c2), _ in neighbors:
                island2 = (r2, c2)
                key = tuple(sorted([(r1, c1), island2]))

                if key not in self.bridge_to_var:
                    for num_bridges in [1, 2]:
                        var = self.var_counter
                        self.bridge_to_var[(key, num_bridges)] = var
                        self.var_to_bridge[var] = (key, num_bridges)
                        self.var_counter += 1

    def get_variable(self, island1_pos, island2_pos, num_bridges):
        """Get logic variable for a bridge"""
        key = tuple(sorted([island1_pos, island2_pos]))
        return self.bridge_to_var.get((key, num_bridges))