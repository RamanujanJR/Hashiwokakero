from itertools import product
from model import HashiPuzzle, BridgeVariable
from utils import BridgeUtils 

class ConstraintPropagation:

    def __init__(self, puzzle: HashiPuzzle):
        self.puzzle = puzzle
        self.forced_bridges = {}
        self.forbidden_bridges = set()
        self.min_bridges = {}

    def apply_all_rules(self):
        self.forced_bridges = {}
        self.forbidden_bridges = set()
        self.min_bridges = {}

        for island in self.puzzle.islands:
            r, c, value = island
            neighbors = self.puzzle.get_neighbors(island)

            # Rule 1: Island số 8 (ở giữa)
            if value == 8 and len(neighbors) == 4:
                self._force_all_double(island, neighbors)

            # Rule 2: Corner rules
            if self._is_corner(r, c):
                self._apply_corner_rules(island, neighbors)

            # Rule 3: Edge rules
            if self._is_edge(r, c) and not self._is_corner(r, c):
                self._apply_edge_rules(island, neighbors)

            # Rule 4: Single neighbor rule
            if len(neighbors) == 1:
                self._apply_single_neighbor_rule(island, neighbors)

            # Rule 5: Forced connections
            self._check_forced_connections(island, neighbors)

            # Rule 6: Sub-saturation rule
            self._apply_subsaturation_rule(island, neighbors)

        # Rule 7: Isolation Prevention
        self._apply_isolation_prevention()

        return self.forced_bridges, self.forbidden_bridges, self.min_bridges

    def _is_corner(self, r, c):
        rows, cols = self.puzzle.rows, self.puzzle.cols
        return (r == 0 or r == rows - 1) and (c == 0 or c == cols - 1)

    def _is_edge(self, r, c):
        rows, cols = self.puzzle.rows, self.puzzle.cols
        return (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)

    def _force_all_double(self, island, neighbors):
        r, c, _ = island
        for (nr, nc), _ in neighbors:
            bridge_key = tuple(sorted([(r, c), (nr, nc)]))
            self.forced_bridges[bridge_key] = 2

    def _apply_corner_rules(self, island, neighbors):
        r, c, value = island

        if value == 4 and len(neighbors) == 2:
            for (nr, nc), _ in neighbors:
                bridge_key = tuple(sorted([(r, c), (nr, nc)]))
                self.forced_bridges[bridge_key] = 2

        elif value == 3 and len(neighbors) == 2:
            for (nr, nc), _ in neighbors:
                bridge_key = tuple(sorted([(r, c), (nr, nc)]))
                self.min_bridges[bridge_key] = 1

    def _apply_edge_rules(self, island, neighbors):
        r, c, value = island

        if value == 6 and len(neighbors) == 3:
            for (nr, nc), _ in neighbors:
                bridge_key = tuple(sorted([(r, c), (nr, nc)]))
                self.forced_bridges[bridge_key] = 2

    def _apply_single_neighbor_rule(self, island, neighbors):
        r, c, value = island

        if value <= 2 and len(neighbors) == 1:
            (nr, nc), _ = neighbors[0]
            bridge_key = tuple(sorted([(r, c), (nr, nc)]))
            self.forced_bridges[bridge_key] = value

    def _check_forced_connections(self, island, neighbors):
        r, c, value = island
        n = len(neighbors)

        if n == 0:
            return

        max_possible = n * 2

        if value == max_possible:
            for (nr, nc), _ in neighbors:
                bridge_key = tuple(sorted([(r, c), (nr, nc)]))
                self.forced_bridges[bridge_key] = 2

    def _apply_subsaturation_rule(self, island, neighbors):
        r, c, value = island
        n = len(neighbors)

        if n == 0:
            return

        max_possible = 2 * n

        if value == max_possible - 1:
            for (nr, nc), _ in neighbors:
                bridge_key = tuple(sorted([(r, c), (nr, nc)]))
                if bridge_key not in self.forced_bridges:
                    self.min_bridges[bridge_key] = 1

    def _apply_isolation_prevention(self):
        for i, island1 in enumerate(self.puzzle.islands):
            r1, c1, val1 = island1
            neighbors = self.puzzle.get_neighbors(island1)

            for (r2, c2), _ in neighbors:
                val2 = None
                for r, c, v in self.puzzle.islands:
                    if r == r2 and c == c2:
                        val2 = v
                        break

                if val2 is None:
                    continue

                bridge_key = tuple(sorted([(r1, c1), (r2, c2)]))

                if val1 == 1 and val2 == 1:
                    self.forbidden_bridges.add(bridge_key)

    def is_bridge_forbidden(self, bridge_key, num_bridges):
        if bridge_key in self.forbidden_bridges:
            return True

        island1, island2 = bridge_key
        val1 = val2 = None

        for r, c, v in self.puzzle.islands:
            if (r, c) == island1:
                val1 = v
            if (r, c) == island2:
                val2 = v

        if val1 == 2 and val2 == 2 and num_bridges == 2:
            return True

        return False

    def get_min_bridges(self, bridge_key):
        return self.min_bridges.get(bridge_key, 0)
    
class CNFGenerator:

    def __init__(self, puzzle: HashiPuzzle):
        self.puzzle = puzzle
        self.bridge_vars = BridgeVariable(puzzle)
        self.clauses = []

    def generate_cnf(self):
        """Generate all CNF constraints"""
        self.clauses = []

        self._add_island_count_constraints()
        self._add_bridge_mutex_constraints()
        self._add_no_crossing_constraints()

        # Remove duplicates
        unique_clauses = set()
        for clause in self.clauses:
            unique_clauses.add(tuple(sorted(clause)))
        self.clauses = [list(clause) for clause in unique_clauses]

        return self.clauses

    def _add_island_count_constraints(self):
        for island in self.puzzle.islands:
            r, c, required_count = island
            island_pos = (r, c)

            neighbors = self.puzzle.get_neighbors(island)
            bridge_vars = []

            for (r2, c2), _ in neighbors:
                neighbor_pos = (r2, c2)
                var1 = self.bridge_vars.get_variable(island_pos, neighbor_pos, 1)
                var2 = self.bridge_vars.get_variable(island_pos, neighbor_pos, 2)
                if var1 and var2:
                    bridge_vars.append((var1, var2))

            if len(bridge_vars) == 0:
                continue

            self._add_exactly_k_constraint(bridge_vars, required_count)

    def _add_exactly_k_constraint(self, bridge_vars, k):
        n = len(bridge_vars)
        self._add_at_least_k(bridge_vars, k)
        self._add_at_most_k(bridge_vars, k)

    def _add_at_least_k(self, bridge_vars, k):
        n = len(bridge_vars)
        max_possible = 2 * n

        if k == 0:
            return

        if k > max_possible:
            self.clauses.append([])
            return

        if n <= 4:
            for assignment in product([0, 1, 2], repeat=n):
                total = sum(assignment)
                if total < k:
                    clause = []
                    for i, bridges in enumerate(assignment):
                        v1, v2 = bridge_vars[i]
                        if bridges == 0:
                            clause.extend([v1, v2])
                        elif bridges == 1:
                            clause.extend([-v1, v2])
                        elif bridges == 2:
                            clause.extend([v1, -v2])

                    if clause:
                        self.clauses.append(clause)

    def _add_at_most_k(self, bridge_vars, k):
        n = len(bridge_vars)

        if k >= 2 * n:
            return

        if n <= 4:
            for assignment in product([0, 1, 2], repeat=n):
                total = sum(assignment)
                if total > k:
                    clause = []
                    for i, bridges in enumerate(assignment):
                        v1, v2 = bridge_vars[i]
                        if bridges == 0:
                            clause.extend([v1, v2])
                        elif bridges == 1:
                            clause.extend([-v1, v2])
                        elif bridges == 2:
                            clause.extend([v1, -v2])

                    if clause:
                        self.clauses.append(clause)

    def _add_bridge_mutex_constraints(self):
        for (key, num_bridges), var in self.bridge_vars.bridge_to_var.items():
            if num_bridges == 1:
                var1 = var
                var2 = self.bridge_vars.bridge_to_var.get((key, 2))
                if var2:
                    self.clauses.append([-var1, -var2])

    def _add_no_crossing_constraints(self):
        all_bridge_keys = set()
        for (key, num_bridges) in self.bridge_vars.bridge_to_var.keys():
            all_bridge_keys.add(key)

        all_bridge_keys = list(all_bridge_keys)

        for i in range(len(all_bridge_keys)):
            for j in range(i + 1, len(all_bridge_keys)):
                key1 = all_bridge_keys[i]
                key2 = all_bridge_keys[j]

                if BridgeUtils.bridges_cross(key1, key2):
                    for n1 in [1, 2]:
                        for n2 in [1, 2]:
                            var1 = self.bridge_vars.bridge_to_var.get((key1, n1))
                            var2 = self.bridge_vars.bridge_to_var.get((key2, n2))
                            if var1 and var2:
                                self.clauses.append([-var1, -var2])