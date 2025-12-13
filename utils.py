import numpy as np

class BridgeUtils:
    """Utility functions shared by bridges"""

    @staticmethod
    def bridges_cross(key1, key2):
        """Check if two bridges cross each other"""
        (r1, c1), (r2, c2) = key1
        (r3, c3), (r4, c4) = key2

        # Bridge 1 horizontal, Bridge 2 vertical
        if r1 == r2 and c3 == c4:
            min_c, max_c = min(c1, c2), max(c1, c2)
            min_r, max_r = min(r3, r4), max(r3, r4)
            if min_c < c3 < max_c and min_r < r1 < max_r:
                return True

        # Bridge 1 vertical, Bridge 2 horizontal
        elif c1 == c2 and r3 == r4:
            min_r, max_r = min(r1, r2), max(r1, r2)
            min_c, max_c = min(c3, c4), max(c3, c4)
            if min_r < r3 < max_r and min_c < c1 < max_c:
                return True

        return False

    @staticmethod
    def is_connected(bridges, all_islands):
        if len(all_islands) == 0:
            return True

        # Build adjacency list
        adj = {(r, c): [] for r, c, _ in all_islands}

        for (pos1, pos2), num_bridges in bridges.items():
            if num_bridges > 0:
                adj[pos1].append(pos2)
                adj[pos2].append(pos1)

        # BFS from first island
        start = (all_islands[0][0], all_islands[0][1])
        visited = {start}
        queue = [start]

        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(all_islands)

    @staticmethod
    def get_all_bridges_sorted(puzzle, island_requirements=None):
        """Get all possible bridges, sorted by priority"""
        bridges = []
        seen = set()

        for island in puzzle.islands:
            r, c, _ = island
            neighbors = puzzle.get_neighbors(island)

            for (r2, c2), _ in neighbors:
                key = tuple(sorted([(r, c), (r2, c2)]))
                if key not in seen:
                    seen.add(key)
                    bridges.append(key)

        # Sort by priority if requirements provided
        if island_requirements:
            def priority(bridge_key):
                i1, i2 = bridge_key
                req1 = island_requirements.get(i1, 0)
                req2 = island_requirements.get(i2, 0)
                return -(req1 + req2)

            bridges.sort(key=priority)

        return bridges


class VisualizationMixin:
    """Mixin class for solution visualization"""

    def visualize_solution(self, solution=None, title="Solution"):
        """Visualize solution as grid"""
        if solution is None:
            solution = getattr(self, 'solution', None)

        if solution is None:
            print("No solution to visualize!")
            return None

        rows, cols = self.puzzle.rows, self.puzzle.cols
        output = [[' 0 ' for _ in range(cols)] for _ in range(rows)]

        # Place islands
        for r, c, val in self.puzzle.islands:
            output[r][c] = f' {val} '

        # Place bridges
        for (island1, island2), num_bridges in solution.items():
            r1, c1 = island1
            r2, c2 = island2

            if r1 == r2:  # Horizontal
                symbol = ' = ' if num_bridges == 2 else ' - '
                min_c, max_c = min(c1, c2), max(c1, c2)
                for c in range(min_c + 1, max_c):
                    output[r1][c] = symbol
            else:  # Vertical
                symbol = ' $ ' if num_bridges == 2 else ' | '
                min_r, max_r = min(r1, r2), max(r1, r2)
                for r in range(min_r + 1, max_r):
                    output[r][c1] = symbol

        print(f"\n{title}:")
        for row in output:
            print('[' + ','.join([f'"{cell.strip()}"' for cell in row]) + ']')

        return output