import time
import heapq
from itertools import product
from collections import defaultdict
from pysat.solvers import Glucose3
from utils import BridgeUtils, VisualizationMixin
from logic import CNFGenerator, ConstraintPropagation
from model import HashiPuzzle 

class PySATSolver(VisualizationMixin):

    def __init__(self, puzzle: HashiPuzzle):
        self.puzzle = puzzle
        self.cnf_gen = CNFGenerator(puzzle)
        self.solution = None
        self.solve_time = 0

    def solve(self):
        start_time = time.time()

        # Generate CNF
        clauses = self.cnf_gen.generate_cnf()

        # Create SAT solver
        solver = Glucose3()

        # Add clauses
        for clause in clauses:
            solver.add_clause(clause)

        # Solve with connectivity check
        while solver.solve():
            model = solver.get_model()
            solution = self._decode_solution(model)

            # Check if solution is connected
            if BridgeUtils.is_connected(solution, self.puzzle.islands):
                self.solution = solution
                self.solve_time = time.time() - start_time
                return True, self.solution

            # Block this solution and try again
            blocking_clause = [-lit for lit in model if lit in self.cnf_gen.bridge_vars.var_to_bridge]
            solver.add_clause(blocking_clause)

        # No valid solution found
        self.solve_time = time.time() - start_time
        return False, None

    def _decode_solution(self, model):
        """Decode SAT model to solution"""
        bridges = {}
        for var in model:
            if var > 0 and var in self.cnf_gen.bridge_vars.var_to_bridge:
                bridge_key, num_bridges = self.cnf_gen.bridge_vars.var_to_bridge[var]
                bridges[bridge_key] = num_bridges
        return bridges

    def get_stats(self):
        return {
            'solve_time': self.solve_time,
            'num_variables': self.cnf_gen.bridge_vars.var_counter - 1,
            'num_clauses': len(self.cnf_gen.clauses),
            'num_islands': len(self.puzzle.islands)
        }
        
class AStarState:
    """Optimized state for A* search"""

    def __init__(self, bridges, island_counts, g_cost):
        self.bridges = bridges
        self.island_counts = island_counts
        self.g_cost = g_cost
        self._hash = None

    def get_hash(self):
        """Cache hash to avoid recomputation"""
        if self._hash is None:
            self._hash = hash(tuple(sorted(self.bridges.items())))
        return self._hash

    def __eq__(self, other):
        return self.get_hash() == other.get_hash()

    def __hash__(self):
        return self.get_hash()


class AStarSolver(VisualizationMixin):
    """A* Solver with complete rule-based heuristic and optimizations"""

    def __init__(self, puzzle, use_advanced_heuristic=True):
        self.puzzle = puzzle
        self.solution = None
        self.solve_time = 0
        self.nodes_expanded = 0
        self.use_advanced_heuristic = use_advanced_heuristic

        # Precompute island requirements
        self.island_requirements = {
            (r, c): req for r, c, req in puzzle.islands
        }

        # Get all bridges sorted by priority
        self.all_bridges = BridgeUtils.get_all_bridges_sorted(
            puzzle, self.island_requirements
        )

        # Build bridge graph for O(1) neighbor lookup
        self.bridge_graph = self._build_bridge_graph()

        # Precompute crossing map for O(1) crossing check
        self.crossing_map = self._build_crossing_map()

    def _build_bridge_graph(self):
        """Build adjacency list for bridges: O(n) construction, O(1) lookup"""
        graph = defaultdict(list)
        for bridge_key in self.all_bridges:
            pos1, pos2 = bridge_key
            graph[pos1].append(pos2)
            graph[pos2].append(pos1)
        return graph

    def _build_crossing_map(self):
        """Precompute which bridges cross each other: O(n²) construction, O(1) lookup"""
        crossing = defaultdict(set)

        for i, b1 in enumerate(self.all_bridges):
            for b2 in self.all_bridges[i+1:]:
                if BridgeUtils.bridges_cross(b1, b2):
                    crossing[b1].add(b2)
                    crossing[b2].add(b1)

        return crossing

    def solve(self):
        """A* search algorithm"""
        start_time = time.time()

        # Initialize
        initial_counts = {pos: 0 for pos in self.island_requirements}
        initial_state = AStarState({}, initial_counts, 0)

        # Priority queue: (f_score, counter, state)
        open_set = []
        counter = 0
        h_initial = self._heuristic(initial_state)
        heapq.heappush(open_set, (h_initial, counter, initial_state))

        closed_set = set()
        best_f_score = {initial_state.get_hash(): h_initial}

        # A* main loop
        while open_set:
            # Timeout check
            if self.nodes_expanded > 100000:
                break
            if time.time() - start_time > 60:
                break

            f_score, _, current = heapq.heappop(open_set)
            self.nodes_expanded += 1

            # Goal check
            if self._is_goal(current):
                self.solution = current.bridges
                self.solve_time = time.time() - start_time
                return True, self.solution

            # Skip if already visited
            current_hash = current.get_hash()
            if current_hash in closed_set:
                continue
            closed_set.add(current_hash)

            # Generate and process successors
            successors = self._get_successors(current)

            for successor in successors:
                succ_hash = successor.get_hash()

                if succ_hash in closed_set:
                    continue

                # Calculate f-score
                h = self._heuristic(successor)
                f = successor.g_cost + h

                # Add to open set if better path found
                if succ_hash not in best_f_score or f < best_f_score[succ_hash]:
                    best_f_score[succ_hash] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, successor))

        self.solve_time = time.time() - start_time
        return False, None

    def _is_goal(self, state):
        """Check if state is a goal state"""
        # Check all islands satisfied
        for pos, req in self.island_requirements.items():
            if state.island_counts[pos] != req:
                return False

        # Check connectivity using BridgeUtils
        return BridgeUtils.is_connected(state.bridges, self.puzzle.islands)

    def _heuristic(self, state):
        """
        Complete 5-component heuristic:
        1. Basic deficit (admissible baseline)
        2. Forced bridges estimation
        3. Isolation detection
        4. Sub-saturation check
        5. Dead-end detection
        """
        if not self.use_advanced_heuristic:
            return self._simple_heuristic(state)

        h = 0.0

        # Component 1: Basic deficit (admissible)
        total_deficit = 0
        island_deficits = {}

        for pos, req in self.island_requirements.items():
            current = state.island_counts[pos]
            deficit = req - current

            if deficit > 0:
                total_deficit += deficit
                island_deficits[pos] = deficit

        h = total_deficit / 2.0  # Each bridge satisfies 2 islands

        # Component 2: Forced bridges estimation
        forced_bonus = self._estimate_forced_bridges(state, island_deficits)
        h += forced_bonus * 0.5  # Scaled to maintain admissibility

        # Component 3: Isolation detection
        isolation_penalty = self._detect_potential_isolation(state)
        h += isolation_penalty

        # Component 4: Sub-saturation check
        subsaturation_penalty = self._check_subsaturation(state, island_deficits)
        h += subsaturation_penalty

        # Component 5: Dead-end detection
        if self._has_dead_end(state, island_deficits):
            h += 100  # Large penalty for impossible states

        return h

    def _simple_heuristic(self, state):
        """Simple admissible heuristic: deficit / 2"""
        total_deficit = 0
        for pos, req in self.island_requirements.items():
            deficit = req - state.island_counts[pos]
            if deficit > 0:
                total_deficit += deficit
        return total_deficit / 2.0

    def _estimate_forced_bridges(self, state, island_deficits):
        """Estimate penalty for forced bridges not yet added"""
        penalty = 0.0

        for pos, deficit in island_deficits.items():
            if deficit <= 0:
                continue

            # Use precomputed bridge_graph for O(1) neighbor lookup
            available_neighbors = []
            for neighbor in self.bridge_graph[pos]:
                bridge_key = tuple(sorted([pos, neighbor]))

                # Skip if bridge already used
                if bridge_key in state.bridges:
                    continue

                # Check if neighbor has capacity
                other_capacity = (self.island_requirements[neighbor] -
                                state.island_counts[neighbor])

                if other_capacity > 0:
                    available_neighbors.append((bridge_key, neighbor, other_capacity))

            n_available = len(available_neighbors)

            # Dead end - no available neighbors
            if n_available == 0:
                penalty += 1000
                continue

            max_possible = n_available * 2

            # All neighbors must have 2 bridges
            if deficit == max_possible:
                penalty += n_available * 0.5

            # Single neighbor with small deficit
            elif n_available == 1 and deficit <= 2:
                penalty += 0.5

            # Sub-saturation: V = 2N-1
            elif deficit == max_possible - 1:
                penalty += n_available * 0.3

        return penalty

    def _detect_potential_isolation(self, state):
        """Detect isolated pairs (1-1 or 2-2) that cannot connect to rest"""
        penalty = 0.0

        for bridge_key, num_bridges in state.bridges.items():
            island1, island2 = bridge_key
            val1 = self.island_requirements[island1]
            val2 = self.island_requirements[island2]
            count1 = state.island_counts[island1]
            count2 = state.island_counts[island2]

            # 1-1 fully connected = isolated
            if val1 == 1 and val2 == 1:
                if count1 == 1 and count2 == 1:
                    penalty += 10.0

            # 2=2 fully connected = isolated
            if val1 == 2 and val2 == 2 and num_bridges == 2:
                if count1 == 2 and count2 == 2:
                    penalty += 10.0

        return penalty

    def _check_subsaturation(self, state, island_deficits):
        """Check sub-saturation rule: V = 2N-1 requires all N directions"""
        penalty = 0.0

        for pos, deficit in island_deficits.items():
            if deficit <= 0:
                continue

            # Count connected and available directions using bridge_graph
            connected = 0
            available = 0

            for neighbor in self.bridge_graph[pos]:
                bridge_key = tuple(sorted([pos, neighbor]))

                if bridge_key in state.bridges:
                    connected += 1
                else:
                    # Check if direction is available
                    other_capacity = (self.island_requirements[neighbor] -
                                    state.island_counts[neighbor])
                    if other_capacity > 0:
                        available += 1

            total_directions = connected + available

            # Sub-saturation: if V = 2N-1, all N directions need at least 1 bridge
            if deficit == 2 * total_directions - 1:
                penalty += available * 0.4

        return penalty

    def _has_dead_end(self, state, island_deficits):
        """Check if any island cannot satisfy its requirement"""
        for pos, deficit in island_deficits.items():
            if deficit <= 0:
                continue

            available_capacity = 0

            # Check all neighbors using bridge_graph
            for neighbor in self.bridge_graph[pos]:
                bridge_key = tuple(sorted([pos, neighbor]))

                # Calculate remaining slots
                current_bridges = state.bridges.get(bridge_key, 0)
                remaining_slots = 2 - current_bridges

                if remaining_slots > 0:
                    # Check crossing using precomputed map (O(1))
                    crosses = any(bridge_key in self.crossing_map[existing]
                                for existing in state.bridges)

                    if not crosses:
                        # Calculate available capacity
                        other_deficit = (self.island_requirements[neighbor] -
                                       state.island_counts[neighbor])
                        available = min(remaining_slots, other_deficit)
                        available_capacity += available

            # If available capacity < deficit, impossible to satisfy
            if available_capacity < deficit:
                return True

        return False

    def _get_successors(self, state):
        """Generate successor states using MRV (Most Constrained Variable) heuristic"""
        successors = []

        # Find most constrained island (smallest remaining requirement)
        best_island = None
        min_remaining = float('inf')

        for pos, req in self.island_requirements.items():
            current = state.island_counts[pos]
            remaining = req - current

            if remaining > 0 and remaining < min_remaining:
                min_remaining = remaining
                best_island = pos

        if best_island is None:
            return successors

        # Try adding bridges from this island using bridge_graph
        for neighbor in self.bridge_graph[best_island]:
            bridge_key = tuple(sorted([best_island, neighbor]))

            # Skip if bridge already used
            if bridge_key in state.bridges:
                continue

            # Check capacities
            cap1 = self.island_requirements[best_island] - state.island_counts[best_island]
            cap2 = self.island_requirements[neighbor] - state.island_counts[neighbor]

            if cap1 <= 0 or cap2 <= 0:
                continue

            max_bridges = min(cap1, cap2, 2)

            # Try 1 or 2 bridges
            for num_bridges in range(1, max_bridges + 1):
                # Check crossing using precomputed map
                if self._can_add_bridge(state, bridge_key):
                    # Create successor state
                    new_bridges = state.bridges.copy()
                    new_bridges[bridge_key] = num_bridges

                    new_counts = state.island_counts.copy()
                    new_counts[best_island] += num_bridges
                    new_counts[neighbor] += num_bridges

                    new_g_cost = state.g_cost + num_bridges

                    successor = AStarState(new_bridges, new_counts, new_g_cost)
                    successors.append(successor)

        return successors

    def _can_add_bridge(self, state, new_bridge_key):
        """Check if bridge can be added without crossing (O(1) using crossing_map)"""
        for existing_key in state.bridges:
            if new_bridge_key in self.crossing_map[existing_key]:
                return False
        return True

    def get_stats(self):
        """Get solving statistics"""
        return {
            'solve_time': self.solve_time,
            'nodes_expanded': self.nodes_expanded,
            'num_islands': len(self.puzzle.islands),
            'heuristic_type': 'Advanced (5-component)' if self.use_advanced_heuristic else 'Simple (deficit/2)'
        }

class NaiveBacktrackingSolver(VisualizationMixin):
    """
    Naive Backtracking Solver - Pure backtracking without CSP optimizations

    Characteristics:
    - Simple depth-first search with backtracking
    - Fixed variable ordering (by bridge list order)
    - Fixed value ordering (0, 1, 2)
    - No forward checking
    - No constraint propagation
    - Check constraints only after full assignment

    Purpose: Baseline for comparison
    """

    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.solution = None
        self.solve_time = 0
        self.nodes_expanded = 0

        # Basic structures
        self.island_requirements = {
            (r, c): req for r, c, req in puzzle.islands
        }
        self.all_bridges = BridgeUtils.get_all_bridges_sorted(puzzle)

    def solve(self):
        """Naive backtracking solve"""
        start_time = time.time()
        self.start_time = start_time
        assignment = {}
        success = self._backtrack_naive(assignment, 0)

        self.solve_time = time.time() - start_time

        if success:
            self.solution = {k: v for k, v in assignment.items() if v > 0}
            return True, self.solution
        else:
            return False, None

    def _backtrack_naive(self, assignment, bridge_index):
        """Pure backtracking - no optimizations"""
        self.nodes_expanded += 1

        # Timeout
        if self.nodes_expanded > 200000:
            return False
        if time.time() - self.start_time > 60:
            return False

        # Base case: all bridges assigned
        if bridge_index >= len(self.all_bridges):
            return self._is_complete_valid_solution(assignment)

        bridge = self.all_bridges[bridge_index]

        # Try all values: 0, 1, 2 (no intelligent ordering)
        for num_bridges in [0, 1, 2]:
            assignment[bridge] = num_bridges

            # Basic check: not exceeding island capacity
            if self._is_consistent(assignment):
                if self._backtrack_naive(assignment, bridge_index + 1):
                    return True

            # Backtrack
            del assignment[bridge]

        return False

    def _is_consistent(self, assignment):
        """Basic consistency check: no island exceeds requirement"""
        island_counts = defaultdict(int)

        for bridge_key, num_bridges in assignment.items():
            island1, island2 = bridge_key
            island_counts[island1] += num_bridges
            island_counts[island2] += num_bridges

        # Check no island exceeds requirement
        for island_pos, count in island_counts.items():
            if count > self.island_requirements.get(island_pos, 0):
                return False

        # Check no crossing
        bridges_used = [(k, v) for k, v in assignment.items() if v > 0]
        for i, (key1, val1) in enumerate(bridges_used):
            for key2, val2 in bridges_used[i+1:]:
                if BridgeUtils.bridges_cross(key1, key2):
                    return False

        return True

    def _is_complete_valid_solution(self, assignment):
        """Check if assignment is complete and valid"""
        island_counts = defaultdict(int)

        for bridge_key, num_bridges in assignment.items():
            island1, island2 = bridge_key
            island_counts[island1] += num_bridges
            island_counts[island2] += num_bridges

        # Check all islands satisfied
        for island_pos, required in self.island_requirements.items():
            if island_counts[island_pos] != required:
                return False

        # Check connectivity using BridgeUtils
        return BridgeUtils.is_connected(assignment, self.puzzle.islands)

    def get_stats(self):
        return {
            'solve_time': self.solve_time,
            'nodes_expanded': self.nodes_expanded,
            'algorithm': 'Naive Backtracking'
        }
        
class OptimizedBacktrackingSolver(VisualizationMixin):
    """
    Optimized Backtracking with CSP Techniques + Constraint Propagation

    Enhancements over Naive:
    1. Constraint Propagation preprocessing (7 rules)
    2. MRV (Minimum Remaining Values) heuristic
    3. Forward Checking with propagation
    4. Arc Consistency (AC-3)
    5. Degree Heuristic for tie-breaking
    6. LCV (Least Constraining Value) ordering
    7. Precomputed structures (crossing cache, neighbor map)
    """

    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.solution = None
        self.solve_time = 0
        self.nodes_expanded = 0

        # Basic structures
        self.island_requirements = {
            (r, c): req for r, c, req in puzzle.islands
        }

        # All bridges (sorted by degree heuristic)
        self.all_bridges = BridgeUtils.get_all_bridges_sorted(
            puzzle, self.island_requirements
        )

        # Precomputed structures
        self.island_neighbors = self._precompute_neighbors()
        self.crossing_cache = self._precompute_crossings()

        # Constraint Propagation
        self.propagation = ConstraintPropagation(puzzle)
        self.forced_bridges = {}
        self.forbidden_bridges = set()
        self.min_bridges = {}

    def _precompute_neighbors(self):
        """Precompute neighbor relationships"""
        neighbors = defaultdict(list)
        for bridge_key in self.all_bridges:
            i1, i2 = bridge_key
            neighbors[i1].append(bridge_key)
            neighbors[i2].append(bridge_key)
        return dict(neighbors)

    def _precompute_crossings(self):
        """Precompute crossing relationships"""
        crossings = defaultdict(set)

        for i, bridge1 in enumerate(self.all_bridges):
            for bridge2 in self.all_bridges[i+1:]:
                if BridgeUtils.bridges_cross(bridge1, bridge2):
                    crossings[bridge1].add(bridge2)
                    crossings[bridge2].add(bridge1)

        return dict(crossings)

    def solve(self):
        """Optimized solve with preprocessing"""
        start_time = time.time()
        self.start_time = start_time

        # Step 1: Apply Constraint Propagation (7 rules)
        forced, forbidden, min_bridges = self.propagation.apply_all_rules()
        self.forced_bridges = forced
        self.forbidden_bridges = forbidden
        self.min_bridges = min_bridges

        print(f"  Constraint Propagation Results:")
        print(f"    - Forced bridges: {len(self.forced_bridges)}")
        print(f"    - Forbidden bridges: {len(self.forbidden_bridges)}")
        print(f"    - Minimum constraints: {len(self.min_bridges)}")

        # Step 2: Initialize domains with constraint propagation results
        domains = self._initialize_domains_with_propagation()

        if domains is None:
            self.solve_time = time.time() - start_time
            return False, None

        # Step 3: Initialize island counts
        island_counts = defaultdict(int)

        # Step 4: Start with forced assignments
        assignment = self.forced_bridges.copy()

        # Update counts from forced bridges
        for bridge_key, num_bridges in self.forced_bridges.items():
            i1, i2 = bridge_key
            island_counts[i1] += num_bridges
            island_counts[i2] += num_bridges

        # Step 5: Apply AC-3 for initial consistency
        domains = self._ac3(domains, assignment, island_counts)

        if domains is None:
            self.solve_time = time.time() - start_time
            return False, None

        # Step 6: Backtracking on remaining variables
        success = self._backtrack_optimized(assignment, domains, island_counts)

        self.solve_time = time.time() - start_time

        if success:
            self.solution = {k: v for k, v in assignment.items() if v > 0}
            return True, self.solution
        else:
            return False, None

    def _initialize_domains_with_propagation(self):
        """Initialize domains using constraint propagation results"""
        domains = {}

        for bridge in self.all_bridges:
            # Skip forced bridges
            if bridge in self.forced_bridges:
                continue

            # Skip forbidden bridges
            if bridge in self.forbidden_bridges:
                continue

            # Check minimum requirement
            min_required = self.min_bridges.get(bridge, 0)

            # Check 2-2 isolation
            i1, i2 = bridge
            val1 = self.island_requirements.get(i1, 0)
            val2 = self.island_requirements.get(i2, 0)

            if val1 == 2 and val2 == 2:
                # Cannot have 2 bridges
                domains[bridge] = [0, 1] if min_required == 0 else [1]
            else:
                # Normal domain
                if min_required == 0:
                    domains[bridge] = [0, 1, 2]
                elif min_required == 1:
                    domains[bridge] = [1, 2]
                else:
                    domains[bridge] = [2]

        return domains

    def _ac3(self, domains, assignment, island_counts):
        """AC-3 algorithm for arc consistency"""
        queue = list(domains.keys())

        while queue:
            bridge = queue.pop(0)

            if bridge not in domains or not domains[bridge]:
                continue

            island1, island2 = bridge

            # Revise for both islands
            if self._revise(bridge, island1, domains, assignment, island_counts):
                if not domains[bridge]:
                    return None

                # Add neighbors to queue
                for neighbor_bridge in self.island_neighbors.get(island1, []):
                    if neighbor_bridge != bridge and neighbor_bridge in domains:
                        if neighbor_bridge not in queue:
                            queue.append(neighbor_bridge)

            if self._revise(bridge, island2, domains, assignment, island_counts):
                if not domains[bridge]:
                    return None

                for neighbor_bridge in self.island_neighbors.get(island2, []):
                    if neighbor_bridge != bridge and neighbor_bridge in domains:
                        if neighbor_bridge not in queue:
                            queue.append(neighbor_bridge)

        return domains

    def _revise(self, bridge, island, domains, assignment, island_counts):
        """Revise domain based on island constraints"""
        revised = False
        original_domain = domains[bridge].copy()

        required = self.island_requirements[island]
        current = island_counts[island]

        # Max possible from other unassigned bridges
        max_from_others = 0
        for other_bridge in self.island_neighbors.get(island, []):
            if other_bridge == bridge or other_bridge in assignment:
                continue
            if other_bridge in domains:
                max_from_others += max(domains[other_bridge])

        # Filter domain
        new_domain = []
        for value in domains[bridge]:
            # Check if can satisfy island
            if current + value + max_from_others >= required:
                # Check upper bound
                if current + value <= required:
                    new_domain.append(value)

        if len(new_domain) < len(original_domain):
            domains[bridge] = new_domain
            revised = True

        return revised

    def _backtrack_optimized(self, assignment, domains, island_counts):
        """Optimized backtracking with all techniques"""
        self.nodes_expanded += 1

        # Time out
        if self.nodes_expanded > 100000:
            return False
        if time.time() - self.start_time > 60:
            return False

        # Goal check
        if self._is_goal(assignment, island_counts):
            return True

        # MRV + Degree heuristic
        bridge = self._select_variable_mrv(domains, assignment, island_counts)

        if bridge is None:
            return False

        # LCV ordering
        ordered_values = self._order_values_lcv(
            bridge, domains, assignment, island_counts
        )

        for value in ordered_values:
            # Forward checking
            new_domains, new_counts = self._forward_check(
                bridge, value, domains, assignment, island_counts
            )

            if new_domains is None:
                continue

            # Assign
            assignment[bridge] = value

            # Recurse
            if self._backtrack_optimized(assignment, new_domains, new_counts):
                return True

            # Backtrack
            del assignment[bridge]

        return False

    def _select_variable_mrv(self, domains, assignment, island_counts):
        """MRV + Degree heuristic"""
        best_bridge = None
        min_domain = float('inf')
        max_degree = -1

        for bridge, domain in domains.items():
            if bridge in assignment or not domain:
                continue

            i1, i2 = bridge
            r1 = self.island_requirements[i1] - island_counts[i1]
            r2 = self.island_requirements[i2] - island_counts[i2]

            if r1 <= 0 and r2 <= 0:
                continue

            domain_size = len(domain)
            degree = len([b for b in self.island_neighbors.get(i1, [])
                         if b not in assignment])
            degree += len([b for b in self.island_neighbors.get(i2, [])
                          if b not in assignment])

            if (domain_size < min_domain or
                (domain_size == min_domain and degree > max_degree)):
                min_domain = domain_size
                max_degree = degree
                best_bridge = bridge

        return best_bridge

    def _order_values_lcv(self, bridge, domains, assignment, island_counts):
        """LCV ordering"""
        if bridge not in domains:
            return []

        value_constraints = []

        for value in domains[bridge]:
            constraints = 0

            # Crossing constraints
            if value > 0:
                for cross_bridge in self.crossing_cache.get(bridge, set()):
                    if cross_bridge in domains and cross_bridge not in assignment:
                        constraints += len([v for v in domains[cross_bridge] if v > 0])

            # Neighbor constraints
            i1, i2 = bridge
            for i in [i1, i2]:
                remaining = self.island_requirements[i] - island_counts[i] - value
                for nb in self.island_neighbors.get(i, []):
                    if nb != bridge and nb not in assignment and nb in domains:
                        if remaining < max(domains[nb]):
                            constraints += 1

            value_constraints.append((value, constraints))

        value_constraints.sort(key=lambda x: x[1])
        return [v for v, _ in value_constraints]

    def _forward_check(self, bridge, value, domains, assignment, island_counts):
        """Forward checking with propagation"""
        new_domains = {k: v.copy() for k, v in domains.items()}
        new_counts = island_counts.copy()

        i1, i2 = bridge
        new_counts[i1] += value
        new_counts[i2] += value

        # Check capacity
        if (new_counts[i1] > self.island_requirements[i1] or
            new_counts[i2] > self.island_requirements[i2]):
            return None, None

        # Remove from domains
        if bridge in new_domains:
            del new_domains[bridge]

        # Crossing constraints
        if value > 0:
            for cross_bridge in self.crossing_cache.get(bridge, set()):
                if cross_bridge in new_domains:
                    new_domains[cross_bridge] = [0]

        # Update neighbor domains
        for island_pos in [i1, i2]:
            remaining = self.island_requirements[island_pos] - new_counts[island_pos]

            if remaining < 0:
                return None, None

            for nb in self.island_neighbors.get(island_pos, []):
                if nb == bridge or nb not in new_domains:
                    continue

                new_domain = [v for v in new_domains[nb] if v <= remaining]

                if not new_domain:
                    return None, None

                new_domains[nb] = new_domain

        # AC-3
        new_assignment = assignment.copy()
        new_assignment[bridge] = value
        new_domains = self._ac3(new_domains, new_assignment, new_counts)

        if new_domains is None:
            return None, None

        return new_domains, new_counts

    def _is_goal(self, assignment, island_counts):
        """Goal check"""
        for pos, req in self.island_requirements.items():
            if island_counts[pos] != req:
                return False
        return BridgeUtils.is_connected(assignment, self.puzzle.islands)

    def get_stats(self):
        return {
            'solve_time': self.solve_time,
            'nodes_expanded': self.nodes_expanded,
            'algorithm': 'Optimized Backtracking (CSP + Propagation)',
            'forced_bridges': len(self.forced_bridges),
            'forbidden_bridges': len(self.forbidden_bridges)
        }
        
class BruteForceSolver(VisualizationMixin):
    """Brute force solver"""

    def __init__(self, puzzle: HashiPuzzle):
        self.puzzle = puzzle
        self.solution = None
        self.solve_time = 0
        self.combinations_tried = 0
        self.island_requirements = {
            (r, c): req for r, c, req in puzzle.islands
        }

    def solve(self):
        start_time = time.time()

        all_bridges = BridgeUtils.get_all_bridges_sorted(self.puzzle)
        n = len(all_bridges)
        max_combinations = min(3 ** n, 100000)

        if n > 10:
            print(f"  Warning: {n} bridges = {3**n} combinations, limiting to {max_combinations}")

        for combination in product([0, 1, 2], repeat=n):
            self.combinations_tried += 1

            if self.combinations_tried > max_combinations:
                break

            bridges = {all_bridges[i]: num
                      for i, num in enumerate(combination) if num > 0}

            if self._is_valid_solution(bridges):
                self.solution = bridges
                self.solve_time = time.time() - start_time
                return True, bridges

        self.solve_time = time.time() - start_time
        return False, None

    def _is_valid_solution(self, bridges):
        for island_pos, required in self.island_requirements.items():
            current = sum(num for key, num in bridges.items()
                         if island_pos in key)
            if current != required:
                return False

        bridge_list = list(bridges.keys())
        for i in range(len(bridge_list)):
            for j in range(i + 1, len(bridge_list)):
                if BridgeUtils.bridges_cross(bridge_list[i], bridge_list[j]):
                    return False

        return BridgeUtils.is_connected(bridges, self.puzzle.islands)

    def get_stats(self):
        return {
            'solve_time': self.solve_time,
            'combinations_tried': self.combinations_tried
        }