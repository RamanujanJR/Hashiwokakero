import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import HashiPuzzle
from solvers import PySATSolver, AStarSolver, NaiveBacktrackingSolver, OptimizedBacktrackingSolver, BruteForceSolver 

class ExperimentRunner:
    """Run experiments across multiple solvers and test cases"""

    def __init__(self, input_files):
        self.input_files = input_files
        self.results = []

    def run_all_experiments(self):
        """Run all solvers on all test cases"""

        solvers_config = [
            ('PySAT', lambda p: PySATSolver(p)),
            ('A* (Simple)', lambda p: AStarSolver(p, use_advanced_heuristic=False)),
            ('A* (Advanced)', lambda p: AStarSolver(p, use_advanced_heuristic=True)),
            ('Naive Backtracking', lambda p: NaiveBacktrackingSolver(p)),
            ('Optimized Backtracking', lambda p: OptimizedBacktrackingSolver(p)),
            ('Brute Force', lambda p: BruteForceSolver(p))
        ]

        for input_file in self.input_files:
            print(f"\n{'='*80}")
            print(f"Processing: {input_file}")
            print(f"{'='*80}")

            try:
                puzzle = HashiPuzzle.read_from_file(input_file)
                grid_size = f"{puzzle.rows}x{puzzle.cols}"
                num_islands = len(puzzle.islands)

                print(f"Grid: {grid_size}, Islands: {num_islands}")

                for solver_name, solver_factory in solvers_config:
                    print(f"\n  Testing {solver_name}...")

                    try:
                        solver = solver_factory(puzzle)
                        success, solution = solver.solve()
                        stats = solver.get_stats()

                        result = {
                            'input_file': input_file,
                            'grid_size': grid_size,
                            'num_islands': num_islands,
                            'solver': solver_name,
                            'success': success,
                            'solve_time': stats.get('solve_time', 0),
                            'nodes_expanded': stats.get('nodes_expanded',
                                                       stats.get('combinations_tried', 0)),
                            'num_variables': stats.get('num_variables', 0),
                            'num_clauses': stats.get('num_clauses', 0)
                        }

                        self.results.append(result)

                        status = "✓ Solved" if success else "✗ Failed"
                        time_str = f"{stats.get('solve_time', 0):.4f}s"
                        print(f"    {status} in {time_str}")

                        if 'nodes_expanded' in stats:
                            print(f"    Nodes expanded: {stats['nodes_expanded']:,}")
                        if 'combinations_tried' in stats:
                            print(f"    Combinations tried: {stats['combinations_tried']:,}")

                    except Exception as e:
                        print(f"    ✗ Error: {str(e)}")
                        self.results.append({
                            'input_file': input_file,
                            'grid_size': grid_size,
                            'num_islands': num_islands,
                            'solver': solver_name,
                            'success': False,
                            'solve_time': 0,
                            'nodes_expanded': 0,
                            'num_variables': 0,
                            'num_clauses': 0
                        })

            except Exception as e:
                print(f"  ✗ Failed to load puzzle: {str(e)}")

        return pd.DataFrame(self.results)

    def run_pysat_only(self, input_files):
        """Run PySAT only on extra large puzzles"""
        results = []

        for input_file in input_files:
            print(f"\n{'='*80}")
            print(f"Processing: {input_file}")
            print(f"{'='*80}")

            try:
                puzzle = HashiPuzzle.read_from_file(input_file)
                grid_size = f"{puzzle.rows}x{puzzle.cols}"
                num_islands = len(puzzle.islands)

                print(f"Grid: {grid_size}, Islands: {num_islands}")
                print(f"\n  Testing PySAT (only viable solver for this size)...")

                try:
                    solver = PySATSolver(puzzle)
                    success, solution = solver.solve()
                    stats = solver.get_stats()

                    result = {
                        'input_file': input_file,
                        'grid_size': grid_size,
                        'num_islands': num_islands,
                        'solver': 'PySAT',
                        'success': success,
                        'solve_time': stats.get('solve_time', 0),
                        'nodes_expanded': 0,
                        'num_variables': stats.get('num_variables', 0),
                        'num_clauses': stats.get('num_clauses', 0)
                    }

                    results.append(result)

                    status = "✓ Solved" if success else "✗ Failed"
                    time_str = f"{stats.get('solve_time', 0):.4f}s"
                    print(f"    {status} in {time_str}")
                    print(f"    Variables: {stats.get('num_variables', 0):,}")
                    print(f"    Clauses: {stats.get('num_clauses', 0):,}")

                except Exception as e:
                    print(f"    ✗ Error: {str(e)}")
                    results.append({
                        'input_file': input_file,
                        'grid_size': grid_size,
                        'num_islands': num_islands,
                        'solver': 'PySAT',
                        'success': False,
                        'solve_time': 0,
                        'nodes_expanded': 0,
                        'num_variables': 0,
                        'num_clauses': 0
                    })

            except Exception as e:
                print(f"  ✗ Failed to load puzzle: {str(e)}")

        return pd.DataFrame(results)
    
class ResultsAnalyzer:
    """Analyze and visualize experiment results"""

    def __init__(self, df):
        self.df = df

    def print_summary_table(self):
        """Print summary statistics table"""
        print("\n" + "="*100)
        print("SUMMARY RESULTS")
        print("="*100)

        # Define proper column order
        grid_sizes_ordered = [
            '4x4', '5x5', '6x6', '7x7', '8x8', '9x9', '10x10', '11x11',
            '12x12', '13x13', '14x14', '15x15', '16x16', '17x17', '18x18',
            '19x19', '20x20', '21x21', '22x22', '23x23', '24x24', '25x25',
            '30x30', '35x35', '40x40'
        ]

        summary = self.df.pivot_table(
            index='solver',
            columns='grid_size',
            values='solve_time',
            aggfunc='mean'
        )

        # Reorder columns
        existing_cols = [col for col in grid_sizes_ordered if col in summary.columns]
        summary = summary[existing_cols]

        print("\nAverage Solve Time (seconds):")
        print(summary.to_string())

        success_rate = self.df.pivot_table(
            index='solver',
            columns='grid_size',
            values='success',
            aggfunc='mean'
        ) * 100

        # Reorder columns
        existing_cols = [col for col in grid_sizes_ordered if col in success_rate.columns]
        success_rate = success_rate[existing_cols]

        nodes = self.df[self.df['nodes_expanded'] > 0].pivot_table(
            index='solver',
            columns='grid_size',
            values='nodes_expanded',
            aggfunc='mean'
        )

        # Reorder columns
        if len(nodes) > 0:
            existing_cols = [col for col in grid_sizes_ordered if col in nodes.columns]
            nodes = nodes[existing_cols]

        print("\n\nAverage Nodes Expanded:")
        print(nodes.to_string())

    def create_comparison_plots(self):
        """Create comprehensive comparison visualizations"""

        # Plot 1: Solve Time vs Grid Size
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        successful = self.df[self.df['success'] == True]

        for solver in successful['solver'].unique():
            solver_data = successful[successful['solver'] == solver]
            ax1.plot(solver_data['num_islands'], solver_data['solve_time'],
                    marker='o', label=solver, linewidth=2)

        ax1.set_xlabel('Number of Islands', fontsize=12)
        ax1.set_ylabel('Solve Time (seconds)', fontsize=12)
        ax1.set_title('Solve Time vs Problem Size', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        plt.tight_layout()
        plt.show()

        # Plot 2: Nodes Expanded vs Grid Size
        plt.figure(figsize=(10, 6))
        ax2 = plt.gca()
        nodes_data = successful[successful['nodes_expanded'] > 0]

        for solver in nodes_data['solver'].unique():
            solver_data = nodes_data[nodes_data['solver'] == solver]
            ax2.plot(solver_data['num_islands'], solver_data['nodes_expanded'],
                    marker='s', label=solver, linewidth=2)

        ax2.set_xlabel('Number of Islands', fontsize=12)
        ax2.set_ylabel('Nodes Expanded', fontsize=12)
        ax2.set_title('Search Space Explored', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        plt.tight_layout()
        plt.show()

        # Plot 3: Success Rate
        plt.figure(figsize=(10, 6))
        ax3 = plt.gca()
        success_by_solver = self.df.groupby('solver')['success'].mean() * 100
        colors = plt.cm.Set3(range(len(success_by_solver)))
        bars = ax3.bar(range(len(success_by_solver)), success_by_solver.values, color=colors)
        ax3.set_xticks(range(len(success_by_solver)))
        ax3.set_xticklabels(success_by_solver.index, rotation=45, ha='right')
        ax3.set_ylabel('Success Rate (%)', fontsize=12)
        ax3.set_title('Overall Success Rate by Solver', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()

        # Plot 4: Time Comparison Heatmap
        plt.figure(figsize=(12, 8))
        ax4 = plt.gca()

        # Define proper column order
        grid_sizes_ordered = [
            '4x4', '5x5', '6x6', '7x7', '8x8', '9x9', '10x10', '11x11',
            '12x12', '13x13', '14x14', '15x15', '16x16', '17x17', '18x18',
            '19x19', '20x20', '21x21', '22x22', '23x23', '24x24', '25x25'
        ]

        time_pivot = successful.pivot_table(
            index='solver',
            columns='grid_size',
            values='solve_time',
            aggfunc='mean'
        )

        # Reorder columns to proper sequence
        existing_cols = [col for col in grid_sizes_ordered if col in time_pivot.columns]
        time_pivot = time_pivot[existing_cols]

        # Create heatmap with better formatting
        sns.heatmap(time_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                   ax=ax4, cbar_kws={'label': 'Time (s)'},
                   annot_kws={'fontsize': 7})  # Smaller font for annotations

        ax4.set_title('Solve Time Heatmap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Grid Size', fontsize=11)
        ax4.set_ylabel('Solver', fontsize=11)

        # Rotate x-axis labels for better readability
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.show()

        # Plot 5: Efficiency Comparison (Islands/Second)
        plt.figure(figsize=(10, 6))
        ax5 = plt.gca()
        successful.loc[:, 'efficiency'] = successful['num_islands'] / successful['solve_time']
        efficiency = successful.groupby('solver')['efficiency'].mean().sort_values()

        bars = ax5.barh(range(len(efficiency)), efficiency.values,
                       color=plt.cm.viridis(np.linspace(0, 1, len(efficiency))))
        ax5.set_yticks(range(len(efficiency)))
        ax5.set_yticklabels(efficiency.index)
        ax5.set_xlabel('Islands Solved per Second', fontsize=12)
        ax5.set_title('Solver Efficiency', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center', fontsize=10)
        plt.tight_layout()
        plt.show()

        # Plot 6: Scalability Analysis
        plt.figure(figsize=(10, 6))
        ax6 = plt.gca()

        for solver in ['PySAT', 'A* (Advanced)', 'Optimized Backtracking']:
            solver_data = successful[successful['solver'] == solver]
            if len(solver_data) > 0:
                ax6.scatter(solver_data['num_islands'],
                          solver_data['solve_time'],
                          s=100, alpha=0.6, label=solver)

        ax6.set_xlabel('Number of Islands', fontsize=12)
        ax6.set_ylabel('Solve Time (seconds)', fontsize=12)
        ax6.set_title('Scalability: Top 3 Solvers', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        plt.tight_layout()
        plt.show()

    def create_detailed_comparison(self):
        """Create detailed per-solver analysis"""

        solvers = self.df['solver'].unique()
        n_solvers = len(solvers)

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, solver in enumerate(solvers):
            if idx >= len(axes):
                break

            ax = axes[idx]
            solver_data = self.df[self.df['solver'] == solver]

            # Separate successful and failed
            success = solver_data[solver_data['success'] == True]
            failed = solver_data[solver_data['success'] == False]

            if len(success) > 0:
                ax.scatter(success['num_islands'], success['solve_time'],
                          color='green', s=100, alpha=0.6, label='Success')

            if len(failed) > 0:
                ax.scatter(failed['num_islands'], [0.001] * len(failed),
                          color='red', s=100, marker='x', alpha=0.6, label='Failed')

            ax.set_xlabel('Number of Islands', fontsize=11)
            ax.set_ylabel('Solve Time (seconds)', fontsize=11)
            ax.set_title(f'{solver}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        # Hide unused subplots
        for idx in range(n_solvers, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """Generate comprehensive text report"""

        print("\n" + "="*100)
        print("DETAILED ANALYSIS REPORT")
        print("="*100)

        # Overall statistics
        print("\n1. OVERALL STATISTICS")
        print("-" * 100)
        total_tests = len(self.df)
        total_success = self.df['success'].sum()
        avg_time = self.df[self.df['success']]['solve_time'].mean()

        print(f"Total test cases: {total_tests}")
        print(f"Successful solves: {total_success} ({total_success/total_tests*100:.1f}%)")
        print(f"Average solve time (successful): {avg_time:.4f}s")

        # Per-solver analysis
        print("\n2. PER-SOLVER PERFORMANCE")
        print("-" * 100)

        for solver in self.df['solver'].unique():
            solver_data = self.df[self.df['solver'] == solver]
            success_data = solver_data[solver_data['success'] == True]

            print(f"\n{solver}:")
            print(f"  Success rate: {len(success_data)}/{len(solver_data)} " +
                  f"({len(success_data)/len(solver_data)*100:.1f}%)")

            if len(success_data) > 0:
                print(f"  Avg solve time: {success_data['solve_time'].mean():.4f}s")
                print(f"  Min solve time: {success_data['solve_time'].min():.4f}s")
                print(f"  Max solve time: {success_data['solve_time'].max():.4f}s")

                if success_data['nodes_expanded'].sum() > 0:
                    print(f"  Avg nodes expanded: {success_data['nodes_expanded'].mean():.0f}")

        # Problem size analysis
        print("\n3. PROBLEM SIZE ANALYSIS")
        print("-" * 100)

        size_groups = self.df.groupby('grid_size')

        for size, group in size_groups:
            success_rate = group['success'].mean() * 100
            avg_time = group[group['success']]['solve_time'].mean()

            print(f"\n{size}:")
            print(f"  Success rate: {success_rate:.1f}%")
            if not np.isnan(avg_time):
                print(f"  Average solve time: {avg_time:.4f}s")

            best_solver = group[group['success']].groupby('solver')['solve_time'].mean().idxmin()
            if not pd.isna(best_solver):
                print(f"  Best solver: {best_solver}")
                
class PySATAnalyzer:
    """Specialized analyzer for PySAT scalability"""

    def __init__(self, pysat_df):
        self.df = pysat_df

    def print_performance_table(self):
        """Print detailed PySAT performance table"""
        print("\nPySAT Performance Across All Puzzle Sizes:")
        print("-" * 100)
        print(f"{'Grid Size':<12} {'Islands':<10} {'Success':<10} {'Time (s)':<12} "
              f"{'Variables':<12} {'Clauses':<12}")
        print("-" * 100)

        for _, row in self.df.iterrows():
            success_str = "✓" if row['success'] else "✗"
            print(f"{row['grid_size']:<12} {row['num_islands']:<10} {success_str:<10} "
                  f"{row['solve_time']:<12.4f} {row['num_variables']:<12,} "
                  f"{row['num_clauses']:<12,}")

    def create_scalability_plots(self):
        """Create PySAT-specific scalability visualizations"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        success = self.df[self.df['success'] == True]

        # Plot 1: Solve time vs Islands
        ax1 = axes[0, 0]
        ax1.plot(success['num_islands'], success['solve_time'],
                 marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Number of Islands', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_title('PySAT: Solve Time vs Problem Size', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Annotate extreme sizes
        for _, row in success.iterrows():
            if row['num_islands'] >= 300:
                ax1.annotate(f"{row['grid_size']}",
                            (row['num_islands'], row['solve_time']),
                            textcoords="offset points", xytext=(0,10),
                            ha='center', fontsize=9, fontweight='bold')

        # Plot 2: Variables & Clauses vs Islands
        ax2 = axes[0, 1]
        ax2.plot(success['num_islands'], success['num_variables'],
                 marker='s', linewidth=2, markersize=8, color='#A23B72', label='Variables')
        ax2.plot(success['num_islands'], success['num_clauses'],
                 marker='^', linewidth=2, markersize=8, color='#F18F01', label='Clauses')
        ax2.set_xlabel('Number of Islands', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_title('PySAT: CNF Complexity Growth', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # Plot 3: Efficiency (Islands/Second)
        ax3 = axes[1, 0]
        success['efficiency'] = success['num_islands'] / success['solve_time']
        colors = plt.cm.viridis(np.linspace(0, 1, len(success)))
        ax3.scatter(success['num_islands'], success['efficiency'],
                   s=100, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)
        ax3.set_xlabel('Number of Islands', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Islands Solved per Second', fontsize=12, fontweight='bold')
        ax3.set_title('PySAT: Solver Efficiency', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Performance by size category
        ax4 = axes[1, 1]

        success_sorted = success.sort_values('num_islands')
        islands = success_sorted['num_islands'].values
        times = success_sorted['solve_time'].values

        size_ranges = [
            (0, 50, 'Small (< 50)'),
            (50, 100, 'Medium (50-100)'),
            (100, 200, 'Large (100-200)'),
            (200, 600, 'XLarge (200+)')
        ]

        for min_i, max_i, label in size_ranges:
            mask = (islands >= min_i) & (islands < max_i)
            if mask.any():
                ax4.scatter(islands[mask], times[mask], s=100, alpha=0.7, label=label)

        ax4.set_xlabel('Number of Islands', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Solve Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('PySAT: Performance by Problem Size Category',
                     fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        ax4.set_xscale('log')

        plt.tight_layout()
        plt.show()

    def print_statistical_summary(self):
        """Print statistical summary for PySAT"""
        success = self.df[self.df['success'] == True]

        print("\nPySAT Statistical Summary:")
        print("-" * 100)
        print(f"Total puzzles tested: {len(self.df)}")
        print(f"Successful solves: {self.df['success'].sum()}")
        print(f"Success rate: {self.df['success'].mean() * 100:.1f}%")
        print(f"\nSmallest puzzle: {self.df['num_islands'].min()} islands")
        print(f"Largest puzzle: {self.df['num_islands'].max()} islands")

        if len(success) > 0:
            print(f"\nFastest solve: {success['solve_time'].min():.4f}s "
                  f"({success.loc[success['solve_time'].idxmin(), 'grid_size']})")
            print(f"Slowest solve: {success['solve_time'].max():.4f}s "
                  f"({success.loc[success['solve_time'].idxmax(), 'grid_size']})")
            print(f"Average solve time: {success['solve_time'].mean():.4f}s")
            print(f"\nMax variables: {success['num_variables'].max():,} "
                  f"({success.loc[success['num_variables'].idxmax(), 'grid_size']})")
            print(f"Max clauses: {success['num_clauses'].max():,} "
                  f"({success.loc[success['num_clauses'].idxmax(), 'grid_size']})")
            
class ExperimentCoordinator:
    """Coordinate the entire experimental workflow"""

    def __init__(self):
        self.input_files_standard = [
            'inputs/input-01.txt', 'inputs/input-02.txt', 'inputs/input-03.txt', 'inputs/input-04.txt',
            'inputs/input-05.txt', 'inputs/input-06.txt', 'inputs/input-07.txt', 'inputs/input-08.txt',
            'inputs/input-09.txt', 'inputs/input-10.txt', 'inputs/input-11.txt', 'inputs/input-12.txt',
            'inputs/input-13.txt', 'inputs/input-14.txt', 'inputs/input-15.txt', 'inputs/input-16.txt',
            'inputs/input-17.txt', 'inputs/input-18.txt', 'inputs/input-19.txt', 'inputs/input-20.txt',
            'inputs/input-21.txt', 'inputs/input-22.txt'
        ]

        self.input_files_xlarge = [
            'inputs/input-23.txt',
            'inputs/input-24.txt',
            'inputs/input-25.txt'
        ]

    def run_phase1(self):
        """Phase 1: Standard test suite with all solvers"""
        print("\n" + "="*100)
        print("PHASE 1: STANDARD TEST SUITE (4x4 to 25x25)")
        print("Testing all 6 solvers on 22 puzzles")
        print("="*100)

        runner = ExperimentRunner(self.input_files_standard)
        return runner.run_all_experiments()

    def run_phase2(self):
        """Phase 2: Extra large puzzles with PySAT only"""
        print("\n" + "="*100)
        print("PHASE 2: EXTRA LARGE PUZZLES (30x30 to 40x40)")
        print("Testing PySAT only on 3 extreme puzzles")
        print("="*100)

        runner = ExperimentRunner(self.input_files_xlarge)
        return runner.run_pysat_only(self.input_files_xlarge)

    def save_results(self, results_standard, results_xlarge):
        """Save experiment results to CSV files"""
        results_combined = pd.concat([results_standard, results_xlarge],
                                     ignore_index=True)

        results_combined.to_csv('solver_results_complete.csv', index=False)
        print("\n✓ Complete results saved to 'solver_results_complete.csv'")

        results_standard.to_csv('solver_results_standard.csv', index=False)
        print("✓ Standard test results saved to 'solver_results_standard.csv'")

        return results_combined

    def analyze_standard_results(self, results_standard):
        """Analyze standard test suite results"""
        print("\n" + "="*100)
        print("ANALYSIS: STANDARD TEST SUITE (All Solvers)")
        print("="*100)

        analyzer = ResultsAnalyzer(results_standard)

        print("\nGenerating standard test visualizations...")
        analyzer.create_comparison_plots()
        analyzer.create_detailed_comparison()

        analyzer.print_summary_table()
        analyzer.generate_report()

    def analyze_pysat_scalability(self, results_combined):
        """Analyze PySAT scalability across all puzzle sizes"""
        print("\n" + "="*100)
        print("ANALYSIS: PYSAT SCALABILITY (4x4 to 40x40)")
        print("="*100)

        pysat_results = results_combined[results_combined['solver'] == 'PySAT']
        pysat_analyzer = PySATAnalyzer(pysat_results)

        pysat_analyzer.print_performance_table()

        print("\nGenerating PySAT scalability visualization...")
        pysat_analyzer.create_scalability_plots()

        pysat_analyzer.print_statistical_summary()

    def visualize_sample_solution(self):
        """Visualize a sample solution"""
        print("\n" + "="*100)
        print("SAMPLE SOLUTION VISUALIZATION")
        print("="*100)

        try:
            puzzle = HashiPuzzle.read_from_file('input-07.txt')
            solver = AStarSolver(puzzle, use_advanced_heuristic=True)
            success, solution = solver.solve()

            if success:
                solver.visualize_solution(title="A* Solution for 10x10 Puzzle (35 islands)")
            else:
                print("A* could not solve, trying PySAT...")
                solver = PySATSolver(puzzle)
                success, solution = solver.solve()
                if success:
                    solver.visualize_solution(title="PySAT Solution for 10x10 Puzzle (35 islands)")
        except Exception as e:
            print(f"Visualization error: {str(e)}")