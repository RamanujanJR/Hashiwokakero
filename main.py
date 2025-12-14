import os
from experiments import ExperimentCoordinator

def main():
    # Setup directory structure check
    if not os.path.exists('inputs'):
        print("ERROR: Directory 'inputs' not found. Please create it and add input-xx.txt files.")
        return

    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)

    # Initialize Coordinator
    coordinator = ExperimentCoordinator()

    # Phase 1: Standard Tests
    results_standard = coordinator.run_phase1()
    
    # Phase 2: Large Tests
    results_xlarge = coordinator.run_phase2()

    # Save & Analyze
    results_combined = coordinator.save_results(results_standard, results_xlarge)
    
    # Visualization & Analysis
    coordinator.analyze_standard_results(results_standard)
    coordinator.analyze_pysat_scalability(results_xlarge)
    coordinator.visualize_sample_solution()
    
    # Save all solutions to output files
    coordinator.save_all_solutions()

if __name__ == "__main__":
    main()