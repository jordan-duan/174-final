"""
Demo Script for Boba Shop Operations Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This script demonstrates the basic functionality of the boba shop simulator.
"""

from boba_simulator import (
    BobaShopSimulator, 
    create_baseline_config, 
    create_automation_config, 
    create_staffing_config,
    run_monte_carlo_simulation,
    analyze_results,
    create_visualizations
)

def run_basic_demo():
    """Run a basic demonstration of the simulator"""
    print("Boba Shop Operations Simulator - Basic Demo")
    print("=" * 50)
    
    # Create baseline configuration
    config = create_baseline_config()
    
    # Run single simulation
    print("\n1. Running Single Simulation (Baseline Policy)")
    simulator = BobaShopSimulator(config)
    results = simulator.run_simulation(simulation_time=120)  # 2 hours
    
    print(f"   Total customers served: {results['total_customers']}")
    print(f"   Average wait time: {results['avg_wait_time']:.2f} minutes")
    print(f"   Throughput: {results['throughput']:.2f} customers/hour")
    print(f"   95th percentile wait time: {results['p95_wait_time']:.2f} minutes")
    
    return results

def run_policy_comparison():
    """Compare different operational policies"""
    print("\n2. Policy Comparison (Monte Carlo Analysis)")
    print("-" * 40)
    
    # Define policies
    policies = {
        'baseline': create_baseline_config(),
        'automation': create_automation_config(),
        'staffing': create_staffing_config()
    }
    
    # Run simulations for each policy
    all_results = {}
    
    for policy_name, config in policies.items():
        print(f"\n   Running {policy_name} policy...")
        results = run_monte_carlo_simulation(config, num_replications=20, simulation_time=120)
        all_results[policy_name] = results
        
        # Analyze results
        analysis = analyze_results(results)
        ci = analysis['confidence_intervals']['avg_wait_time']
        
        print(f"   Average wait time: {ci['mean']:.2f} ± {ci['std']:.2f} minutes")
        print(f"   95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    
    # Create visualizations
    print("\n   Generating comparison visualizations...")
    create_visualizations(all_results)
    
    return all_results

def run_quick_analysis():
    """Run a quick statistical analysis"""
    print("\n3. Quick Statistical Analysis")
    print("-" * 30)
    
    # Import analysis modules
    try:
        from advanced_analysis import LittleLawValidator
        
        # Run baseline simulation
        config = create_baseline_config()
        results = run_monte_carlo_simulation(config, num_replications=10, simulation_time=120)
        
        # Validate Little's Law
        validator = LittleLawValidator()
        validation = validator.validate_littles_law(results)
        
        if 'error' not in validation:
            print(f"   Little's Law Validation:")
            print(f"   - Average error: {validation['avg_error']:.2f}%")
            print(f"   - Max error: {validation['max_error']:.2f}%")
            print(f"   - Validation passed: {validation['validation_passed']}")
        
    except ImportError:
        print("   Advanced analysis modules not available")
    
    return results

def main():
    """Main demo function"""
    print("Starting Boba Shop Simulator Demo...")
    print("This demo will run several simulations to demonstrate the system capabilities.")
    print("Note: This may take a few minutes to complete.\n")
    
    try:
        # Run basic demo
        basic_results = run_basic_demo()
        
        # Run policy comparison
        policy_results = run_policy_comparison()
        
        # Run quick analysis
        analysis_results = run_quick_analysis()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey findings:")
        
        # Extract key insights
        baseline_wait = policy_results['baseline'][0]['avg_wait_time']
        automation_wait = policy_results['automation'][0]['avg_wait_time']
        staffing_wait = policy_results['staffing'][0]['avg_wait_time']
        
        print(f"- Baseline policy average wait: {baseline_wait:.2f} minutes")
        print(f"- Automation policy average wait: {automation_wait:.2f} minutes")
        print(f"- Staffing policy average wait: {staffing_wait:.2f} minutes")
        
        # Determine best policy
        best_policy = min([
            ('baseline', baseline_wait),
            ('automation', automation_wait),
            ('staffing', staffing_wait)
        ], key=lambda x: x[1])
        
        print(f"- Best performing policy: {best_policy[0]} ({best_policy[1]:.2f} min wait)")
        
        print("\nCheck the generated visualization files for detailed results!")
        
    except Exception as e:
        print(f"\nDemo encountered an error: {e}")
        print("Please check that all dependencies are installed correctly.")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
