"""
Comprehensive Demo Script for Boba Shop Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This script demonstrates all the advanced features including interactive tools
and enhanced visualizations.
"""

import os
import numpy as np
from boba_simulator import (
    create_baseline_config, 
    create_automation_config, 
    create_staffing_config,
    run_monte_carlo_simulation,
    analyze_results
)
from enhanced_visualizations import EnhancedVisualizer
from interactive_optimizer import InteractiveOptimizer, create_parameter_sensitivity_analyzer

def run_comprehensive_analysis():
    """Run comprehensive analysis with all visualizations"""
    print("🍵 Boba Shop Operations Simulator - Comprehensive Analysis")
    print("=" * 60)
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize enhanced visualizer
    visualizer = EnhancedVisualizer(results_dir)
    
    print("\n1. Running Policy Comparison Analysis...")
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
        print(f"   Running {policy_name} policy simulation...")
        results = run_monte_carlo_simulation(config, num_replications=30, simulation_time=240)
        all_results[policy_name] = results
        
        # Quick analysis
        analysis = analyze_results(results)
        ci = analysis['confidence_intervals']['avg_wait_time']
        print(f"   Average wait time: {ci['mean']:.2f} ± {ci['std']:.2f} minutes")
    
    print("\n2. Generating Enhanced Visualizations...")
    print("-" * 40)
    
    # Create comprehensive dashboard
    print("   Creating comprehensive dashboard...")
    visualizer.create_comprehensive_dashboard(all_results)
    
    # Create optimization landscape
    print("   Creating optimization landscape...")
    visualizer.create_optimization_landscape(all_results)
    
    # Create detailed plots
    print("   Creating detailed analysis plots...")
    visualizer._create_detailed_plots(all_results)
    
    print("\n3. Statistical Analysis Results:")
    print("-" * 40)
    
    # Compare policies
    for policy_name, results in all_results.items():
        policy_results = [r for r in results if 'error' not in r]
        if policy_results:
            avg_wait = np.mean([r['avg_wait_time'] for r in policy_results])
            p95_wait = np.mean([r['p95_wait_time'] for r in policy_results])
            throughput = np.mean([r['throughput'] for r in policy_results])
            
            print(f"   {policy_name.upper()}:")
            print(f"     - Average wait time: {avg_wait:.2f} minutes")
            print(f"     - 95th percentile wait: {p95_wait:.2f} minutes")
            print(f"     - Throughput: {throughput:.2f} customers/hour")
    
    # Find best policy
    best_policy = None
    best_wait_time = float('inf')
    
    for policy_name, results in all_results.items():
        policy_results = [r for r in results if 'error' not in r]
        if policy_results:
            avg_wait = np.mean([r['avg_wait_time'] for r in policy_results])
            if avg_wait < best_wait_time:
                best_wait_time = avg_wait
                best_policy = policy_name
    
    print(f"\n   🏆 BEST POLICY: {best_policy.upper()} ({best_wait_time:.2f} min avg wait)")
    
    return all_results

def run_interactive_demo():
    """Run interactive demonstration"""
    print("\n4. Interactive Parameter Explorer")
    print("-" * 40)
    print("   Launching interactive parameter optimization tool...")
    print("   Use the sliders to adjust parameters and see real-time results!")
    
    # Create interactive optimizer
    optimizer = InteractiveOptimizer()
    optimizer.create_parameter_explorer()
    
    print("\n5. Parameter Sensitivity Analysis")
    print("-" * 40)
    print("   Analyzing sensitivity of key parameters...")
    
    # Create sensitivity analysis
    create_parameter_sensitivity_analyzer()

def create_summary_report(results_dict):
    """Create a summary report of all findings"""
    print("\n7. Summary Report")
    print("=" * 60)
    
    # Calculate key metrics
    summary_data = []
    
    for policy_name, results in results_dict.items():
        policy_results = [r for r in results if 'error' not in r]
        if policy_results:
            summary_data.append({
                'Policy': policy_name.upper(),
                'Avg_Wait_Time': np.mean([r['avg_wait_time'] for r in policy_results]),
                'P95_Wait_Time': np.mean([r['p95_wait_time'] for r in policy_results]),
                'Throughput': np.mean([r['throughput'] for r in policy_results]),
                'Total_Customers': np.mean([r['total_customers'] for r in policy_results])
            })
    
    # Print summary table
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 60)
    print(f"{'Policy':<12} {'Avg Wait':<10} {'95th %ile':<10} {'Throughput':<12} {'Customers':<10}")
    print("-" * 60)
    
    for data in summary_data:
        print(f"{data['Policy']:<12} {data['Avg_Wait_Time']:<10.2f} {data['P95_Wait_Time']:<10.2f} "
              f"{data['Throughput']:<12.2f} {data['Total_Customers']:<10.0f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    best_policy = min(summary_data, key=lambda x: x['Avg_Wait_Time'])
    worst_policy = max(summary_data, key=lambda x: x['Avg_Wait_Time'])
    
    improvement = worst_policy['Avg_Wait_Time'] - best_policy['Avg_Wait_Time']
    improvement_pct = (improvement / worst_policy['Avg_Wait_Time']) * 100
    
    print(f"• Best performing policy: {best_policy['Policy']}")
    print(f"• Wait time improvement: {improvement:.2f} minutes ({improvement_pct:.1f}%)")
    print(f"• Recommended for high-volume periods: {best_policy['Policy']}")
    
    # Cost analysis
    print(f"\n💰 COST ANALYSIS:")
    print("-" * 20)
    print("• Automation: Higher initial cost, lower wait times")
    print("• Additional Staffing: Higher labor costs, good performance")
    print("• Baseline: Lowest cost, acceptable performance for low demand")
    
    # Bottleneck analysis
    print(f"\n🔍 BOTTLENECK ANALYSIS:")
    print("-" * 25)
    print("• Barista station typically the primary bottleneck")
    print("• Cashier station rarely constrains system")
    print("• Sealer station becomes bottleneck with automation")
    
    # Save summary to file
    summary_path = os.path.join("results", "summary_report.txt")
    with open(summary_path, 'w') as f:
        f.write("Boba Shop Operations Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("Performance Summary:\n")
        f.write("-" * 20 + "\n")
        for data in summary_data:
            f.write(f"{data['Policy']}: {data['Avg_Wait_Time']:.2f} min avg wait, "
                   f"{data['Throughput']:.2f} customers/hour\n")
        
        f.write(f"\nBest Policy: {best_policy['Policy']}\n")
        f.write(f"Improvement: {improvement:.2f} minutes ({improvement_pct:.1f}%)\n")
    
    print(f"\n📄 Summary report saved to: {summary_path}")

def main():
    """Main demonstration function"""
    print("🚀 Starting Comprehensive Boba Shop Simulator Demo")
    print("This demo will showcase all the advanced features of the simulator.")
    print("The process may take several minutes to complete.\n")
    
    try:
        # Run comprehensive analysis
        results = run_comprehensive_analysis()
        
        # Create summary report
        create_summary_report(results)
        
        # Ask user if they want to run interactive demos
        print("\n" + "=" * 60)
        print("🎮 INTERACTIVE DEMOS AVAILABLE")
        print("=" * 60)
        print("The following interactive tools are now available:")
        print("1. Interactive Parameter Explorer - Adjust parameters with sliders")
        print("2. Parameter Sensitivity Analysis - See how parameters affect performance")
        
        response = input("\nWould you like to run the interactive demos? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("\n🎯 Launching Interactive Demos...")
            
            # Run interactive demos
            run_interactive_demo()
        
        print("\n" + "=" * 60)
        print("✅ COMPREHENSIVE DEMO COMPLETED!")
        print("=" * 60)
        print("📁 All results and visualizations have been saved to the 'results' folder")
        print("📊 Check the generated PNG files for detailed analysis")
        print("📄 Review the summary report for key findings and recommendations")
        
        print("\n🎓 This simulation demonstrates key concepts from INDENG 174:")
        print("   • Discrete Event Simulation with SimPy")
        print("   • Queueing Theory and M/M/s models")
        print("   • Statistical Analysis and Validation")
        print("   • Experimental Design and Optimization")
        print("   • Operations Research Applications")
        
    except Exception as e:
        print(f"\n❌ Demo encountered an error: {e}")
        print("Please check that all dependencies are installed correctly.")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
