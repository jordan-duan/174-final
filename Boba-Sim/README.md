# Boba Shop Operations Simulator

**Authors:** Jordan Duan and Valerie He  
**Course:** INDENG 174 - Professor Zheng  
**University:** UC Berkeley

## Project Overview

This project implements a comprehensive discrete-event simulation of a boba shop to analyze operational efficiency and identify optimal staffing and automation strategies. The simulation models customer arrivals, service processes, inventory management, and various operational policies to provide data-driven insights for boba shop optimization.

## Key Features

### 🎯 **Core Simulation Components**
- **Poisson Process Customer Arrivals** with time-varying rates (morning, lunch, afternoon, evening, night)
- **Multi-Station Service Process** (Cashier → Barista → Sealer)
- **Pearl Inventory Management** with batch replenishment and reorder points
- **Multiple Drink Types** with different service time distributions (Milk Tea, Fruit Tea, Specialty, Simple)

### 📊 **Statistical Analysis & Validation**
- **Little's Law Validation** (L = λW) to ensure simulation accuracy
- **M/M/s Queueing Theory** comparisons for theoretical validation
- **Monte Carlo Simulation** with confidence intervals
- **Chi-Square Goodness of Fit** tests for distribution validation
- **Kolmogorov-Smirnov Tests** for service time distributions

### 🔬 **Experimental Design**
- **Factorial Experiments** testing multiple factors simultaneously
- **Sensitivity Analysis** to identify critical parameters
- **Cost-Benefit Analysis** comparing different operational policies
- **Optimization Engine** for finding optimal staffing levels

### 📈 **Advanced Visualizations**
- **Effects Plots** showing main effects and interactions
- **Heatmaps** for two-factor interactions
- **Queue Length Traces** over time
- **Gantt Charts** for customer service timelines
- **Box Plots** for performance comparisons

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation
```bash
# Navigate to the boba-sim directory
cd boba-sim

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Simulation
```python
from boba_simulator import BobaShopSimulator, create_baseline_config

# Create baseline configuration
config = create_baseline_config()

# Run simulation
simulator = BobaShopSimulator(config)
results = simulator.run_simulation(simulation_time=480)  # 8 hours

# View results
print(f"Total customers served: {results['total_customers']}")
print(f"Average wait time: {results['avg_wait_time']:.2f} minutes")
print(f"Throughput: {results['throughput']:.2f} customers/hour")
```

### Monte Carlo Analysis
```python
from boba_simulator import run_monte_carlo_simulation, create_baseline_config

# Run 100 replications
config = create_baseline_config()
results = run_monte_carlo_simulation(config, num_replications=100)

# Analyze results
from boba_simulator import analyze_results
analysis = analyze_results(results)
print(f"95% Confidence Interval: {analysis['confidence_intervals']['avg_wait_time']}")
```

### Policy Comparison
```python
from boba_simulator import (
    create_baseline_config, 
    create_automation_config, 
    create_staffing_config,
    run_monte_carlo_simulation
)

# Compare different policies
policies = {
    'baseline': create_baseline_config(),
    'automation': create_automation_config(),
    'staffing': create_staffing_config()
}

results = {}
for policy_name, config in policies.items():
    results[policy_name] = run_monte_carlo_simulation(config, num_replications=50)

# Create visualizations
from boba_simulator import create_visualizations
create_visualizations(results)
```

### Factorial Experiment
```python
from experiment_design import run_factorial_experiment

# Run factorial experiment
experiment = run_factorial_experiment()

# Analyze main effects
main_effects = experiment.analyze_main_effects()
print("Main Effects Analysis:")
for response_var, effects in main_effects.items():
    print(f"\n{response_var}:")
    print(effects)
```

### Advanced Analysis
```python
from advanced_analysis import run_comprehensive_analysis

# Run comprehensive statistical analysis
run_comprehensive_analysis()
```

### Interactive Tools
```python
from interactive_optimizer import InteractiveOptimizer
from enhanced_visualizations import EnhancedVisualizer

# Create interactive parameter explorer with sliders
optimizer = InteractiveOptimizer()
optimizer.create_parameter_explorer()

# Create comprehensive visualizations
visualizer = EnhancedVisualizer()
visualizer.create_comprehensive_dashboard(results_dict)
```

### Animated Simulations
```python
from simulation_animation import BobaShopAnimator

# Create animated shop layout
config = create_baseline_config()
animator = BobaShopAnimator(config)
animation = animator.create_animated_simulation(duration=60)
```

### Complete Demo
```python
# Run the comprehensive demo with all features
python comprehensive_demo.py
```

## Configuration Options

### Service Station Capacities
```python
config = {
    'cashier_capacity': 1,      # Number of cashiers
    'barista_capacity': 2,      # Number of baristas
    'sealer_capacity': 1,       # Number of sealers
}
```

### Inventory Management
```python
config = {
    'initial_pearl_inventory': 50,    # Starting pearl count
    'pearl_reorder_point': 10,        # When to reorder
    'pearl_batch_size': 30,           # Batch size for cooking
    'pearl_cook_time': 15,            # Cooking time (minutes)
}
```

### Arrival Rate Patterns
The simulation includes realistic time-varying arrival rates:
- **Morning (8 AM - 12 PM):** 8 customers/hour
- **Lunch (12 PM - 2 PM):** 25 customers/hour
- **Afternoon (2 PM - 6 PM):** 12 customers/hour
- **Evening (6 PM - 9 PM):** 18 customers/hour
- **Night (9 PM - 8 AM):** 5 customers/hour

## Operational Policies

### 1. **Baseline Policy**
- 1 Cashier, 2 Baristas, 1 Sealer
- Standard service times
- Basic inventory management

### 2. **Automation Policy**
- Auto-sealer increases sealer capacity to 2
- Reduced service time variability
- Higher initial investment cost

### 3. **Staffing Policy**
- Additional barista (3 total)
- Higher labor costs
- Improved service capacity

## Key Results & Insights

### Performance Metrics
- **Average Wait Time:** Typically 3-8 minutes depending on policy
- **95th Percentile Wait Time:** 8-15 minutes
- **Throughput:** 15-25 customers/hour during peak times
- **System Utilization:** 60-90% depending on station

### Policy Recommendations
Based on simulation results:
1. **Automation** provides best cost-per-minute-saved for high-volume periods
2. **Additional Staffing** most effective during moderate demand
3. **Baseline** sufficient for low-demand periods

### Bottleneck Analysis
- **Barista station** typically the primary bottleneck
- **Cashier station** rarely constrains system
- **Sealer station** becomes bottleneck with automation

## Statistical Validation

### Little's Law Validation
The simulation consistently validates Little's Law (L = λW) with:
- Average error < 5%
- Maximum error < 10%
- Validation passed for all test scenarios

### Distribution Fitting
- **Service Times:** Lognormal distribution (validated with KS test)
- **Inter-arrival Times:** Exponential distribution
- **Wait Times:** Approximated by M/M/s queueing theory

## File Structure

```
boba-sim/
├── boba_simulator.py          # Main simulation engine
├── experiment_design.py       # Factorial experiments and DOE
├── advanced_analysis.py       # Statistical validation and analysis
├── enhanced_visualizations.py # Advanced visualizations and dashboards
├── interactive_optimizer.py   # Interactive parameter optimization tools
├── simulation_animation.py    # Animated simulation visualizations
├── comprehensive_demo.py      # Complete demonstration script
├── demo.py                   # Basic demo script
├── test_installation.py      # Installation verification
├── requirements.txt           # Python dependencies
├── results/                  # Output directory for visualizations
└── README.md                 # This file
```

## Output Files

The simulation generates comprehensive output files in the `results/` directory:
- `dashboard_*.png` - Comprehensive analysis dashboard
- `optimization_landscape_*.png` - 3D optimization landscape
- `wait_time_boxplot_*.png` - Wait time distribution analysis
- `throughput_boxplot_*.png` - Throughput distribution analysis
- `radar_chart_*.png` - Multi-dimensional performance comparison
- `time_series_analysis_*.png` - Time series analysis plots
- `summary_report.txt` - Text summary of key findings

## Research Questions Addressed

1. **Which operational changes most effectively reduce customer wait times?**
   - Automation vs. additional staffing
   - Impact of different service time distributions

2. **How does system performance vary with demand patterns?**
   - Peak vs. off-peak performance
   - Sensitivity to arrival rate changes

3. **What is the optimal staffing configuration for different scenarios?**
   - Cost-benefit analysis
   - Bottleneck identification

4. **How robust are the results to parameter uncertainty?**
   - Sensitivity analysis
   - Confidence interval estimation

## Future Enhancements

- **Real-time Animation** using Pygame
- **Machine Learning** integration for demand forecasting
- **Multi-location** simulation capabilities
- **Advanced Inventory** optimization algorithms
- **Customer Satisfaction** modeling

## Academic Context

This project demonstrates key concepts from INDENG 174:
- **Discrete Event Simulation** using SimPy
- **Queueing Theory** and M/M/s models
- **Statistical Analysis** and validation methods
- **Experimental Design** and factorial analysis
- **Operations Research** optimization techniques

## Contact

For questions about this project, please contact:
- **Jordan Duan** - [Email]
- **Valerie He** - [Email]

---

*This simulation provides a comprehensive framework for analyzing boba shop operations and can be adapted for other service industry applications.*
