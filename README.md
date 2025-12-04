# Boba Shop Operations Simulator

**Authors:** Jordan Duan and Valerie He  
**Course:** INDENG 174 - Professor Zheng  
**University:** UC Berkeley
**Project folder:** https://drive.google.com/drive/folders/1BxaLf80BN4tZPZe95iASAND5fyajXonG?usp=drive_link 

**Code:** https://github.com/jordan-duan/174-final 

**ReadMe:**https://github.com/jordan-duan/174-final/blob/main/README.md

**Presentation Video:** [add video link here]

## Overview

This project implements a discrete-event simulation of a boba shop to analyze operational efficiency and identify optimal staffing and automation strategies. We use simulation to model complex, interconnected dynamics that cannot be captured by traditional analytic queueing formulas alone.

## Why We're Doing This

Boba shops face complex operational challenges: time-varying demand patterns, multi-stage service processes, inventory dependencies, and heterogeneous customer preferences. Traditional queueing models (M/M/s) assume constant arrival rates, exponential service times, and no inventory constraints—assumptions that don't hold in real food-service operations. Discrete-event simulation allows us to model these realistic complexities and evaluate how different policies (staffing, automation) affect performance metrics like wait times, throughput, and utilization.

## Quick Start

### Installation
```bash
cd Boba-Sim
pip install -r requirements.txt
```

### Run the Demo
```bash
python comprehensive_demo.py
```

This will:
1. Run Monte Carlo simulations for baseline, automation, and staffing policies
2. Generate comprehensive visualizations and analysis dashboards
3. Prompt you to launch interactive tools with sliders for parameter exploration
4. Save all results to the `results/` directory

### Interactive Tools (with Sliders)
```bash
python comprehensive_demo.py
# When prompted, answer 'y' to launch interactive demos
```

Or run directly:
```python
from interactive_optimizer import InteractiveOptimizer
optimizer = InteractiveOptimizer()
optimizer.create_parameter_explorer()  # Opens interactive dashboard with 6 sliders
```

## Statistical Methods

- **Monte Carlo Replication**: 50-300 independent replications per policy to obtain statistically stable estimates with confidence intervals
- **Little's Law Validation**: Validates simulation accuracy using L = λW (error < 5-10%)
- **M/M/s Queueing Theory**: Theoretical benchmarks for comparison with simulation results
- **Confidence Intervals**: 95% confidence intervals for all performance metrics
- **Distribution Fitting**: Kolmogorov-Smirnov tests for service time distributions
- **Sensitivity Analysis**: Parameter sensitivity studies to identify critical factors

## Utilization of Class Concepts

### Discrete-Event Simulation
- State changes occur only at specific events: customer arrivals, service start/completion, inventory replenishment
- Each customer triggers event sequence: `arrival → cashier.request → barista.request → sealer.request → exit`
- SimPy advances time to next scheduled event (event-driven, not time-stepped)
- SimPy `Resource` objects (cashier, barista, sealer) automatically handle queueing, blocking, and service allocation
- Pearl cooking process implemented as SimPy process triggered when inventory ≤ reorder point
- Models queue buildup, resource contention, event scheduling, simultaneous arrivals, and stage dependencies

### Monte Carlo Replication
- Runs 50-300 independent replications per policy configuration with different random seeds
- Records metrics per replication: mean cycle time, queue wait times, throughput, station utilization, stockout delays
- Computes sample means, variances, 95% confidence intervals, P95 cycle times, and distributional comparisons across policies
- Ensures conclusions are based on statistically stable patterns, not random variation

### Random Variate Generation
- **Exponential Distribution**: Inter-arrival times sampled from `Exponential(1/λ(t))` to model Poisson arrival process
- **Lognormal Distribution**: Service times at cashier, barista, and sealer use lognormal distributions (parameterized by mean and CV) to capture right-skewed, realistic service behavior
- **Categorical Distribution**: Drink type sampled from categorical distribution (Milk Tea 40%, Fruit Tea 30%, Specialty 20%, Simple 10%)
- Complex drinks increase barista workload and cause congestion spikes
- Pearl-requiring drinks drive inventory usage and stockout delays

### Inventory Modeling
- Pearl inventory tracked as continuous quantity, decremented when drinks require pearls
- When inventory ≤ reorder point, batch cook process begins (SimPy process with cooking delay)
- During stockouts, customers requiring pearls wait at barista stage until replenishment completes
- Inventory delays propagate backward through service pipeline, creating nonlinear dependencies impossible in standard queueing models

### Little's Law Validation
- Validates simulation using L = λW where:
  - L = average number of customers in system
  - λ = average arrival rate  
  - W = average time customer spends in system
- Computes relative error `|L - λW| / L` for each replication
- Error < 5-10% indicates: steady state reached, adequate warmup period, no customers lost, event logic consistent

### Time-Varying Arrival Rates λ(t)
- Arrival rates defined piecewise by time of day:
  - Morning (8 AM - 12 PM): 8 customers/hour
  - Lunch (12 PM - 2 PM): 25 customers/hour
  - Afternoon (2 PM - 6 PM): 12 customers/hour
  - Evening (6 PM - 9 PM): 18 customers/hour
  - Night (9 PM - 8 AM): 5 customers/hour
- At simulated time t, interarrival distribution uses `Exponential(1/λ(t))`
- Creates dynamic queue buildup during rushes, variability in utilization, longer wait times when λ(t) exceeds capacity
- Makes system non-stationary, requiring simulation rather than steady-state formulas

## Key Results

Based on simulation analysis:
- **Best Policy**: Staffing (3 baristas) reduces average wait time by 4.7% vs. baseline
- **Average Wait Times**: 5.4-5.7 minutes depending on policy
- **Throughput**: 6.97-7.81 customers/hour
- **Bottleneck**: Barista station typically constrains system performance

## File Structure

```
Boba-Sim/
├── boba_simulator.py          # Main simulation engine (DES, RVG, inventory)
├── advanced_analysis.py       # Little's Law, M/M/s, statistical validation
├── comprehensive_demo.py       # Main demo script
├── interactive_optimizer.py   # Interactive tools with sliders
├── enhanced_visualizations.py # Analysis dashboards
├── experiment_design.py       # Factorial experiments
└── results/                  # Output visualizations and reports
```

## Output Files

Results saved to `results/` directory:
- `dashboard_*.png` - Comprehensive analysis dashboard
- `summary_report.txt` - Key findings and recommendations
- `wait_time_boxplot_*.png` - Wait time distributions
- `optimization_landscape_*.png` - Performance landscape
- Additional analysis visualizations

---
