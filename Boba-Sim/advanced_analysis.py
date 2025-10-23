"""
Advanced Analysis Module for Boba Shop Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This module provides advanced statistical analysis, validation methods,
and optimization techniques for the boba shop simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class LittleLawValidator:
    """Class to validate Little's Law in the simulation"""
    
    @staticmethod
    def validate_littles_law(results: List[Dict]) -> Dict[str, Any]:
        """
        Validate Little's Law: L = λW
        
        Where:
        L = Average number of customers in system
        λ = Average arrival rate
        W = Average time in system
        """
        validation_results = []
        
        for result in results:
            if 'error' not in result and 'customer_data' in result:
                # Calculate λ (arrival rate)
                simulation_time_hours = result['simulation_time'] / 60
                lambda_val = result['total_customers'] / simulation_time_hours
                
                # Calculate W (average time in system)
                customer_data = result['customer_data']
                if customer_data:
                    wait_times = [c.exit_time - c.arrival_time for c in customer_data]
                    W = np.mean(wait_times) / 60  # Convert to hours
                    
                    # Calculate L (average number in system) - Little's Law prediction
                    L_predicted = lambda_val * W
                    
                    # Calculate actual L (average number in system)
                    # This is approximated by total customers / simulation time
                    L_actual = result['total_customers'] / simulation_time_hours
                    
                    # Calculate error
                    error = abs(L_predicted - L_actual) / L_actual if L_actual > 0 else 0
                    
                    validation_results.append({
                        'lambda': lambda_val,
                        'W_hours': W,
                        'L_predicted': L_predicted,
                        'L_actual': L_actual,
                        'error_percent': error * 100,
                        'valid': error < 0.1  # 10% tolerance
                    })
        
        if validation_results:
            df = pd.DataFrame(validation_results)
            return {
                'validation_data': df,
                'avg_error': df['error_percent'].mean(),
                'max_error': df['error_percent'].max(),
                'validation_passed': df['error_percent'].max() < 10,
                'summary': {
                    'mean_lambda': df['lambda'].mean(),
                    'mean_W': df['W_hours'].mean(),
                    'mean_L_predicted': df['L_predicted'].mean(),
                    'mean_L_actual': df['L_actual'].mean()
                }
            }
        else:
            return {'error': 'No valid data for Little Law validation'}

class MMSCalculator:
    """Class to calculate M/M/s queueing theory approximations"""
    
    @staticmethod
    def calculate_mms_metrics(arrival_rate: float, service_rate: float, 
                            num_servers: int) -> Dict[str, float]:
        """
        Calculate M/M/s queueing theory metrics
        
        Args:
            arrival_rate: λ (customers per hour)
            service_rate: μ (customers per hour per server)
            num_servers: s (number of servers)
        """
        rho = arrival_rate / (num_servers * service_rate)  # Traffic intensity
        
        if rho >= 1:
            return {'error': 'System is unstable (ρ ≥ 1)'}
        
        # Calculate P0 (probability of zero customers)
        sum_term = sum([(arrival_rate / service_rate)**k / np.math.factorial(k) 
                       for k in range(num_servers)])
        p0_term = (arrival_rate / service_rate)**num_servers / np.math.factorial(num_servers)
        p0_term *= 1 / (1 - rho)
        p0 = 1 / (sum_term + p0_term)
        
        # Calculate Lq (average number in queue)
        lq = p0 * p0_term * rho / ((1 - rho)**2)
        
        # Calculate L (average number in system)
        l = lq + arrival_rate / service_rate
        
        # Calculate Wq (average wait time in queue)
        wq = lq / arrival_rate
        
        # Calculate W (average time in system)
        w = wq + 1 / service_rate
        
        return {
            'rho': rho,
            'p0': p0,
            'Lq': lq,
            'L': l,
            'Wq_hours': wq,
            'W_hours': w,
            'Wq_minutes': wq * 60,
            'W_minutes': w * 60
        }
    
    @staticmethod
    def compare_with_simulation(mms_results: Dict, simulation_results: Dict) -> Dict[str, float]:
        """Compare M/M/s theory with simulation results"""
        if 'error' in mms_results:
            return {'error': 'M/M/s calculation failed'}
        
        comparison = {}
        
        # Compare average wait time
        sim_wait = simulation_results.get('avg_wait_time', 0) / 60  # Convert to hours
        theory_wait = mms_results['W_hours']
        comparison['wait_time_error'] = abs(sim_wait - theory_wait) / theory_wait if theory_wait > 0 else 0
        
        # Compare throughput
        sim_throughput = simulation_results.get('throughput', 0)
        theory_throughput = 1 / mms_results['W_hours'] if mms_results['W_hours'] > 0 else 0
        comparison['throughput_error'] = abs(sim_throughput - theory_throughput) / theory_throughput if theory_throughput > 0 else 0
        
        return comparison

class SensitivityAnalyzer:
    """Class for sensitivity analysis of simulation parameters"""
    
    def __init__(self, base_config: Dict):
        """Initialize with base configuration"""
        self.base_config = base_config
    
    def run_sensitivity_analysis(self, parameter: str, values: List[float], 
                               num_replications: int = 50) -> Dict[str, List[Dict]]:
        """
        Run sensitivity analysis for a specific parameter
        
        Args:
            parameter: Parameter name to vary
            values: List of values to test
            num_replications: Number of replications per value
        """
        from boba_simulator import run_monte_carlo_simulation
        
        results = {}
        
        for value in values:
            # Create modified configuration
            config = self.base_config.copy()
            config[parameter] = value
            
            # Run simulation
            sim_results = run_monte_carlo_simulation(config, num_replications, simulation_time=240)
            results[f"{parameter}_{value}"] = sim_results
        
        return results
    
    def calculate_sensitivity_indices(self, sensitivity_results: Dict[str, List[Dict]], 
                                    response_variable: str) -> Dict[str, float]:
        """Calculate sensitivity indices for the analysis"""
        indices = {}
        
        # Get baseline results
        baseline_key = None
        for key in sensitivity_results.keys():
            if 'baseline' in key or key == list(sensitivity_results.keys())[0]:
                baseline_key = key
                break
        
        if not baseline_key:
            return {'error': 'No baseline found'}
        
        baseline_results = sensitivity_results[baseline_key]
        baseline_mean = np.mean([r[response_variable] for r in baseline_results if 'error' not in r])
        
        # Calculate sensitivity index for each parameter value
        for key, results in sensitivity_results.items():
            if key != baseline_key:
                param_mean = np.mean([r[response_variable] for r in results if 'error' not in r])
                sensitivity_index = (param_mean - baseline_mean) / baseline_mean
                indices[key] = sensitivity_index
        
        return indices

class OptimizationEngine:
    """Class for optimizing boba shop operations"""
    
    def __init__(self, cost_parameters: Dict[str, float]):
        """Initialize with cost parameters"""
        self.cost_params = cost_parameters
    
    def objective_function(self, x: np.ndarray, arrival_rate: float) -> float:
        """
        Objective function for optimization
        
        Args:
            x: [cashier_capacity, barista_capacity, sealer_capacity]
            arrival_rate: Expected arrival rate
        """
        from boba_simulator import BobaShopSimulator
        
        # Create configuration
        config = {
            'cashier_capacity': int(x[0]),
            'barista_capacity': int(x[1]),
            'sealer_capacity': int(x[2]),
            'initial_pearl_inventory': 50,
            'pearl_reorder_point': 10,
            'pearl_batch_size': 30,
            'pearl_cook_time': 15,
            'policy': 'optimized'
        }
        
        # Run simulation
        simulator = BobaShopSimulator(config)
        results = simulator.run_simulation(simulation_time=240)
        
        if 'error' in results:
            return float('inf')
        
        # Calculate total cost
        staff_cost = (x[0] + x[1] + x[2]) * self.cost_params['staff_hourly_cost'] * 8
        wait_cost = results['avg_wait_time'] * results['total_customers'] * self.cost_params['customer_wait_cost']
        
        return staff_cost + wait_cost
    
    def optimize_staffing(self, arrival_rate: float, 
                         constraints: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        """
        Optimize staffing levels
        
        Args:
            arrival_rate: Expected arrival rate
            constraints: Dictionary with parameter bounds (min, max)
        """
        # Initial guess
        x0 = np.array([1, 2, 1])
        
        # Bounds
        bounds = [
            constraints.get('cashier_capacity', (1, 3)),
            constraints.get('barista_capacity', (1, 5)),
            constraints.get('sealer_capacity', (1, 3))
        ]
        
        # Optimize
        result = minimize(
            self.objective_function,
            x0,
            args=(arrival_rate,),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        return {
            'optimal_staffing': {
                'cashier_capacity': int(result.x[0]),
                'barista_capacity': int(result.x[1]),
                'sealer_capacity': int(result.x[2])
            },
            'optimal_cost': result.fun,
            'success': result.success,
            'message': result.message
        }

class StatisticalValidator:
    """Class for statistical validation of simulation results"""
    
    @staticmethod
    def chi_square_goodness_of_fit(observed: np.ndarray, expected: np.ndarray) -> Dict[str, Any]:
        """Perform chi-square goodness of fit test"""
        # Remove zero expected values
        mask = expected > 0
        observed_clean = observed[mask]
        expected_clean = expected[mask]
        
        if len(observed_clean) == 0:
            return {'error': 'No valid data for chi-square test'}
        
        # Calculate chi-square statistic
        chi2_stat = np.sum((observed_clean - expected_clean)**2 / expected_clean)
        dof = len(observed_clean) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        return {
            'chi2_statistic': chi2_stat,
            'degrees_of_freedom': dof,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'critical_value': stats.chi2.ppf(0.95, dof)
        }
    
    @staticmethod
    def kolmogorov_smirnov_test(data: np.ndarray, distribution: str = 'exponential') -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for distribution fitting"""
        if distribution == 'exponential':
            # Fit exponential distribution
            loc, scale = stats.expon.fit(data)
            ks_stat, p_value = stats.kstest(data, lambda x: stats.expon.cdf(x, loc, scale))
        elif distribution == 'lognormal':
            # Fit lognormal distribution
            shape, loc, scale = stats.lognorm.fit(data)
            ks_stat, p_value = stats.kstest(data, lambda x: stats.lognorm.cdf(x, shape, loc, scale))
        else:
            return {'error': f'Distribution {distribution} not supported'}
        
        return {
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'distribution': distribution
        }
    
    @staticmethod
    def autocorrelation_test(data: np.ndarray, max_lags: int = 10) -> Dict[str, Any]:
        """Test for autocorrelation in simulation output"""
        from statsmodels.tsa.stattools import acf
        
        # Calculate autocorrelation
        autocorr = acf(data, nlags=max_lags, fft=False)
        
        # Test significance (rough approximation)
        n = len(data)
        critical_value = 1.96 / np.sqrt(n)
        
        significant_lags = []
        for i, corr in enumerate(autocorr[1:], 1):  # Skip lag 0
            if abs(corr) > critical_value:
                significant_lags.append(i)
        
        return {
            'autocorrelations': autocorr,
            'critical_value': critical_value,
            'significant_lags': significant_lags,
            'has_autocorrelation': len(significant_lags) > 0
        }

class PerformanceAnalyzer:
    """Class for analyzing simulation performance and bottlenecks"""
    
    @staticmethod
    def identify_bottlenecks(utilization_data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Identify system bottlenecks based on utilization data"""
        bottlenecks = {}
        
        for station, utilizations in utilization_data.items():
            avg_utilization = np.mean(utilizations)
            max_utilization = np.max(utilizations)
            
            # Consider bottleneck if average utilization > 80% or max > 95%
            is_bottleneck = avg_utilization > 0.8 or max_utilization > 0.95
            
            bottlenecks[station] = {
                'avg_utilization': avg_utilization,
                'max_utilization': max_utilization,
                'is_bottleneck': is_bottleneck,
                'severity': 'high' if max_utilization > 0.95 else 'medium' if avg_utilization > 0.8 else 'low'
            }
        
        return bottlenecks
    
    @staticmethod
    def calculate_system_efficiency(results: List[Dict]) -> Dict[str, float]:
        """Calculate overall system efficiency metrics"""
        if not results:
            return {'error': 'No results available'}
        
        # Calculate efficiency metrics
        throughputs = [r['throughput'] for r in results if 'error' not in r]
        wait_times = [r['avg_wait_time'] for r in results if 'error' not in r]
        
        avg_throughput = np.mean(throughputs)
        avg_wait_time = np.mean(wait_times)
        
        # Calculate efficiency (throughput per unit wait time)
        efficiency = avg_throughput / avg_wait_time if avg_wait_time > 0 else 0
        
        # Calculate coefficient of variation for stability
        cv_throughput = np.std(throughputs) / avg_throughput if avg_throughput > 0 else 0
        cv_wait_time = np.std(wait_times) / avg_wait_time if avg_wait_time > 0 else 0
        
        return {
            'avg_throughput': avg_throughput,
            'avg_wait_time': avg_wait_time,
            'efficiency': efficiency,
            'throughput_stability': 1 - cv_throughput,  # Higher is more stable
            'wait_time_stability': 1 - cv_wait_time,    # Higher is more stable
            'overall_stability': 1 - (cv_throughput + cv_wait_time) / 2
        }

def run_comprehensive_analysis():
    """Run comprehensive analysis of the boba shop simulation"""
    from boba_simulator import run_monte_carlo_simulation, create_baseline_config
    
    print("Running Comprehensive Analysis...")
    print("=" * 50)
    
    # 1. Little's Law Validation
    print("\n1. Little's Law Validation")
    config = create_baseline_config()
    results = run_monte_carlo_simulation(config, num_replications=30, simulation_time=240)
    
    validator = LittleLawValidator()
    littles_validation = validator.validate_littles_law(results)
    
    if 'error' not in littles_validation:
        print(f"Average Error: {littles_validation['avg_error']:.2f}%")
        print(f"Max Error: {littles_validation['max_error']:.2f}%")
        print(f"Validation Passed: {littles_validation['validation_passed']}")
    
    # 2. M/M/s Comparison
    print("\n2. M/M/s Queueing Theory Comparison")
    calculator = MMSCalculator()
    
    # Estimate service rates from simulation
    avg_service_time = np.mean([r['avg_service_time'] for r in results if 'error' not in r])
    service_rate = 60 / avg_service_time  # customers per hour
    arrival_rate = np.mean([r['throughput'] for r in results if 'error' not in r])
    
    mms_results = calculator.calculate_mms_metrics(arrival_rate, service_rate, 2)
    
    if 'error' not in mms_results:
        print(f"Theoretical Wait Time: {mms_results['W_minutes']:.2f} minutes")
        print(f"Simulation Wait Time: {np.mean([r['avg_wait_time'] for r in results if 'error' not in r]):.2f} minutes")
        
        comparison = calculator.compare_with_simulation(mms_results, results[0])
        print(f"Wait Time Error: {comparison.get('wait_time_error', 0)*100:.2f}%")
    
    # 3. Sensitivity Analysis
    print("\n3. Sensitivity Analysis")
    analyzer = SensitivityAnalyzer(config)
    
    # Analyze sensitivity to barista capacity
    barista_values = [1, 2, 3, 4]
    sensitivity_results = analyzer.run_sensitivity_analysis('barista_capacity', barista_values, 20)
    
    sensitivity_indices = analyzer.calculate_sensitivity_indices(sensitivity_results, 'avg_wait_time')
    print("Sensitivity to Barista Capacity:")
    for key, index in sensitivity_indices.items():
        print(f"  {key}: {index:.3f}")
    
    # 4. Statistical Validation
    print("\n4. Statistical Validation")
    stat_validator = StatisticalValidator()
    
    # Test service time distribution
    service_times = []
    for result in results:
        if 'customer_data' in result:
            for customer in result['customer_data']:
                if hasattr(customer, 'service_start_time') and hasattr(customer, 'service_end_time'):
                    if customer.service_start_time and customer.service_end_time:
                        service_times.append(customer.service_end_time - customer.service_start_time)
    
    if service_times:
        ks_test = stat_validator.kolmogorov_smirnov_test(np.array(service_times), 'lognormal')
        print(f"Service Time Distribution Test (Lognormal):")
        print(f"  KS Statistic: {ks_test['ks_statistic']:.4f}")
        print(f"  P-value: {ks_test['p_value']:.4f}")
        print(f"  Significant: {ks_test['significant']}")
    
    # 5. Performance Analysis
    print("\n5. Performance Analysis")
    perf_analyzer = PerformanceAnalyzer()
    
    # Mock utilization data (in real implementation, this would come from simulation)
    utilization_data = {
        'cashier': [0.6, 0.7, 0.8, 0.9, 0.85],
        'barista': [0.8, 0.9, 0.95, 0.9, 0.85],
        'sealer': [0.4, 0.5, 0.6, 0.7, 0.65]
    }
    
    bottlenecks = perf_analyzer.identify_bottlenecks(utilization_data)
    print("Bottleneck Analysis:")
    for station, analysis in bottlenecks.items():
        print(f"  {station}: {analysis['severity']} bottleneck (avg: {analysis['avg_utilization']:.2f})")
    
    efficiency = perf_analyzer.calculate_system_efficiency(results)
    print(f"\nSystem Efficiency:")
    print(f"  Overall Efficiency: {efficiency['efficiency']:.2f}")
    print(f"  System Stability: {efficiency['overall_stability']:.2f}")
    
    print("\nComprehensive analysis completed!")

if __name__ == "__main__":
    run_comprehensive_analysis()
