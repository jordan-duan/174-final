"""
Experimental Design and Analysis Module for Boba Shop Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This module implements factorial experimental design, statistical analysis,
and advanced visualization techniques for the boba shop simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import product
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FactorialExperiment:
    """Class to design and analyze factorial experiments"""
    
    def __init__(self, factors: Dict[str, List], response_variables: List[str]):
        """
        Initialize factorial experiment
        
        Args:
            factors: Dictionary of factor names and their levels
            response_variables: List of response variable names
        """
        self.factors = factors
        self.response_variables = response_variables
        self.experiment_matrix = self._create_experiment_matrix()
        self.results = None
        
    def _create_experiment_matrix(self) -> pd.DataFrame:
        """Create full factorial experiment matrix"""
        factor_names = list(self.factors.keys())
        factor_levels = list(self.factors.values())
        
        # Generate all combinations
        combinations = list(product(*factor_levels))
        
        # Create DataFrame
        experiment_df = pd.DataFrame(combinations, columns=factor_names)
        
        # Add experiment ID
        experiment_df['experiment_id'] = range(len(experiment_df))
        
        return experiment_df
    
    def add_results(self, results: pd.DataFrame):
        """Add experimental results to the design matrix"""
        self.results = pd.merge(self.experiment_matrix, results, on='experiment_id', how='left')
    
    def analyze_main_effects(self) -> Dict[str, pd.DataFrame]:
        """Analyze main effects for each factor and response variable"""
        if self.results is None:
            raise ValueError("No results added to experiment")
        
        main_effects = {}
        
        for response_var in self.response_variables:
            if response_var in self.results.columns:
                effects_df = pd.DataFrame()
                
                for factor in self.factors.keys():
                    # Calculate main effect for this factor
                    factor_means = self.results.groupby(factor)[response_var].mean()
                    effect = factor_means.max() - factor_means.min()
                    
                    effects_df = pd.concat([effects_df, pd.DataFrame({
                        'factor': [factor],
                        'main_effect': [effect],
                        'min_level': [factor_means.idxmin()],
                        'max_level': [factor_means.idxmax()]
                    })], ignore_index=True)
                
                main_effects[response_var] = effects_df
        
        return main_effects
    
    def analyze_interactions(self) -> Dict[str, pd.DataFrame]:
        """Analyze two-factor interactions"""
        if self.results is None:
            raise ValueError("No results added to experiment")
        
        interactions = {}
        factor_names = list(self.factors.keys())
        
        for response_var in self.response_variables:
            if response_var in self.results.columns:
                interaction_df = pd.DataFrame()
                
                # Calculate all two-factor interactions
                for i, factor1 in enumerate(factor_names):
                    for j, factor2 in enumerate(factor_names[i+1:], i+1):
                        # Calculate interaction effect
                        interaction_means = self.results.groupby([factor1, factor2])[response_var].mean().unstack()
                        
                        # Simple interaction measure (difference of differences)
                        interaction_effect = (
                            (interaction_means.iloc[0, 0] - interaction_means.iloc[0, 1]) -
                            (interaction_means.iloc[1, 0] - interaction_means.iloc[1, 1])
                        )
                        
                        interaction_df = pd.concat([interaction_df, pd.DataFrame({
                            'factor1': [factor1],
                            'factor2': [factor2],
                            'interaction_effect': [abs(interaction_effect)]
                        })], ignore_index=True)
                
                interactions[response_var] = interaction_df.sort_values('interaction_effect', ascending=False)
        
        return interactions

class StatisticalAnalyzer:
    """Class for statistical analysis of simulation results"""
    
    @staticmethod
    def chi_square_test(observed: np.ndarray, expected: np.ndarray) -> Dict[str, float]:
        """Perform chi-square goodness of fit test"""
        chi2_stat = np.sum((observed - expected)**2 / expected)
        dof = len(observed) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
        
        return {
            'chi2_statistic': chi2_stat,
            'degrees_of_freedom': dof,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def anova_test(data_groups: List[np.ndarray]) -> Dict[str, float]:
        """Perform one-way ANOVA test"""
        f_stat, p_value = stats.f_oneway(*data_groups)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def t_test_independent(group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """Perform independent samples t-test"""
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (np.mean(group1) - np.mean(group2)) / np.sqrt(
                (np.var(group1) + np.var(group2)) / 2
            )
        }
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # t-distribution critical value
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_critical * std / np.sqrt(n)
        
        return mean - margin_error, mean + margin_error

class AdvancedVisualizer:
    """Class for advanced visualization of simulation results"""
    
    @staticmethod
    def plot_effects_plot(experiment: FactorialExperiment, response_var: str, 
                         save_path: str = None):
        """Create effects plot for factorial experiment"""
        if experiment.results is None:
            raise ValueError("No results available for plotting")
        
        factors = list(experiment.factors.keys())
        n_factors = len(factors)
        
        fig, axes = plt.subplots(1, n_factors, figsize=(5*n_factors, 6))
        if n_factors == 1:
            axes = [axes]
        
        for i, factor in enumerate(factors):
            # Calculate means and confidence intervals
            factor_data = experiment.results.groupby(factor)[response_var].agg(['mean', 'std', 'count'])
            factor_data['ci'] = 1.96 * factor_data['std'] / np.sqrt(factor_data['count'])
            
            # Plot main effects
            x_pos = range(len(factor_data))
            axes[i].errorbar(x_pos, factor_data['mean'], yerr=factor_data['ci'], 
                           marker='o', capsize=5, capthick=2)
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(factor_data.index)
            axes[i].set_xlabel(factor)
            axes[i].set_ylabel(response_var)
            axes[i].set_title(f'Main Effect of {factor}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_interaction_plot(experiment: FactorialExperiment, response_var: str,
                            factor1: str, factor2: str, save_path: str = None):
        """Create interaction plot for two factors"""
        if experiment.results is None:
            raise ValueError("No results available for plotting")
        
        # Create pivot table for interaction plot
        pivot_data = experiment.results.groupby([factor1, factor2])[response_var].mean().unstack()
        
        plt.figure(figsize=(10, 6))
        
        for level in pivot_data.columns:
            plt.plot(pivot_data.index, pivot_data[level], marker='o', label=f'{factor2}={level}')
        
        plt.xlabel(factor1)
        plt.ylabel(response_var)
        plt.title(f'Interaction Plot: {factor1} × {factor2}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_heatmap(experiment: FactorialExperiment, response_var: str,
                    factor1: str, factor2: str, save_path: str = None):
        """Create heatmap for two factors"""
        if experiment.results is None:
            raise ValueError("No results available for plotting")
        
        # Create pivot table
        pivot_data = experiment.results.groupby([factor1, factor2])[response_var].mean().unstack()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='viridis', 
                   cbar_kws={'label': response_var})
        plt.title(f'Heatmap: {factor1} × {factor2} vs {response_var}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_queue_trace(queue_data: Dict[str, List[Tuple[float, int]]], 
                        save_path: str = None):
        """Plot queue length over time"""
        plt.figure(figsize=(15, 8))
        
        for station, data in queue_data.items():
            if data:  # Check if data exists
                times, lengths = zip(*data)
                plt.plot(times, lengths, label=station, alpha=0.7)
        
        plt.xlabel('Time (minutes)')
        plt.ylabel('Queue Length')
        plt.title('Queue Length Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_gantt_chart(customer_data: List, save_path: str = None):
        """Create Gantt chart for customer service timeline"""
        if not customer_data:
            print("No customer data available for Gantt chart")
            return
        
        # Prepare data for Gantt chart
        gantt_data = []
        for i, customer in enumerate(customer_data[:20]):  # Limit to first 20 customers
            if hasattr(customer, 'arrival_time') and hasattr(customer, 'exit_time'):
                gantt_data.append({
                    'customer_id': customer.customer_id,
                    'start': customer.arrival_time,
                    'duration': customer.exit_time - customer.arrival_time,
                    'end': customer.exit_time
                })
        
        if not gantt_data:
            print("No valid customer data for Gantt chart")
            return
        
        df_gantt = pd.DataFrame(gantt_data)
        
        # Create Gantt chart
        fig, ax = plt.subplots(figsize=(15, 10))
        
        for i, row in df_gantt.iterrows():
            ax.barh(row['customer_id'], row['duration'], left=row['start'], 
                   alpha=0.7, height=0.8)
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Customer ID')
        ax.set_title('Customer Service Timeline (Gantt Chart)')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CostBenefitAnalyzer:
    """Class for cost-benefit analysis of different policies"""
    
    def __init__(self, cost_parameters: Dict[str, float]):
        """
        Initialize cost-benefit analyzer
        
        Args:
            cost_parameters: Dictionary with cost parameters
                - staff_hourly_cost: Cost per hour for staff
                - automation_cost: One-time cost for automation
                - automation_maintenance: Annual maintenance cost
                - customer_wait_cost: Cost per minute of customer wait
        """
        self.cost_params = cost_parameters
    
    def calculate_total_cost(self, policy: str, results: Dict, 
                           simulation_hours: float = 8) -> Dict[str, float]:
        """Calculate total cost for a given policy"""
        costs = {}
        
        # Staff costs
        if policy == 'baseline':
            staff_cost = (1 + 2 + 1) * self.cost_params['staff_hourly_cost'] * simulation_hours
        elif policy == 'automation':
            staff_cost = (1 + 2 + 1) * self.cost_params['staff_hourly_cost'] * simulation_hours
            automation_cost = self.cost_params['automation_cost'] / 365  # Daily cost
        elif policy == 'staffing':
            staff_cost = (1 + 3 + 1) * self.cost_params['staff_hourly_cost'] * simulation_hours
        else:
            staff_cost = 0
        
        # Customer wait costs
        avg_wait_time = results.get('avg_wait_time', 0)
        total_customers = results.get('total_customers', 0)
        wait_cost = avg_wait_time * total_customers * self.cost_params['customer_wait_cost']
        
        costs['staff_cost'] = staff_cost
        costs['wait_cost'] = wait_cost
        costs['total_cost'] = staff_cost + wait_cost
        
        if policy == 'automation':
            costs['automation_cost'] = automation_cost
            costs['total_cost'] += automation_cost
        
        return costs
    
    def calculate_cost_per_minute_saved(self, baseline_results: Dict, 
                                      policy_results: Dict, policy: str) -> float:
        """Calculate cost per minute of wait time saved"""
        baseline_wait = baseline_results.get('avg_wait_time', 0)
        policy_wait = policy_results.get('avg_wait_time', 0)
        
        wait_time_saved = baseline_wait - policy_wait
        
        if wait_time_saved <= 0:
            return float('inf')  # No improvement
        
        baseline_costs = self.calculate_total_cost('baseline', baseline_results)
        policy_costs = self.calculate_total_cost(policy, policy_results)
        
        cost_difference = policy_costs['total_cost'] - baseline_costs['total_cost']
        
        return cost_difference / wait_time_saved

def run_factorial_experiment():
    """Run a factorial experiment with different factors"""
    from boba_simulator import BobaShopSimulator, run_monte_carlo_simulation
    
    # Define factors for the experiment
    factors = {
        'policy': ['baseline', 'automation', 'staffing'],
        'arrival_multiplier': [0.8, 1.0, 1.2],  # Vary arrival rates
        'barista_skill': ['novice', 'trained']   # Different service time distributions
    }
    
    # Create experiment design
    experiment = FactorialExperiment(factors, ['avg_wait_time', 'throughput', 'p95_wait_time'])
    
    print(f"Running {len(experiment.experiment_matrix)} experimental conditions...")
    
    # Run experiments
    results_list = []
    
    for _, row in experiment.experiment_matrix.iterrows():
        # Create configuration based on experiment design
        config = {
            'cashier_capacity': 1,
            'barista_capacity': 3 if row['policy'] == 'staffing' else 2,
            'sealer_capacity': 2 if row['policy'] == 'automation' else 1,
            'initial_pearl_inventory': 50,
            'pearl_reorder_point': 10,
            'pearl_batch_size': 30,
            'pearl_cook_time': 15,
            'policy': row['policy'],
            'arrival_multiplier': row['arrival_multiplier'],
            'barista_skill': row['barista_skill']
        }
        
        # Run simulation (fewer replications for factorial experiment)
        results = run_monte_carlo_simulation(config, num_replications=10, simulation_time=240)
        
        # Calculate average results
        avg_results = {
            'experiment_id': row['experiment_id'],
            'avg_wait_time': np.mean([r['avg_wait_time'] for r in results if 'error' not in r]),
            'throughput': np.mean([r['throughput'] for r in results if 'error' not in r]),
            'p95_wait_time': np.mean([r['p95_wait_time'] for r in results if 'error' not in r])
        }
        
        results_list.append(avg_results)
    
    # Add results to experiment
    results_df = pd.DataFrame(results_list)
    experiment.add_results(results_df)
    
    return experiment

if __name__ == "__main__":
    # Run factorial experiment
    experiment = run_factorial_experiment()
    
    # Analyze main effects
    main_effects = experiment.analyze_main_effects()
    print("\nMain Effects Analysis:")
    for response_var, effects in main_effects.items():
        print(f"\n{response_var}:")
        print(effects)
    
    # Create visualizations
    visualizer = AdvancedVisualizer()
    
    for response_var in ['avg_wait_time', 'throughput']:
        visualizer.plot_effects_plot(experiment, response_var, 
                                   f'effects_plot_{response_var}.png')
    
    # Interaction analysis
    interactions = experiment.analyze_interactions()
    print("\nInteraction Effects:")
    for response_var, interaction_df in interactions.items():
        print(f"\n{response_var}:")
        print(interaction_df.head())
    
    print("\nFactorial experiment completed!")
