"""
Enhanced Visualizations for Boba Shop Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This module provides comprehensive visualizations including interactive plots,
real-time simulation graphics, and detailed analysis charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedVisualizer:
    """Enhanced visualization class with interactive features"""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize with results directory"""
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_comprehensive_dashboard(self, results_dict: Dict[str, List[Dict]]):
        """Create a comprehensive dashboard with multiple visualizations"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Performance Comparison (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_comparison(ax1, results_dict)
        
        # 2. Wait Time Distribution (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_wait_time_distribution(ax2, results_dict)
        
        # 3. Throughput Over Time (Second Row Left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_throughput_over_time(ax3, results_dict)
        
        # 4. Utilization Heatmap (Second Row Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_utilization_heatmap(ax4, results_dict)
        
        # 5. Cost-Benefit Analysis (Third Row Left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_cost_benefit_analysis(ax5, results_dict)
        
        # 6. Bottleneck Analysis (Third Row Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_bottleneck_analysis(ax6, results_dict)
        
        # 7. Statistical Summary (Bottom Row)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_statistical_summary(ax7, results_dict)
        
        plt.suptitle('Boba Shop Operations Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Save the dashboard
        dashboard_path = os.path.join(self.results_dir, f'dashboard_{self.timestamp}.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to: {dashboard_path}")
        
        plt.show()
    
    def _plot_performance_comparison(self, ax, results_dict):
        """Plot performance comparison across policies"""
        policies = list(results_dict.keys())
        metrics = ['avg_wait_time', 'p95_wait_time', 'throughput']
        metric_labels = ['Avg Wait Time (min)', '95th %ile Wait (min)', 'Throughput (cust/hr)']
        
        x = np.arange(len(policies))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = []
            errors = []
            for policy in policies:
                policy_results = [r[metric] for r in results_dict[policy] if 'error' not in r]
                values.append(np.mean(policy_results))
                errors.append(np.std(policy_results))
            
            ax.bar(x + i*width, values, width, label=label, yerr=errors, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Policy')
        ax.set_ylabel('Performance Metrics')
        ax.set_title('Performance Comparison Across Policies')
        ax.set_xticks(x + width)
        ax.set_xticklabels(policies)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_wait_time_distribution(self, ax, results_dict):
        """Plot wait time distributions"""
        for policy, results in results_dict.items():
            wait_times = [r['avg_wait_time'] for r in results if 'error' not in r]
            ax.hist(wait_times, alpha=0.6, label=policy, bins=15, density=True)
        
        ax.set_xlabel('Wait Time (minutes)')
        ax.set_ylabel('Density')
        ax.set_title('Wait Time Distribution by Policy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_throughput_over_time(self, ax, results_dict):
        """Plot throughput over time (simulated)"""
        time_points = np.linspace(0, 8, 50)  # 8 hours
        
        for policy, results in results_dict.items():
            # Simulate throughput variation over time
            base_throughput = np.mean([r['throughput'] for r in results if 'error' not in r])
            
            # Add realistic daily variation
            daily_pattern = 1 + 0.3 * np.sin(2 * np.pi * time_points / 8 - np.pi/2)
            throughput_over_time = base_throughput * daily_pattern
            
            ax.plot(time_points, throughput_over_time, label=policy, linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Throughput (customers/hour)')
        ax.set_title('Throughput Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_utilization_heatmap(self, ax, results_dict):
        """Plot utilization heatmap"""
        stations = ['Cashier', 'Barista', 'Sealer']
        policies = list(results_dict.keys())
        
        # Create utilization matrix (mock data for demonstration)
        utilization_matrix = np.random.uniform(0.3, 0.95, (len(stations), len(policies)))
        
        im = ax.imshow(utilization_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(stations)):
            for j in range(len(policies)):
                text = ax.text(j, i, f'{utilization_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xticks(range(len(policies)))
        ax.set_yticks(range(len(stations)))
        ax.set_xticklabels(policies)
        ax.set_yticklabels(stations)
        ax.set_title('Station Utilization by Policy')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Utilization Rate')
    
    def _plot_cost_benefit_analysis(self, ax, results_dict):
        """Plot cost-benefit analysis"""
        # Mock cost data (in real implementation, this would come from cost analysis)
        policies = list(results_dict.keys())
        costs = [100, 150, 120]  # Baseline, Automation, Staffing
        benefits = [0, -2.5, -1.8]  # Wait time reduction (negative = improvement)
        
        colors = ['red', 'blue', 'green']
        
        for i, (policy, cost, benefit) in enumerate(zip(policies, costs, benefits)):
            ax.scatter(cost, benefit, s=200, c=colors[i], alpha=0.7, label=policy)
            ax.annotate(policy, (cost, benefit), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Cost (relative)')
        ax.set_ylabel('Wait Time Reduction (minutes)')
        ax.set_title('Cost-Benefit Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_bottleneck_analysis(self, ax, results_dict):
        """Plot bottleneck analysis"""
        stations = ['Cashier', 'Barista', 'Sealer']
        policies = list(results_dict.keys())
        
        # Mock bottleneck severity data
        bottleneck_data = {
            'baseline': [0.2, 0.8, 0.4],
            'automation': [0.2, 0.7, 0.6],
            'staffing': [0.2, 0.5, 0.4]
        }
        
        x = np.arange(len(stations))
        width = 0.25
        
        for i, policy in enumerate(policies):
            values = bottleneck_data.get(policy, [0.5, 0.5, 0.5])
            ax.bar(x + i*width, values, width, label=policy, alpha=0.8)
        
        ax.set_xlabel('Station')
        ax.set_ylabel('Bottleneck Severity')
        ax.set_title('Bottleneck Analysis by Policy')
        ax.set_xticks(x + width)
        ax.set_xticklabels(stations)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_summary(self, ax, results_dict):
        """Plot statistical summary table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_data = []
        for policy, results in results_dict.items():
            policy_results = [r for r in results if 'error' not in r]
            if policy_results:
                summary_data.append([
                    policy,
                    f"{np.mean([r['avg_wait_time'] for r in policy_results]):.2f}",
                    f"{np.mean([r['p95_wait_time'] for r in policy_results]):.2f}",
                    f"{np.mean([r['throughput'] for r in policy_results]):.2f}",
                    f"{np.mean([r['total_customers'] for r in policy_results]):.0f}"
                ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Policy', 'Avg Wait (min)', '95th %ile (min)', 'Throughput (cust/hr)', 'Total Customers'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
    
    def create_interactive_parameter_explorer(self):
        """Create interactive parameter exploration tool"""
        from boba_simulator import BobaShopSimulator, create_baseline_config
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # Initial parameters
        initial_cashier = 1
        initial_barista = 2
        initial_sealer = 1
        initial_arrival_mult = 1.0
        
        # Create sliders
        ax_cashier = plt.axes([0.1, 0.1, 0.15, 0.03])
        ax_barista = plt.axes([0.3, 0.1, 0.15, 0.03])
        ax_sealer = plt.axes([0.5, 0.1, 0.15, 0.03])
        ax_arrival = plt.axes([0.7, 0.1, 0.15, 0.03])
        
        slider_cashier = Slider(ax_cashier, 'Cashiers', 1, 3, valinit=initial_cashier, valstep=1)
        slider_barista = Slider(ax_barista, 'Baristas', 1, 5, valinit=initial_barista, valstep=1)
        slider_sealer = Slider(ax_sealer, 'Sealers', 1, 3, valinit=initial_sealer, valstep=1)
        slider_arrival = Slider(ax_arrival, 'Arrival Rate', 0.5, 2.0, valinit=initial_arrival_mult, valstep=0.1)
        
        # Initial plot
        def update_plot(val):
            ax1.clear()
            ax2.clear()
            
            # Get current values
            cashier_cap = int(slider_cashier.val)
            barista_cap = int(slider_barista.val)
            sealer_cap = int(slider_sealer.val)
            arrival_mult = slider_arrival.val
            
            # Create configuration
            config = create_baseline_config()
            config['cashier_capacity'] = cashier_cap
            config['barista_capacity'] = barista_cap
            config['sealer_capacity'] = sealer_cap
            
            # Run simulation
            simulator = BobaShopSimulator(config)
            results = simulator.run_simulation(simulation_time=120)
            
            if 'error' not in results:
                # Plot 1: Performance metrics
                metrics = ['avg_wait_time', 'p95_wait_time', 'throughput']
                values = [results[metric] for metric in metrics]
                labels = ['Avg Wait (min)', '95th %ile (min)', 'Throughput (cust/hr)']
                
                bars = ax1.bar(labels, values, color=['red', 'orange', 'green'], alpha=0.7)
                ax1.set_title('Current Performance')
                ax1.set_ylabel('Value')
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{value:.2f}', ha='center', va='bottom')
                
                # Plot 2: Configuration visualization
                stations = ['Cashier', 'Barista', 'Sealer']
                capacities = [cashier_cap, barista_cap, sealer_cap]
                
                bars2 = ax2.bar(stations, capacities, color=['blue', 'green', 'purple'], alpha=0.7)
                ax2.set_title('Current Configuration')
                ax2.set_ylabel('Capacity')
                ax2.set_ylim(0, 6)
                
                # Add value labels
                for bar, value in zip(bars2, capacities):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{value}', ha='center', va='bottom', fontweight='bold')
            
            fig.canvas.draw()
        
        # Connect sliders to update function
        slider_cashier.on_changed(update_plot)
        slider_barista.on_changed(update_plot)
        slider_sealer.on_changed(update_plot)
        slider_arrival.on_changed(update_plot)
        
        # Initial plot
        update_plot(None)
        
        plt.suptitle('Interactive Parameter Explorer', fontsize=16, fontweight='bold')
        plt.show()
    
    def create_real_time_simulation_animation(self, config, duration=60):
        """Create real-time simulation animation"""
        from boba_simulator import BobaShopSimulator
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Initialize simulation
        simulator = BobaShopSimulator(config)
        
        # Data storage for animation
        time_data = []
        queue_data = {'cashier': [], 'barista': [], 'sealer': []}
        customer_data = []
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Run simulation for one time step
            current_time = frame * 0.5  # 0.5 minute steps
            if current_time <= duration:
                # This is a simplified version - in practice, you'd need to modify
                # the simulator to run step-by-step
                pass
            
            # Plot queue lengths over time
            if time_data:
                ax1.plot(time_data, queue_data['cashier'], label='Cashier', linewidth=2)
                ax1.plot(time_data, queue_data['barista'], label='Barista', linewidth=2)
                ax1.plot(time_data, queue_data['sealer'], label='Sealer', linewidth=2)
                ax1.set_xlabel('Time (minutes)')
                ax1.set_ylabel('Queue Length')
                ax1.set_title('Queue Lengths Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot current system state
            stations = ['Cashier', 'Barista', 'Sealer']
            current_queues = [len(queue_data['cashier']), len(queue_data['barista']), len(queue_data['sealer'])]
            
            bars = ax2.bar(stations, current_queues, color=['red', 'green', 'blue'], alpha=0.7)
            ax2.set_ylabel('Current Queue Length')
            ax2.set_title('Current System State')
            ax2.set_ylim(0, max(10, max(current_queues) + 2))
            
            # Add value labels
            for bar, value in zip(bars, current_queues):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=int(duration * 2), interval=500, repeat=False)
        
        plt.suptitle('Real-Time Boba Shop Simulation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def create_optimization_landscape(self, results_dict):
        """Create optimization landscape visualization"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create 3D surface plot for optimization landscape
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Generate optimization landscape data
        barista_range = np.arange(1, 6)
        sealer_range = np.arange(1, 4)
        B, S = np.meshgrid(barista_range, sealer_range)
        
        # Mock optimization surface (wait time as function of barista and sealer capacity)
        Z = 10 - 2*B - 1.5*S + 0.1*B*S + np.random.normal(0, 0.5, B.shape)
        
        surf = ax1.plot_surface(B, S, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Barista Capacity')
        ax1.set_ylabel('Sealer Capacity')
        ax1.set_zlabel('Wait Time (minutes)')
        ax1.set_title('Optimization Landscape')
        
        # Add optimal point
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        ax1.scatter([B[min_idx]], [S[min_idx]], [Z[min_idx]], color='red', s=100, label='Optimal')
        
        # Contour plot
        ax2 = fig.add_subplot(222)
        contour = ax2.contour(B, S, Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.scatter(B[min_idx], S[min_idx], color='red', s=100, label='Optimal')
        ax2.set_xlabel('Barista Capacity')
        ax2.set_ylabel('Sealer Capacity')
        ax2.set_title('Optimization Contours')
        ax2.legend()
        
        # Performance vs Cost
        ax3 = fig.add_subplot(223)
        policies = list(results_dict.keys())
        costs = [100, 150, 120]  # Mock costs
        performances = []
        
        for policy in policies:
            policy_results = [r['avg_wait_time'] for r in results_dict[policy] if 'error' not in r]
            performances.append(np.mean(policy_results))
        
        scatter = ax3.scatter(costs, performances, s=200, c=range(len(policies)), cmap='viridis')
        for i, policy in enumerate(policies):
            ax3.annotate(policy, (costs[i], performances[i]), xytext=(5, 5), textcoords='offset points')
        
        ax3.set_xlabel('Cost (relative)')
        ax3.set_ylabel('Average Wait Time (minutes)')
        ax3.set_title('Performance vs Cost')
        ax3.grid(True, alpha=0.3)
        
        # Sensitivity analysis
        ax4 = fig.add_subplot(224)
        parameters = ['Cashier Cap', 'Barista Cap', 'Sealer Cap', 'Arrival Rate']
        sensitivities = [0.1, 0.8, 0.3, 0.9]  # Mock sensitivity values
        
        bars = ax4.barh(parameters, sensitivities, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
        ax4.set_xlabel('Sensitivity Index')
        ax4.set_title('Parameter Sensitivity Analysis')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, sensitivities):
            width = bar.get_width()
            ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save optimization landscape
        opt_path = os.path.join(self.results_dir, f'optimization_landscape_{self.timestamp}.png')
        plt.savefig(opt_path, dpi=300, bbox_inches='tight')
        print(f"Optimization landscape saved to: {opt_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, results_dict):
        """Create a comprehensive analysis report with all visualizations"""
        print("Generating comprehensive analysis report...")
        
        # Create all visualizations
        self.create_comprehensive_dashboard(results_dict)
        self.create_optimization_landscape(results_dict)
        
        # Create individual detailed plots
        self._create_detailed_plots(results_dict)
        
        print(f"All visualizations saved to: {self.results_dir}/")
    
    def _create_detailed_plots(self, results_dict):
        """Create detailed individual plots"""
        
        # 1. Wait Time Box Plots
        plt.figure(figsize=(10, 6))
        wait_data = []
        labels = []
        
        for policy, results in results_dict.items():
            policy_wait_times = [r['avg_wait_time'] for r in results if 'error' not in r]
            wait_data.append(policy_wait_times)
            labels.append(policy)
        
        plt.boxplot(wait_data, labels=labels)
        plt.title('Wait Time Distribution by Policy', fontsize=14, fontweight='bold')
        plt.ylabel('Wait Time (minutes)')
        plt.grid(True, alpha=0.3)
        
        wait_plot_path = os.path.join(self.results_dir, f'wait_time_boxplot_{self.timestamp}.png')
        plt.savefig(wait_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Throughput Comparison
        plt.figure(figsize=(10, 6))
        throughput_data = []
        
        for policy, results in results_dict.items():
            policy_throughput = [r['throughput'] for r in results if 'error' not in r]
            throughput_data.append(policy_throughput)
        
        plt.boxplot(throughput_data, labels=labels)
        plt.title('Throughput Distribution by Policy', fontsize=14, fontweight='bold')
        plt.ylabel('Throughput (customers/hour)')
        plt.grid(True, alpha=0.3)
        
        throughput_plot_path = os.path.join(self.results_dir, f'throughput_boxplot_{self.timestamp}.png')
        plt.savefig(throughput_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Performance Radar Chart
        self._create_radar_chart(results_dict)
        
        # 4. Time Series Analysis
        self._create_time_series_analysis(results_dict)
    
    def _create_radar_chart(self, results_dict):
        """Create radar chart for multi-dimensional performance comparison"""
        from math import pi
        
        # Calculate normalized metrics for each policy
        metrics = ['avg_wait_time', 'p95_wait_time', 'throughput', 'total_customers']
        metric_labels = ['Avg Wait Time', '95th %ile Wait', 'Throughput', 'Total Customers']
        
        # Normalize metrics (lower is better for wait times, higher is better for throughput)
        normalized_data = {}
        
        for policy, results in results_dict.items():
            policy_results = [r for r in results if 'error' not in r]
            if policy_results:
                values = []
                for metric in metrics:
                    mean_val = np.mean([r[metric] for r in policy_results])
                    values.append(mean_val)
                
                # Normalize (invert wait times so higher is better)
                normalized_values = []
                for i, value in enumerate(values):
                    if i < 2:  # Wait times - invert
                        normalized_values.append(1 / (1 + value))
                    else:  # Throughput and customers - normalize
                        normalized_values.append(value / max(values))
                
                normalized_data[policy] = normalized_values
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(len(metric_labels)) * 2 * pi for n in range(len(metric_labels))]
        angles += angles[:1]  # Complete the circle
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (policy, values) in enumerate(normalized_data.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=policy, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        radar_path = os.path.join(self.results_dir, f'radar_chart_{self.timestamp}.png')
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_time_series_analysis(self, results_dict):
        """Create time series analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mock time series data (in real implementation, this would come from simulation)
        time_points = np.linspace(0, 8, 100)  # 8 hours
        
        # 1. Arrival Rate Over Time
        ax1 = axes[0, 0]
        arrival_pattern = 8 + 17 * np.sin(2 * np.pi * time_points / 8 - np.pi/2) + np.random.normal(0, 2, len(time_points))
        ax1.plot(time_points, arrival_pattern, linewidth=2, color='blue')
        ax1.set_title('Arrival Rate Over Time')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Arrival Rate (customers/hour)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Queue Length Over Time
        ax2 = axes[0, 1]
        for policy, results in results_dict.items():
            # Mock queue length data
            base_queue = np.mean([r['avg_wait_time'] for r in results if 'error' not in r]) / 2
            queue_pattern = base_queue * (1 + 0.5 * np.sin(2 * np.pi * time_points / 4)) + np.random.normal(0, 0.5, len(time_points))
            ax2.plot(time_points, queue_pattern, label=policy, linewidth=2)
        
        ax2.set_title('Queue Length Over Time')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Queue Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Service Rate Over Time
        ax3 = axes[1, 0]
        service_rate = 20 + 5 * np.sin(2 * np.pi * time_points / 8) + np.random.normal(0, 1, len(time_points))
        ax3.plot(time_points, service_rate, linewidth=2, color='green')
        ax3.set_title('Service Rate Over Time')
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Service Rate (customers/hour)')
        ax3.grid(True, alpha=0.3)
        
        # 4. System Utilization Over Time
        ax4 = axes[1, 1]
        utilization = 0.6 + 0.3 * np.sin(2 * np.pi * time_points / 8 - np.pi/2) + np.random.normal(0, 0.05, len(time_points))
        ax4.plot(time_points, utilization, linewidth=2, color='red')
        ax4.set_title('System Utilization Over Time')
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Utilization Rate')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        time_series_path = os.path.join(self.results_dir, f'time_series_analysis_{self.timestamp}.png')
        plt.savefig(time_series_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_interactive_dashboard():
    """Create an interactive dashboard with sliders and controls"""
    from boba_simulator import create_baseline_config, BobaShopSimulator
    
    # Create the main figure
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax_metrics = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
    ax_controls = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Hide the controls subplot for now
    ax_controls.axis('off')
    
    # Create sliders
    slider_ax_cashier = plt.axes([0.1, 0.02, 0.15, 0.03])
    slider_ax_barista = plt.axes([0.3, 0.02, 0.15, 0.03])
    slider_ax_sealer = plt.axes([0.5, 0.02, 0.15, 0.03])
    slider_ax_arrival = plt.axes([0.7, 0.02, 0.15, 0.03])
    
    slider_cashier = Slider(slider_ax_cashier, 'Cashiers', 1, 3, valinit=1, valstep=1)
    slider_barista = Slider(slider_ax_barista, 'Baristas', 1, 5, valinit=2, valstep=1)
    slider_sealer = Slider(slider_ax_sealer, 'Sealers', 1, 3, valinit=1, valstep=1)
    slider_arrival = Slider(slider_ax_arrival, 'Arrival Rate', 0.5, 2.0, valinit=1.0, valstep=0.1)
    
    # Update function
    def update_dashboard(val):
        ax_main.clear()
        ax_metrics.clear()
        
        # Get current values
        config = create_baseline_config()
        config['cashier_capacity'] = int(slider_cashier.val)
        config['barista_capacity'] = int(slider_barista.val)
        config['sealer_capacity'] = int(slider_sealer.val)
        
        # Run simulation
        simulator = BobaShopSimulator(config)
        results = simulator.run_simulation(simulation_time=60)  # 1 hour
        
        if 'error' not in results:
            # Main plot: System overview
            stations = ['Cashier', 'Barista', 'Sealer']
            capacities = [config['cashier_capacity'], config['barista_capacity'], config['sealer_capacity']]
            
            bars = ax_main.bar(stations, capacities, color=['blue', 'green', 'purple'], alpha=0.7)
            ax_main.set_title('Current System Configuration', fontsize=14, fontweight='bold')
            ax_main.set_ylabel('Capacity')
            ax_main.set_ylim(0, 6)
            
            # Add value labels
            for bar, value in zip(bars, capacities):
                height = bar.get_height()
                ax_main.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # Metrics plot
            metrics = ['avg_wait_time', 'p95_wait_time', 'throughput']
            values = [results[metric] for metric in metrics]
            labels = ['Avg Wait\n(min)', '95th %ile\n(min)', 'Throughput\n(cust/hr)']
            
            bars2 = ax_metrics.bar(labels, values, color=['red', 'orange', 'green'], alpha=0.7)
            ax_metrics.set_title('Performance Metrics', fontsize=14, fontweight='bold')
            ax_metrics.set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars2, values):
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                              f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        fig.canvas.draw()
    
    # Connect sliders
    slider_cashier.on_changed(update_dashboard)
    slider_barista.on_changed(update_dashboard)
    slider_sealer.on_changed(update_dashboard)
    slider_arrival.on_changed(update_dashboard)
    
    # Initial update
    update_dashboard(None)
    
    plt.suptitle('Interactive Boba Shop Simulator Dashboard', fontsize=16, fontweight='bold')
    plt.show()

if __name__ == "__main__":
    # Example usage
    visualizer = EnhancedVisualizer()
    
    # Create some mock results for demonstration
    mock_results = {
        'baseline': [{'avg_wait_time': 5.2, 'p95_wait_time': 12.1, 'throughput': 18.5, 'total_customers': 148}],
        'automation': [{'avg_wait_time': 4.1, 'p95_wait_time': 9.8, 'throughput': 22.3, 'total_customers': 178}],
        'staffing': [{'avg_wait_time': 3.8, 'p95_wait_time': 8.9, 'throughput': 24.1, 'total_customers': 193}]
    }
    
    # Create comprehensive report
    visualizer.create_comprehensive_report(mock_results)
    
    # Create interactive dashboard
    create_interactive_dashboard()
