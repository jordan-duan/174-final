"""
Simulation Animation Module for Boba Shop Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This module provides animated visualizations of the boba shop simulation
showing customer flow, queue dynamics, and system performance in real-time.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Tuple, Any
import time
from boba_simulator import BobaShopSimulator, create_baseline_config

class BobaShopAnimator:
    """Animated visualization of the boba shop simulation"""
    
    def __init__(self, config):
        """Initialize the animator with simulation configuration"""
        self.config = config
        self.simulator = BobaShopSimulator(config)
        
        # Animation data
        self.time_data = []
        self.queue_data = {'cashier': [], 'barista': [], 'sealer': []}
        self.customer_data = []
        self.performance_data = {'wait_time': [], 'throughput': []}
        
        # Animation parameters
        self.current_time = 0
        self.animation_speed = 1.0  # Speed multiplier
        
    def create_shop_layout(self, ax):
        """Create visual representation of the boba shop layout"""
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # Shop layout
        # Cashier station
        cashier_rect = patches.Rectangle((1, 6), 2, 1, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7)
        ax.add_patch(cashier_rect)
        ax.text(2, 6.5, 'CASHIER', ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Barista station
        barista_rect = patches.Rectangle((1, 4), 2, 1, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7)
        ax.add_patch(barista_rect)
        ax.text(2, 4.5, 'BARISTA', ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Sealer station
        sealer_rect = patches.Rectangle((1, 2), 2, 1, linewidth=2, edgecolor='purple', facecolor='plum', alpha=0.7)
        ax.add_patch(sealer_rect)
        ax.text(2, 2.5, 'SEALER', ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Queues
        # Cashier queue
        cashier_queue_rect = patches.Rectangle((4, 5.5), 3, 0.5, linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5)
        ax.add_patch(cashier_queue_rect)
        ax.text(5.5, 5.75, 'Queue', ha='center', va='center', fontsize=8)
        
        # Barista queue
        barista_queue_rect = patches.Rectangle((4, 3.5), 3, 0.5, linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5)
        ax.add_patch(barista_queue_rect)
        ax.text(5.5, 3.75, 'Queue', ha='center', va='center', fontsize=8)
        
        # Sealer queue
        sealer_queue_rect = patches.Rectangle((4, 1.5), 3, 0.5, linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5)
        ax.add_patch(sealer_queue_rect)
        ax.text(5.5, 1.75, 'Queue', ha='center', va='center', fontsize=8)
        
        # Customer flow arrows
        ax.arrow(3, 6.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(3, 4.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax.arrow(3, 2.5, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Exit
        ax.arrow(0.5, 2.5, -0.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.text(0.2, 2.5, 'EXIT', ha='center', va='center', fontweight='bold', color='red', fontsize=8)
        
        ax.set_title('Boba Shop Layout', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def draw_customers(self, ax, queue_lengths):
        """Draw customers in queues and at stations"""
        # Draw customers in cashier queue
        for i in range(min(queue_lengths['cashier'], 5)):  # Max 5 visible
            x = 4.2 + i * 0.3
            y = 5.6
            customer = patches.Circle((x, y), 0.1, facecolor='orange', edgecolor='black')
            ax.add_patch(customer)
        
        # Draw customers in barista queue
        for i in range(min(queue_lengths['barista'], 5)):
            x = 4.2 + i * 0.3
            y = 3.6
            customer = patches.Circle((x, y), 0.1, facecolor='orange', edgecolor='black')
            ax.add_patch(customer)
        
        # Draw customers in sealer queue
        for i in range(min(queue_lengths['sealer'], 5)):
            x = 4.2 + i * 0.3
            y = 1.6
            customer = patches.Circle((x, y), 0.1, facecolor='orange', edgecolor='black')
            ax.add_patch(customer)
        
        # Draw customers being served
        # Cashier
        if queue_lengths['cashier'] > 0:
            customer = patches.Circle((2, 6.5), 0.1, facecolor='red', edgecolor='black')
            ax.add_patch(customer)
        
        # Barista
        if queue_lengths['barista'] > 0:
            customer = patches.Circle((2, 4.5), 0.1, facecolor='red', edgecolor='black')
            ax.add_patch(customer)
        
        # Sealer
        if queue_lengths['sealer'] > 0:
            customer = patches.Circle((2, 2.5), 0.1, facecolor='red', edgecolor='black')
            ax.add_patch(customer)
    
    def create_animated_simulation(self, duration=60):
        """Create animated simulation visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Initialize plots
        self.create_shop_layout(ax1)
        
        # Performance plots
        ax2.set_title('Queue Lengths Over Time')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Queue Length')
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Performance Metrics')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Wait Time (minutes)')
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('System Status')
        ax4.set_xlabel('Station')
        ax4.set_ylabel('Current Queue Length')
        
        # Animation function
        def animate(frame):
            # Update time
            self.current_time += 0.5  # 0.5 minute steps
            
            # Simulate queue lengths (in real implementation, this would come from simulation)
            queue_lengths = {
                'cashier': max(0, int(2 + 3 * np.sin(self.current_time / 10) + np.random.normal(0, 0.5))),
                'barista': max(0, int(3 + 4 * np.sin(self.current_time / 8) + np.random.normal(0, 0.8))),
                'sealer': max(0, int(1 + 2 * np.sin(self.current_time / 12) + np.random.normal(0, 0.3)))
            }
            
            # Update data
            self.time_data.append(self.current_time)
            for station in queue_lengths:
                self.queue_data[station].append(queue_lengths[station])
            
            # Keep only last 100 points
            if len(self.time_data) > 100:
                self.time_data = self.time_data[-100:]
                for station in self.queue_data:
                    self.queue_data[station] = self.queue_data[station][-100:]
            
            # Update shop layout
            self.create_shop_layout(ax1)
            self.draw_customers(ax1, queue_lengths)
            
            # Update queue length plot
            ax2.clear()
            ax2.plot(self.time_data, self.queue_data['cashier'], label='Cashier', linewidth=2, color='blue')
            ax2.plot(self.time_data, self.queue_data['barista'], label='Barista', linewidth=2, color='green')
            ax2.plot(self.time_data, self.queue_data['sealer'], label='Sealer', linewidth=2, color='purple')
            ax2.set_title('Queue Lengths Over Time')
            ax2.set_xlabel('Time (minutes)')
            ax2.set_ylabel('Queue Length')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Update performance plot
            ax3.clear()
            # Simulate wait time based on queue lengths
            avg_wait = np.mean(list(queue_lengths.values())) * 2  # Rough approximation
            self.performance_data['wait_time'].append(avg_wait)
            
            if len(self.performance_data['wait_time']) > 100:
                self.performance_data['wait_time'] = self.performance_data['wait_time'][-100:]
            
            ax3.plot(self.time_data, self.performance_data['wait_time'], linewidth=2, color='red')
            ax3.set_title('Average Wait Time')
            ax3.set_xlabel('Time (minutes)')
            ax3.set_ylabel('Wait Time (minutes)')
            ax3.grid(True, alpha=0.3)
            
            # Update system status
            ax4.clear()
            stations = ['Cashier', 'Barista', 'Sealer']
            current_queues = [queue_lengths['cashier'], queue_lengths['barista'], queue_lengths['sealer']]
            colors = ['red' if q > 3 else 'orange' if q > 1 else 'green' for q in current_queues]
            
            bars = ax4.bar(stations, current_queues, color=colors, alpha=0.7)
            ax4.set_title('Current System Status')
            ax4.set_ylabel('Queue Length')
            ax4.set_ylim(0, 8)
            
            # Add value labels
            for bar, value in zip(bars, current_queues):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Add status text
            total_customers = sum(current_queues)
            ax4.text(0.5, 7, f'Total in System: {total_customers}', 
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=int(duration * 2), 
                                     interval=500, repeat=True, blit=False)
        
        plt.suptitle('Animated Boba Shop Simulation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return anim

class PerformanceAnimator:
    """Animated performance visualization"""
    
    def __init__(self):
        """Initialize performance animator"""
        self.time_data = []
        self.metrics_data = {
            'wait_time': [],
            'throughput': [],
            'utilization': []
        }
    
    def create_performance_animation(self, results_dict, duration=30):
        """Create animated performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Initialize data
        policies = list(results_dict.keys())
        policy_colors = ['red', 'blue', 'green', 'orange']
        
        # Animation function
        def animate(frame):
            current_time = frame * 0.5
            
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # Simulate performance data over time
            for i, (policy, results) in enumerate(results_dict.items()):
                policy_results = [r for r in results if 'error' not in r]
                if policy_results:
                    base_wait = np.mean([r['avg_wait_time'] for r in policy_results])
                    base_throughput = np.mean([r['throughput'] for r in policy_results])
                    
                    # Add time variation
                    wait_variation = base_wait * (1 + 0.2 * np.sin(current_time / 5))
                    throughput_variation = base_throughput * (1 + 0.1 * np.sin(current_time / 8))
                    
                    # Update data
                    if policy not in self.metrics_data:
                        self.metrics_data[policy] = {'wait_time': [], 'throughput': []}
                    
                    self.metrics_data[policy]['wait_time'].append(wait_variation)
                    self.metrics_data[policy]['throughput'].append(throughput_variation)
                    
                    # Keep only last 50 points
                    if len(self.metrics_data[policy]['wait_time']) > 50:
                        self.metrics_data[policy]['wait_time'] = self.metrics_data[policy]['wait_time'][-50:]
                        self.metrics_data[policy]['throughput'] = self.metrics_data[policy]['throughput'][-50:]
            
            # Plot 1: Wait time comparison
            time_points = np.arange(len(self.metrics_data[policies[0]]['wait_time'])) * 0.5
            for i, policy in enumerate(policies):
                if policy in self.metrics_data and self.metrics_data[policy]['wait_time']:
                    axes[0, 0].plot(time_points, self.metrics_data[policy]['wait_time'], 
                                   label=policy, color=policy_colors[i], linewidth=2)
            
            axes[0, 0].set_title('Wait Time Comparison Over Time')
            axes[0, 0].set_xlabel('Time (minutes)')
            axes[0, 0].set_ylabel('Wait Time (minutes)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Throughput comparison
            for i, policy in enumerate(policies):
                if policy in self.metrics_data and self.metrics_data[policy]['throughput']:
                    axes[0, 1].plot(time_points, self.metrics_data[policy]['throughput'], 
                                   label=policy, color=policy_colors[i], linewidth=2)
            
            axes[0, 1].set_title('Throughput Comparison Over Time')
            axes[0, 1].set_xlabel('Time (minutes)')
            axes[0, 1].set_ylabel('Throughput (customers/hour)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Current performance bars
            current_wait_times = []
            current_throughputs = []
            
            for policy in policies:
                if policy in self.metrics_data and self.metrics_data[policy]['wait_time']:
                    current_wait_times.append(self.metrics_data[policy]['wait_time'][-1])
                    current_throughputs.append(self.metrics_data[policy]['throughput'][-1])
                else:
                    current_wait_times.append(0)
                    current_throughputs.append(0)
            
            x_pos = np.arange(len(policies))
            bars1 = axes[1, 0].bar(x_pos, current_wait_times, color=policy_colors[:len(policies)], alpha=0.7)
            axes[1, 0].set_title('Current Wait Times')
            axes[1, 0].set_xlabel('Policy')
            axes[1, 0].set_ylabel('Wait Time (minutes)')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(policies)
            
            # Add value labels
            for bar, value in zip(bars1, current_wait_times):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Current throughput bars
            bars2 = axes[1, 1].bar(x_pos, current_throughputs, color=policy_colors[:len(policies)], alpha=0.7)
            axes[1, 1].set_title('Current Throughput')
            axes[1, 1].set_xlabel('Policy')
            axes[1, 1].set_ylabel('Throughput (customers/hour)')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(policies)
            
            # Add value labels
            for bar, value in zip(bars2, current_throughputs):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=int(duration * 2), 
                                     interval=1000, repeat=True, blit=False)
        
        plt.suptitle('Animated Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return anim

def create_simulation_animation_demo():
    """Create demonstration of simulation animations"""
    print("🎬 Creating Simulation Animation Demo...")
    
    # Create shop layout animation
    config = create_baseline_config()
    shop_animator = BobaShopAnimator(config)
    
    print("   Creating shop layout animation...")
    shop_anim = shop_animator.create_animated_simulation(duration=30)
    
    # Create performance animation
    from boba_simulator import run_monte_carlo_simulation, create_automation_config, create_staffing_config
    
    print("   Creating performance comparison animation...")
    
    # Get some results for animation
    policies = {
        'baseline': create_baseline_config(),
        'automation': create_automation_config(),
        'staffing': create_staffing_config()
    }
    
    results_dict = {}
    for policy_name, policy_config in policies.items():
        results = run_monte_carlo_simulation(policy_config, num_replications=5, simulation_time=60)
        results_dict[policy_name] = results
    
    perf_animator = PerformanceAnimator()
    perf_anim = perf_animator.create_performance_animation(results_dict, duration=20)
    
    print("   Animation demo completed!")
    return shop_anim, perf_anim

if __name__ == "__main__":
    create_simulation_animation_demo()
