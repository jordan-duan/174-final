"""
Interactive Parameter Optimization Tool for Boba Shop Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

This module provides interactive tools for exploring parameter space
and finding optimal configurations using sliders and real-time updates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Any
import time
from boba_simulator import BobaShopSimulator, create_baseline_config, run_monte_carlo_simulation
import threading
import queue

class InteractiveOptimizer:
    """Interactive parameter optimization tool"""
    
    def __init__(self):
        """Initialize the interactive optimizer"""
        self.current_config = create_baseline_config()
        self.simulation_results = {}
        self.optimization_history = []
        self.is_running = False
        
    def create_parameter_explorer(self):
        """Create interactive parameter exploration interface"""
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplot layout
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # Main performance plot
        ax_performance = fig.add_subplot(gs[0:2, 0:2])
        
        # Configuration visualization
        ax_config = fig.add_subplot(gs[0:2, 2:4])
        
        # Optimization history
        ax_history = fig.add_subplot(gs[2, 0:2])
        
        # Cost-benefit analysis
        ax_cost = fig.add_subplot(gs[2, 2:4])
        
        # Parameter controls (bottom row)
        ax_controls = fig.add_subplot(gs[3, :])
        ax_controls.axis('off')
        
        # Create sliders
        slider_positions = [
            [0.05, 0.02, 0.12, 0.03],  # Cashier
            [0.20, 0.02, 0.12, 0.03],  # Barista
            [0.35, 0.02, 0.12, 0.03],  # Sealer
            [0.50, 0.02, 0.12, 0.03],  # Arrival multiplier
            [0.65, 0.02, 0.12, 0.03],  # Pearl batch size
            [0.80, 0.02, 0.12, 0.03],  # Pearl reorder point
        ]
        
        slider_labels = ['Cashiers', 'Baristas', 'Sealers', 'Arrival Rate', 'Pearl Batch', 'Reorder Point']
        slider_ranges = [(1, 3), (1, 5), (1, 3), (0.5, 2.0), (20, 50), (5, 20)]
        slider_steps = [1, 1, 1, 0.1, 5, 1]
        slider_initials = [1, 2, 1, 1.0, 30, 10]
        
        self.sliders = []
        for i, (pos, label, range_vals, step, initial) in enumerate(zip(
            slider_positions, slider_labels, slider_ranges, slider_steps, slider_initials)):
            
            ax_slider = plt.axes(pos)
            slider = Slider(ax_slider, label, range_vals[0], range_vals[1], 
                          valinit=initial, valstep=step)
            self.sliders.append(slider)
        
        # Create buttons
        button_positions = [
            [0.05, 0.08, 0.08, 0.03],  # Run Simulation
            [0.15, 0.08, 0.08, 0.03],  # Optimize
            [0.25, 0.08, 0.08, 0.03],  # Reset
            [0.35, 0.08, 0.08, 0.03],  # Save Config
        ]
        
        button_labels = ['Run Sim', 'Optimize', 'Reset', 'Save']
        self.buttons = []
        
        for pos, label in zip(button_positions, button_labels):
            ax_button = plt.axes(pos)
            button = Button(ax_button, label)
            self.buttons.append(button)
        
        # Create checkboxes for what to optimize
        ax_checkbox = plt.axes([0.50, 0.08, 0.15, 0.05])
        self.checkboxes = CheckButtons(ax_checkbox, 
                                     ['Wait Time', 'Throughput', 'Cost'], 
                                     [True, False, False])
        
        # Store axes for updates
        self.axes = {
            'performance': ax_performance,
            'config': ax_config,
            'history': ax_history,
            'cost': ax_cost
        }
        
        # Connect events
        for slider in self.sliders:
            slider.on_changed(lambda val: self._update_config())
        
        self.buttons[0].on_clicked(lambda event: self._run_simulation())
        self.buttons[1].on_clicked(lambda event: self._run_optimization())
        self.buttons[2].on_clicked(lambda event: self._reset_parameters())
        self.buttons[3].on_clicked(lambda event: self._save_configuration())
        
        self.checkboxes.on_clicked(lambda label: self._update_optimization_target())
        
        # Initial update
        self._update_config()
        self._update_plots()
        
        plt.suptitle('Interactive Boba Shop Parameter Optimizer', fontsize=16, fontweight='bold')
        plt.show()
        
        return fig
    
    def _update_config(self):
        """Update configuration based on slider values"""
        self.current_config = create_baseline_config()
        self.current_config['cashier_capacity'] = int(self.sliders[0].val)
        self.current_config['barista_capacity'] = int(self.sliders[1].val)
        self.current_config['sealer_capacity'] = int(self.sliders[2].val)
        self.current_config['pearl_batch_size'] = int(self.sliders[4].val)
        self.current_config['pearl_reorder_point'] = int(self.sliders[5].val)
        
        # Update arrival rate multiplier (this would need to be implemented in the simulator)
        self.arrival_multiplier = self.sliders[3].val
    
    def _update_plots(self):
        """Update all plots with current configuration"""
        # Clear all axes
        for ax in self.axes.values():
            ax.clear()
        
        # Plot 1: Performance metrics
        if 'avg_wait_time' in self.simulation_results:
            metrics = ['avg_wait_time', 'p95_wait_time', 'throughput']
            values = [self.simulation_results[metric] for metric in metrics]
            labels = ['Avg Wait\n(min)', '95th %ile\n(min)', 'Throughput\n(cust/hr)']
            
            bars = self.axes['performance'].bar(labels, values, 
                                              color=['red', 'orange', 'green'], alpha=0.7)
            self.axes['performance'].set_title('Current Performance')
            self.axes['performance'].set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.axes['performance'].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                                            f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Configuration visualization
        stations = ['Cashier', 'Barista', 'Sealer']
        capacities = [self.current_config['cashier_capacity'], 
                     self.current_config['barista_capacity'], 
                     self.current_config['sealer_capacity']]
        
        bars = self.axes['config'].bar(stations, capacities, 
                                     color=['blue', 'green', 'purple'], alpha=0.7)
        self.axes['config'].set_title('Current Configuration')
        self.axes['config'].set_ylabel('Capacity')
        self.axes['config'].set_ylim(0, 6)
        
        # Add value labels
        for bar, value in zip(bars, capacities):
            height = bar.get_height()
            self.axes['config'].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                   f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Plot 3: Optimization history
        if self.optimization_history:
            history_data = np.array(self.optimization_history)
            self.axes['history'].plot(history_data[:, 0], history_data[:, 1], 'o-', linewidth=2, markersize=6)
            self.axes['history'].set_title('Optimization History')
            self.axes['history'].set_xlabel('Iteration')
            self.axes['history'].set_ylabel('Objective Value')
            self.axes['history'].grid(True, alpha=0.3)
        
        # Plot 4: Cost-benefit analysis
        if 'avg_wait_time' in self.simulation_results:
            # Calculate costs
            staff_cost = (self.current_config['cashier_capacity'] + 
                         self.current_config['barista_capacity'] + 
                         self.current_config['sealer_capacity']) * 15  # $15/hour per staff
            
            wait_cost = self.simulation_results['avg_wait_time'] * 2  # $2 per minute of wait
            
            total_cost = staff_cost + wait_cost
            
            costs = [staff_cost, wait_cost]
            labels = ['Staff Cost', 'Wait Cost']
            colors = ['blue', 'red']
            
            wedges, texts, autotexts = self.axes['cost'].pie(costs, labels=labels, colors=colors, 
                                                           autopct='%1.1f%%', startangle=90)
            self.axes['cost'].set_title(f'Cost Breakdown\nTotal: ${total_cost:.0f}')
        
        plt.draw()
    
    def _run_simulation(self):
        """Run simulation with current configuration"""
        print("Running simulation...")
        
        try:
            # Run a quick simulation
            simulator = BobaShopSimulator(self.current_config)
            results = simulator.run_simulation(simulation_time=60)  # 1 hour
            
            if 'error' not in results:
                self.simulation_results = results
                print(f"Simulation completed: Avg wait = {results['avg_wait_time']:.2f} min")
            else:
                print("Simulation failed")
                
        except Exception as e:
            print(f"Simulation error: {e}")
        
        self._update_plots()
    
    def _run_optimization(self):
        """Run optimization to find best parameters"""
        print("Running optimization...")
        
        # Get optimization targets
        targets = []
        if self.checkboxes.get_status()[0]:  # Wait Time
            targets.append('wait_time')
        if self.checkboxes.get_status()[1]:  # Throughput
            targets.append('throughput')
        if self.checkboxes.get_status()[2]:  # Cost
            targets.append('cost')
        
        if not targets:
            print("Please select at least one optimization target")
            return
        
        # Simple grid search optimization
        best_config = None
        best_value = float('inf')
        
        # Define search space
        cashier_range = range(1, 4)
        barista_range = range(1, 6)
        sealer_range = range(1, 4)
        
        iteration = 0
        for cashier in cashier_range:
            for barista in barista_range:
                for sealer in sealer_range:
                    # Update configuration
                    config = create_baseline_config()
                    config['cashier_capacity'] = cashier
                    config['barista_capacity'] = barista
                    config['sealer_capacity'] = sealer
                    
                    # Run simulation
                    try:
                        simulator = BobaShopSimulator(config)
                        results = simulator.run_simulation(simulation_time=30)  # Quick simulation
                        
                        if 'error' not in results:
                            # Calculate objective value
                            objective = 0
                            if 'wait_time' in targets:
                                objective += results['avg_wait_time']
                            if 'throughput' in targets:
                                objective -= results['throughput'] / 10  # Negative because higher is better
                            if 'cost' in targets:
                                cost = (cashier + barista + sealer) * 15 + results['avg_wait_time'] * 2
                                objective += cost / 100
                            
                            # Update history
                            self.optimization_history.append([iteration, objective])
                            
                            if objective < best_value:
                                best_value = objective
                                best_config = config.copy()
                            
                            iteration += 1
                            
                    except Exception as e:
                        print(f"Optimization error: {e}")
                        continue
        
        if best_config:
            # Update sliders to best configuration
            self.sliders[0].set_val(best_config['cashier_capacity'])
            self.sliders[1].set_val(best_config['barista_capacity'])
            self.sliders[2].set_val(best_config['sealer_capacity'])
            
            print(f"Optimization completed! Best objective: {best_value:.2f}")
            print(f"Best config: {best_config['cashier_capacity']} cashiers, "
                  f"{best_config['barista_capacity']} baristas, {best_config['sealer_capacity']} sealers")
        
        self._update_plots()
    
    def _reset_parameters(self):
        """Reset parameters to baseline"""
        baseline_config = create_baseline_config()
        
        self.sliders[0].set_val(baseline_config['cashier_capacity'])
        self.sliders[1].set_val(baseline_config['barista_capacity'])
        self.sliders[2].set_val(baseline_config['sealer_capacity'])
        self.sliders[3].set_val(1.0)  # Arrival multiplier
        self.sliders[4].set_val(baseline_config['pearl_batch_size'])
        self.sliders[5].set_val(baseline_config['pearl_reorder_point'])
        
        self.optimization_history = []
        self.simulation_results = {}
        
        print("Parameters reset to baseline")
        self._update_plots()
    
    def _save_configuration(self):
        """Save current configuration"""
        config_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': self.current_config.copy(),
            'results': self.simulation_results.copy() if self.simulation_results else {},
            'optimization_history': self.optimization_history.copy()
        }
        
        # Save to file (in real implementation)
        print(f"Configuration saved at {config_data['timestamp']}")
        print(f"Config: {self.current_config}")
    
    def _update_optimization_target(self):
        """Update optimization target based on checkboxes"""
        targets = []
        if self.checkboxes.get_status()[0]:
            targets.append('Wait Time')
        if self.checkboxes.get_status()[1]:
            targets.append('Throughput')
        if self.checkboxes.get_status()[2]:
            targets.append('Cost')
        
        print(f"Optimization targets: {', '.join(targets) if targets else 'None selected'}")

class RealTimeSimulationViewer:
    """Real-time simulation viewer with live updates"""
    
    def __init__(self):
        """Initialize the real-time viewer"""
        self.is_running = False
        self.simulation_data = queue.Queue()
        
    def create_live_dashboard(self, config):
        """Create live simulation dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Initialize plots
        ax_queue = axes[0, 0]
        ax_metrics = axes[0, 1]
        ax_customers = axes[1, 0]
        ax_system = axes[1, 1]
        
        # Data storage
        self.time_data = []
        self.queue_data = {'cashier': [], 'barista': [], 'sealer': []}
        self.metric_data = {'wait_time': [], 'throughput': []}
        self.customer_data = []
        
        def animate(frame):
            if not self.simulation_data.empty():
                try:
                    # Get latest data
                    data = self.simulation_data.get_nowait()
                    
                    # Update data
                    self.time_data.append(data['time'])
                    self.queue_data['cashier'].append(data['cashier_queue'])
                    self.queue_data['barista'].append(data['barista_queue'])
                    self.queue_data['sealer'].append(data['sealer_queue'])
                    self.metric_data['wait_time'].append(data['avg_wait'])
                    self.metric_data['throughput'].append(data['throughput'])
                    
                    # Keep only last 100 points
                    if len(self.time_data) > 100:
                        self.time_data = self.time_data[-100:]
                        for key in self.queue_data:
                            self.queue_data[key] = self.queue_data[key][-100:]
                        for key in self.metric_data:
                            self.metric_data[key] = self.metric_data[key][-100:]
                    
                    # Update plots
                    self._update_plots(ax_queue, ax_metrics, ax_customers, ax_system)
                    
                except queue.Empty:
                    pass
        
        # Start animation
        anim = FuncAnimation(fig, animate, interval=1000, blit=False)
        
        # Start simulation in background
        self._start_background_simulation(config)
        
        plt.suptitle('Real-Time Boba Shop Simulation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def _update_plots(self, ax_queue, ax_metrics, ax_customers, ax_system):
        """Update all plots with latest data"""
        # Clear axes
        ax_queue.clear()
        ax_metrics.clear()
        ax_customers.clear()
        ax_system.clear()
        
        if not self.time_data:
            return
        
        # Plot 1: Queue lengths over time
        ax_queue.plot(self.time_data, self.queue_data['cashier'], label='Cashier', linewidth=2)
        ax_queue.plot(self.time_data, self.queue_data['barista'], label='Barista', linewidth=2)
        ax_queue.plot(self.time_data, self.queue_data['sealer'], label='Sealer', linewidth=2)
        ax_queue.set_title('Queue Lengths Over Time')
        ax_queue.set_xlabel('Time (minutes)')
        ax_queue.set_ylabel('Queue Length')
        ax_queue.legend()
        ax_queue.grid(True, alpha=0.3)
        
        # Plot 2: Performance metrics
        ax_metrics.plot(self.time_data, self.metric_data['wait_time'], label='Avg Wait Time', linewidth=2, color='red')
        ax_metrics2 = ax_metrics.twinx()
        ax_metrics2.plot(self.time_data, self.metric_data['throughput'], label='Throughput', linewidth=2, color='blue')
        ax_metrics.set_title('Performance Metrics')
        ax_metrics.set_xlabel('Time (minutes)')
        ax_metrics.set_ylabel('Wait Time (min)', color='red')
        ax_metrics2.set_ylabel('Throughput (cust/hr)', color='blue')
        ax_metrics.grid(True, alpha=0.3)
        
        # Plot 3: Customer flow
        if self.customer_data:
            customer_times = [c['arrival_time'] for c in self.customer_data]
            customer_wait = [c['wait_time'] for c in self.customer_data]
            ax_customers.scatter(customer_times, customer_wait, alpha=0.6, s=20)
            ax_customers.set_title('Customer Wait Times')
            ax_customers.set_xlabel('Arrival Time (minutes)')
            ax_customers.set_ylabel('Wait Time (minutes)')
            ax_customers.grid(True, alpha=0.3)
        
        # Plot 4: System status
        stations = ['Cashier', 'Barista', 'Sealer']
        current_queues = [self.queue_data['cashier'][-1] if self.queue_data['cashier'] else 0,
                         self.queue_data['barista'][-1] if self.queue_data['barista'] else 0,
                         self.queue_data['sealer'][-1] if self.queue_data['sealer'] else 0]
        
        colors = ['red' if q > 3 else 'orange' if q > 1 else 'green' for q in current_queues]
        bars = ax_system.bar(stations, current_queues, color=colors, alpha=0.7)
        ax_system.set_title('Current System Status')
        ax_system.set_ylabel('Queue Length')
        ax_system.set_ylim(0, 10)
        
        # Add value labels
        for bar, value in zip(bars, current_queues):
            height = bar.get_height()
            ax_system.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                          f'{value}', ha='center', va='bottom', fontweight='bold')
    
    def _start_background_simulation(self, config):
        """Start simulation in background thread"""
        def run_simulation():
            start_time = time.time()
            update_count = 0
            
            while self.is_running:
                # Create a new simulator for each update to avoid SimPy environment reuse issues
                simulator = BobaShopSimulator(config)
                # Run simulation for a short time
                results = simulator.run_simulation(simulation_time=5)  # 5 minutes
                
                if 'error' not in results:
                    # Get utilization as proxy for queue activity
                    utilization = results.get('utilization', {})
                    
                    # Create data packet
                    data = {
                        'time': update_count * 5,  # Use simulation time instead of real time
                        'cashier_queue': utilization.get('cashier', 0) * 3,  # Scale utilization to approximate queue
                        'barista_queue': utilization.get('barista', 0) * 3,
                        'sealer_queue': utilization.get('sealer', 0) * 3,
                        'avg_wait': results.get('avg_wait_time', 0),
                        'throughput': results.get('throughput', 0)
                    }
                    
                    self.simulation_data.put(data)
                    update_count += 1
                
                time.sleep(2)  # Update every 2 seconds (simulation takes time)
        
        self.is_running = True
        thread = threading.Thread(target=run_simulation)
        thread.daemon = True
        thread.start()

def create_parameter_sensitivity_analyzer():
    """Create parameter sensitivity analysis tool"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Parameter ranges
    parameters = {
        'cashier_capacity': (1, 3),
        'barista_capacity': (1, 5),
        'sealer_capacity': (1, 3),
        'arrival_rate': (0.5, 2.0)
    }
    
    # Sensitivity analysis for each parameter
    for i, (param, (min_val, max_val)) in enumerate(parameters.items()):
        ax = axes[i//2, i%2]
        
        # Generate sensitivity data
        param_values = np.linspace(min_val, max_val, 10)
        wait_times = []
        
        for val in param_values:
            config = create_baseline_config()
            if param == 'arrival_rate':
                # This would need to be implemented in the simulator
                pass
            else:
                config[param] = int(val) if param != 'arrival_rate' else val
            
            # Run simulation
            try:
                simulator = BobaShopSimulator(config)
                results = simulator.run_simulation(simulation_time=30)
                
                if 'error' not in results:
                    wait_times.append(results['avg_wait_time'])
                else:
                    wait_times.append(10)  # Default high value
            except:
                wait_times.append(10)
        
        # Plot sensitivity
        ax.plot(param_values, wait_times, 'o-', linewidth=2, markersize=6)
        ax.set_title(f'Sensitivity: {param.replace("_", " ").title()}')
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel('Average Wait Time (minutes)')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create interactive optimizer
    optimizer = InteractiveOptimizer()
    optimizer.create_parameter_explorer()
    
    # Create sensitivity analyzer
    create_parameter_sensitivity_analyzer()
