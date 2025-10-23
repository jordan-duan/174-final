"""
Boba Shop Operations Simulator
Authors: Jordan Duan and Valerie He
INDENG 174 - Professor Zheng

A discrete-event simulation of a boba shop to analyze operational efficiency
and identify optimal staffing and automation strategies.
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import seaborn as sns
from scipy import stats
import heapq

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class DrinkType(Enum):
    """Types of boba drinks with different service characteristics"""
    MILK_TEA = "milk_tea"
    FRUIT_TEA = "fruit_tea"
    SPECIALTY = "specialty"
    SIMPLE = "simple"

class ServiceStation(Enum):
    """Service stations in the boba shop"""
    CASHIER = "cashier"
    BARISTA = "barista"
    SEALER = "sealer"

@dataclass
class DrinkConfig:
    """Configuration for different drink types"""
    name: str
    service_time_mean: float  # minutes
    service_time_cv: float    # coefficient of variation
    requires_pearls: bool
    complexity_score: int     # 1-3 scale

@dataclass
class Customer:
    """Customer entity in the simulation"""
    customer_id: int
    arrival_time: float
    drink_type: DrinkType
    order_time: Optional[float] = None
    service_start_time: Optional[float] = None
    service_end_time: Optional[float] = None
    exit_time: Optional[float] = None

class BobaShopSimulator:
    """Main simulation class for the boba shop"""
    
    def __init__(self, config: Dict):
        """
        Initialize the boba shop simulator
        
        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.env = simpy.Environment()
        
        # Service stations
        self.cashier = simpy.Resource(self.env, capacity=config['cashier_capacity'])
        self.barista = simpy.Resource(self.env, capacity=config['barista_capacity'])
        self.sealer = simpy.Resource(self.env, capacity=config['sealer_capacity'])
        
        # Inventory system
        self.pearl_inventory = config['initial_pearl_inventory']
        self.pearl_reorder_point = config['pearl_reorder_point']
        self.pearl_batch_size = config['pearl_batch_size']
        self.pearl_cooker = simpy.Resource(self.env, capacity=1)
        
        # Statistics tracking
        self.customers_served = 0
        self.total_wait_time = 0
        self.total_service_time = 0
        self.customer_data = []
        self.queue_lengths = {station: [] for station in ServiceStation}
        self.utilization_data = {station: [] for station in ServiceStation}
        
        # Drink configurations
        self.drink_configs = {
            DrinkType.MILK_TEA: DrinkConfig("Milk Tea", 3.5, 0.3, True, 2),
            DrinkType.FRUIT_TEA: DrinkConfig("Fruit Tea", 4.0, 0.4, True, 3),
            DrinkType.SPECIALTY: DrinkConfig("Specialty", 5.0, 0.5, True, 3),
            DrinkType.SIMPLE: DrinkConfig("Simple", 2.0, 0.2, False, 1)
        }
        
        # Time-varying arrival rates (customers per hour)
        self.arrival_rates = {
            'morning': 8,    # 8 AM - 12 PM
            'lunch': 25,     # 12 PM - 2 PM
            'afternoon': 12, # 2 PM - 6 PM
            'evening': 18,   # 6 PM - 9 PM
            'night': 5       # 9 PM - 8 AM
        }
        
    def get_current_arrival_rate(self, current_time: float) -> float:
        """Get arrival rate based on time of day"""
        hour = (current_time % 24)
        if 8 <= hour < 12:
            return self.arrival_rates['morning']
        elif 12 <= hour < 14:
            return self.arrival_rates['lunch']
        elif 14 <= hour < 18:
            return self.arrival_rates['afternoon']
        elif 18 <= hour < 21:
            return self.arrival_rates['evening']
        else:
            return self.arrival_rates['night']
    
    def generate_drink_type(self) -> DrinkType:
        """Generate drink type based on probabilities"""
        drink_probs = {
            DrinkType.MILK_TEA: 0.4,
            DrinkType.FRUIT_TEA: 0.3,
            DrinkType.SPECIALTY: 0.2,
            DrinkType.SIMPLE: 0.1
        }
        return np.random.choice(list(drink_probs.keys()), p=list(drink_probs.values()))
    
    def generate_service_time(self, drink_type: DrinkType, station: ServiceStation) -> float:
        """Generate service time based on drink type and station"""
        config = self.drink_configs[drink_type]
        
        # Base service times by station (minutes)
        base_times = {
            ServiceStation.CASHIER: 1.0,
            ServiceStation.BARISTA: config.service_time_mean,
            ServiceStation.SEALER: 0.5
        }
        
        mean_time = base_times[station]
        cv = config.service_time_cv
        
        # Generate lognormal service time
        sigma = np.sqrt(np.log(1 + cv**2))
        mu = np.log(mean_time) - 0.5 * sigma**2
        
        return max(0.1, np.random.lognormal(mu, sigma))
    
    def customer_arrival_process(self):
        """Generate customer arrivals using Poisson process"""
        customer_id = 0
        
        while True:
            # Get current arrival rate
            current_rate = self.get_current_arrival_rate(self.env.now)
            
            # Generate inter-arrival time (exponential)
            inter_arrival_time = np.random.exponential(60 / current_rate)  # Convert to minutes
            
            yield self.env.timeout(inter_arrival_time)
            
            # Create new customer
            customer = Customer(
                customer_id=customer_id,
                arrival_time=self.env.now,
                drink_type=self.generate_drink_type()
            )
            
            # Start customer service process
            self.env.process(self.customer_service_process(customer))
            customer_id += 1
    
    def customer_service_process(self, customer: Customer):
        """Process a customer through all service stations"""
        start_time = self.env.now
        
        # 1. Cashier (Ordering)
        with self.cashier.request() as request:
            yield request
            customer.order_time = self.env.now
            
            # Record queue length
            self.queue_lengths[ServiceStation.CASHIER].append(
                (self.env.now, len(self.cashier.queue))
            )
            
            service_time = self.generate_service_time(customer.drink_type, ServiceStation.CASHIER)
            yield self.env.timeout(service_time)
        
        # 2. Barista (Assembly/Shake)
        with self.barista.request() as request:
            yield request
            customer.service_start_time = self.env.now
            
            # Record queue length
            self.queue_lengths[ServiceStation.BARISTA].append(
                (self.env.now, len(self.barista.queue))
            )
            
            # Check if pearls are needed and available
            if (customer.drink_type in [DrinkType.MILK_TEA, DrinkType.FRUIT_TEA, DrinkType.SPECIALTY] 
                and self.pearl_inventory <= 0):
                # Wait for pearl replenishment
                yield self.env.process(self.wait_for_pearls())
            
            # Consume pearls if needed
            if customer.drink_type in [DrinkType.MILK_TEA, DrinkType.FRUIT_TEA, DrinkType.SPECIALTY]:
                self.pearl_inventory -= 1
                
                # Check if reorder is needed
                if self.pearl_inventory <= self.pearl_reorder_point:
                    self.env.process(self.reorder_pearls())
            
            service_time = self.generate_service_time(customer.drink_type, ServiceStation.BARISTA)
            yield self.env.timeout(service_time)
        
        # 3. Sealer
        with self.sealer.request() as request:
            yield request
            
            # Record queue length
            self.queue_lengths[ServiceStation.SEALER].append(
                (self.env.now, len(self.sealer.queue))
            )
            
            service_time = self.generate_service_time(customer.drink_type, ServiceStation.SEALER)
            yield self.env.timeout(service_time)
        
        # Customer exits
        customer.exit_time = self.env.now
        customer.service_end_time = self.env.now
        
        # Record statistics
        self.customers_served += 1
        total_time = customer.exit_time - customer.arrival_time
        self.total_wait_time += total_time
        self.customer_data.append(customer)
    
    def wait_for_pearls(self):
        """Wait for pearl replenishment when inventory is empty"""
        with self.pearl_cooker.request() as request:
            yield request
            # Pearl cooking time (batch process)
            yield self.env.timeout(self.config['pearl_cook_time'])
            self.pearl_inventory += self.pearl_batch_size
    
    def reorder_pearls(self):
        """Reorder pearls when inventory reaches reorder point"""
        with self.pearl_cooker.request() as request:
            yield request
            # Pearl cooking time
            yield self.env.timeout(self.config['pearl_cook_time'])
            self.pearl_inventory += self.pearl_batch_size
    
    def run_simulation(self, simulation_time: float = 480):  # 8 hours default
        """Run the simulation for specified time"""
        # Start customer arrival process
        self.env.process(self.customer_arrival_process())
        
        # Run simulation
        self.env.run(until=simulation_time)
        
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """Calculate and return simulation statistics"""
        if not self.customer_data:
            return {"error": "No customers served"}
        
        # Calculate basic statistics
        total_customers = len(self.customer_data)
        avg_wait_time = np.mean([c.exit_time - c.arrival_time for c in self.customer_data])
        avg_service_time = np.mean([c.service_end_time - c.service_start_time 
                                  for c in self.customer_data if c.service_start_time])
        
        # Calculate percentiles
        wait_times = [c.exit_time - c.arrival_time for c in self.customer_data]
        p95_wait_time = np.percentile(wait_times, 95)
        
        # Calculate throughput
        throughput = total_customers / (self.env.now / 60)  # customers per hour
        
        # Calculate utilization
        utilization = {}
        for station in ServiceStation:
            if station == ServiceStation.CASHIER:
                resource = self.cashier
            elif station == ServiceStation.BARISTA:
                resource = self.barista
            else:
                resource = self.sealer
            
            # Simple utilization calculation (can be improved)
            utilization[station.value] = min(1.0, len(self.customer_data) / 
                                           (self.env.now * self.config[f'{station.value}_capacity']))
        
        return {
            "total_customers": total_customers,
            "avg_wait_time": avg_wait_time,
            "avg_service_time": avg_service_time,
            "p95_wait_time": p95_wait_time,
            "throughput": throughput,
            "utilization": utilization,
            "simulation_time": self.env.now,
            "customer_data": self.customer_data
        }

def create_baseline_config() -> Dict:
    """Create baseline configuration for the boba shop"""
    return {
        'cashier_capacity': 1,
        'barista_capacity': 2,
        'sealer_capacity': 1,
        'initial_pearl_inventory': 50,
        'pearl_reorder_point': 10,
        'pearl_batch_size': 30,
        'pearl_cook_time': 15,  # minutes
        'policy': 'baseline'
    }

def create_automation_config() -> Dict:
    """Create configuration with automation (auto-sealer)"""
    config = create_baseline_config()
    config['sealer_capacity'] = 2  # Auto-sealer increases capacity
    config['policy'] = 'automation'
    return config

def create_staffing_config() -> Dict:
    """Create configuration with additional staff"""
    config = create_baseline_config()
    config['barista_capacity'] = 3  # Additional barista
    config['policy'] = 'staffing'
    return config

def run_monte_carlo_simulation(config: Dict, num_replications: int = 100, 
                             simulation_time: float = 480) -> List[Dict]:
    """Run Monte Carlo simulation with multiple replications"""
    results = []
    
    for i in range(num_replications):
        # Create new simulator instance for each replication
        simulator = BobaShopSimulator(config)
        stats = simulator.run_simulation(simulation_time)
        stats['replication'] = i
        results.append(stats)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_replications} replications")
    
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze Monte Carlo simulation results"""
    df = pd.DataFrame(results)
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for metric in ['avg_wait_time', 'p95_wait_time', 'throughput']:
        if metric in df.columns:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            n = len(df)
            
            # 95% confidence interval
            ci_margin = 1.96 * std_val / np.sqrt(n)
            confidence_intervals[metric] = {
                'mean': mean_val,
                'lower': mean_val - ci_margin,
                'upper': mean_val + ci_margin,
                'std': std_val
            }
    
    return {
        'summary_stats': df.describe(),
        'confidence_intervals': confidence_intervals,
        'raw_data': df
    }

def validate_littles_law(results: List[Dict]) -> Dict:
    """Validate Little's Law: WIP = λ × W"""
    validation_results = []
    
    for result in results:
        if 'error' not in result:
            # Calculate WIP (Work in Progress) - average number of customers in system
            wip = result['total_customers'] / (result['simulation_time'] / 60)  # customers per hour
            
            # Calculate λ (arrival rate) - average arrival rate
            lambda_val = result['total_customers'] / (result['simulation_time'] / 60)
            
            # Calculate W (average wait time in hours)
            w = result['avg_wait_time'] / 60
            
            # Little's Law: WIP = λ × W
            predicted_wip = lambda_val * w
            actual_wip = wip
            
            validation_results.append({
                'lambda': lambda_val,
                'wait_time_hours': w,
                'predicted_wip': predicted_wip,
                'actual_wip': actual_wip,
                'error': abs(predicted_wip - actual_wip) / actual_wip if actual_wip > 0 else 0
            })
    
    return pd.DataFrame(validation_results)

def create_visualizations(results_dict: Dict[str, List[Dict]]):
    """Create visualizations for simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Wait Time Comparison
    ax1 = axes[0, 0]
    policies = list(results_dict.keys())
    wait_times = [np.mean([r['avg_wait_time'] for r in results_dict[policy] if 'error' not in r]) 
                  for policy in policies]
    wait_errors = [np.std([r['avg_wait_time'] for r in results_dict[policy] if 'error' not in r]) 
                   for policy in policies]
    
    ax1.bar(policies, wait_times, yerr=wait_errors, capsize=5, alpha=0.7)
    ax1.set_title('Average Wait Time by Policy')
    ax1.set_ylabel('Wait Time (minutes)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Throughput Comparison
    ax2 = axes[0, 1]
    throughputs = [np.mean([r['throughput'] for r in results_dict[policy] if 'error' not in r]) 
                   for policy in policies]
    throughput_errors = [np.std([r['throughput'] for r in results_dict[policy] if 'error' not in r]) 
                         for policy in policies]
    
    ax2.bar(policies, throughputs, yerr=throughput_errors, capsize=5, alpha=0.7, color='green')
    ax2.set_title('Throughput by Policy')
    ax2.set_ylabel('Customers per Hour')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 95th Percentile Wait Time
    ax3 = axes[1, 0]
    p95_times = [np.mean([r['p95_wait_time'] for r in results_dict[policy] if 'error' not in r]) 
                 for policy in policies]
    p95_errors = [np.std([r['p95_wait_time'] for r in results_dict[policy] if 'error' not in r]) 
                  for policy in policies]
    
    ax3.bar(policies, p95_times, yerr=p95_errors, capsize=5, alpha=0.7, color='red')
    ax3.set_title('95th Percentile Wait Time by Policy')
    ax3.set_ylabel('Wait Time (minutes)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Wait Time Distribution (Box Plot)
    ax4 = axes[1, 1]
    wait_data = []
    for policy in policies:
        policy_wait_times = [r['avg_wait_time'] for r in results_dict[policy] if 'error' not in r]
        wait_data.append(policy_wait_times)
    
    ax4.boxplot(wait_data, labels=policies)
    ax4.set_title('Wait Time Distribution by Policy')
    ax4.set_ylabel('Wait Time (minutes)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('boba_simulation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the boba shop simulation"""
    print("Boba Shop Operations Simulator")
    print("=" * 40)
    
    # Define configurations for different policies
    configs = {
        'baseline': create_baseline_config(),
        'automation': create_automation_config(),
        'staffing': create_staffing_config()
    }
    
    # Run simulations for each policy
    all_results = {}
    
    for policy_name, config in configs.items():
        print(f"\nRunning {policy_name} policy simulation...")
        results = run_monte_carlo_simulation(config, num_replications=50, simulation_time=480)
        all_results[policy_name] = results
        
        # Analyze results for this policy
        analysis = analyze_results(results)
        print(f"\n{policy_name.upper()} POLICY RESULTS:")
        print(f"Average Wait Time: {analysis['confidence_intervals']['avg_wait_time']['mean']:.2f} ± "
              f"{analysis['confidence_intervals']['avg_wait_time']['std']:.2f} minutes")
        print(f"95th Percentile Wait Time: {analysis['confidence_intervals']['p95_wait_time']['mean']:.2f} ± "
              f"{analysis['confidence_intervals']['p95_wait_time']['std']:.2f} minutes")
        print(f"Throughput: {analysis['confidence_intervals']['throughput']['mean']:.2f} ± "
              f"{analysis['confidence_intervals']['throughput']['std']:.2f} customers/hour")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(all_results)
    
    # Validate Little's Law
    print("\nValidating Little's Law...")
    for policy_name, results in all_results.items():
        validation = validate_littles_law(results)
        avg_error = validation['error'].mean()
        print(f"{policy_name}: Average Little's Law error: {avg_error:.3f}")
    
    print("\nSimulation completed! Check 'boba_simulation_results.png' for visualizations.")

if __name__ == "__main__":
    main()
