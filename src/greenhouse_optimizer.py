"""
Greenhouse Energy Optimizer for HACK4EARTH Green AI Challenge
Track B: Use AI to reduce greenhouse energy consumption

Optimizes HVAC, lighting, and CO₂ injection to minimize:
1. Energy consumption (kWh)
2. Operating costs (€)
3. Carbon emissions (kg CO₂e)

While maintaining optimal growing conditions and crop yield targets.
"""

import numpy as np
import pandas as pd
import torch
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pica_framework"))

try:
    from src.model_hybrid import EnhancedGNNWithPINN
    from src.multi_objective_optimizer import MultiObjectiveOptimizer
except ImportError:
    print("Warning: Could not import model components. Some features may be limited.")
    EnhancedGNNWithPINN = None
    MultiObjectiveOptimizer = None


class GreenhouseEnergyOptimizer:
    """
    Track B: Use AI to reduce greenhouse energy consumption
    
    Optimizes control actions to minimize energy while maintaining crop health
    """
    
    def __init__(self, model_path='../pica_framework/models/trained_models/trained_model.pth'):
        """
        Initialize optimizer
        
        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        if EnhancedGNNWithPINN is not None:
            self.model = EnhancedGNNWithPINN(num_nodes=8, hidden_dim=64)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Loaded model from {model_path}")
            else:
                print(f"⚠️  Model not found, using random initialization")
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None
            print("⚠️  Running in mock mode")
        
        # Initialize multi-objective optimizer if available
        if MultiObjectiveOptimizer is not None and self.model is not None:
            self.optimizer = MultiObjectiveOptimizer(self.model)
        else:
            self.optimizer = None
        
        # Energy costs (Germany 2025)
        self.electricity_cost = 0.32  # €/kWh
        self.gas_cost = 0.08  # €/kWh
        
        # Carbon intensity
        self.grid_carbon = 0.420  # kg CO₂/kWh (Germany average)
        self.gas_carbon = 0.252   # kg CO₂/kWh (natural gas)
    
    def calculate_baseline_energy(self, control_sequence):
        """
        Calculate energy consumption for traditional control
        
        Args:
            control_sequence: Array of control actions [uV, uCO2, uQh, uQc]
            
        Returns:
            Dictionary with energy metrics
        """
        # Convert to numpy if tensor
        if isinstance(control_sequence, torch.Tensor):
            control_sequence = control_sequence.detach().cpu().numpy()
        
        # Traditional rule-based control energy consumption
        total_heating = np.sum(np.maximum(control_sequence[:, 2], 0))  # uQh
        total_cooling = np.sum(np.maximum(control_sequence[:, 3], 0))  # uQc
        total_ventilation = np.sum(np.maximum(control_sequence[:, 0], 0))  # uV
        total_co2 = np.sum(np.maximum(control_sequence[:, 1], 0))  # uCO2
        
        # Energy consumption (kWh per 5-min timestep)
        # Typical greenhouse: 100 m², 1000 m³ volume
        heating_kwh = total_heating * 0.5  # 0.5 kWh per timestep when heating
        cooling_kwh = total_cooling * 0.3  # 0.3 kWh per timestep when cooling
        ventilation_kwh = total_ventilation * 0.1  # 0.1 kWh per timestep
        co2_kwh = total_co2 * 0.05  # 0.05 kWh per timestep for CO2 injection
        
        total_kwh = heating_kwh + cooling_kwh + ventilation_kwh + co2_kwh
        
        return {
            'total_kwh': total_kwh,
            'heating_kwh': heating_kwh,
            'cooling_kwh': cooling_kwh,
            'ventilation_kwh': ventilation_kwh,
            'co2_kwh': co2_kwh,
            'cost_eur': total_kwh * self.electricity_cost,
            'carbon_kg': total_kwh * self.grid_carbon
        }
    
    def optimize_controls(self, initial_state, target_state, horizon=288):
        """
        Optimize control sequence for energy efficiency
        
        Args:
            initial_state: Current greenhouse state [T, H, CO2, B]
            target_state: Desired state
            horizon: Planning horizon (288 = 24 hours @ 5-min intervals)
            
        Returns:
            Dictionary with optimized controls and metrics
        """
        if self.optimizer is None:
            print("⚠️  Optimizer not available, using simple heuristic")
            return self._simple_optimization(initial_state, target_state, horizon)
        
        # Define objectives
        def energy_objective(states, controls):
            return self.compute_energy_cost(controls)
        
        def comfort_objective(states, controls):
            return self.compute_comfort_penalty(states, target_state)
        
        def yield_objective(states, controls):
            # Maximize biomass (minimize negative biomass)
            return -torch.mean(states[:, 3])
        
        objectives = {
            'energy': energy_objective,
            'comfort': comfort_objective,
            'yield': yield_objective
        }
        
        # Define constraints
        constraints = {
            'temperature': (18.0, 25.0),  # °C
            'humidity': (60.0, 85.0),      # %
            'co2': (400.0, 1200.0)         # ppm
        }
        
        # Run multi-objective optimization
        try:
            pareto_solutions = self.optimizer.optimize(
                initial_state=initial_state,
                objectives=objectives,
                constraints=constraints,
                horizon=horizon,
                n_solutions=50
            )
            
            # Select solution with best energy efficiency while meeting constraints
            best_solution = self.select_efficient_solution(pareto_solutions)
            return best_solution
        except Exception as e:
            print(f"⚠️  Optimization failed: {e}, using fallback")
            return self._simple_optimization(initial_state, target_state, horizon)
    
    def _simple_optimization(self, initial_state, target_state, horizon):
        """Simple rule-based optimization as fallback"""
        # Generate simple control sequence
        controls = torch.zeros(horizon, 4)
        
        # Simple proportional control
        for t in range(horizon):
            temp_error = target_state[0] - initial_state[0]
            humid_error = target_state[1] - initial_state[1]
            co2_error = target_state[2] - initial_state[2]
            
            # Heating/cooling based on temperature error
            if temp_error > 0:
                controls[t, 2] = min(temp_error * 0.1, 1.0)  # Heating
            else:
                controls[t, 3] = min(-temp_error * 0.1, 1.0)  # Cooling
            
            # Ventilation based on humidity
            if humid_error < 0:
                controls[t, 0] = min(-humid_error * 0.01, 1.0)
            
            # CO2 injection
            if co2_error > 0:
                controls[t, 1] = min(co2_error * 0.001, 1.0)
        
        return {
            'controls': controls,
            'states': None,
            'energy_cost': self.compute_energy_cost(controls).item()
        }
    
    def compute_energy_cost(self, controls):
        """Compute total energy consumption"""
        if isinstance(controls, np.ndarray):
            controls = torch.from_numpy(controls).float()
        
        heating = torch.sum(torch.clamp(controls[:, 2], min=0)) * 0.5
        cooling = torch.sum(torch.clamp(controls[:, 3], min=0)) * 0.3
        ventilation = torch.sum(torch.clamp(controls[:, 0], min=0)) * 0.1
        co2 = torch.sum(torch.clamp(controls[:, 1], min=0)) * 0.05
        
        return heating + cooling + ventilation + co2
    
    def compute_comfort_penalty(self, states, target_state):
        """Penalize deviations from optimal growing conditions"""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        if isinstance(target_state, np.ndarray):
            target_state = torch.from_numpy(target_state).float()
        
        temp_penalty = torch.mean((states[:, 0] - target_state[0])**2)
        humid_penalty = torch.mean((states[:, 1] - target_state[1])**2)
        co2_penalty = torch.mean((states[:, 2] - target_state[2])**2)
        
        return temp_penalty + humid_penalty + co2_penalty
    
    def select_efficient_solution(self, pareto_solutions):
        """Select most energy-efficient solution from Pareto front"""
        # Find solution with minimum energy cost
        min_energy = float('inf')
        best_solution = None
        
        for solution in pareto_solutions:
            if solution['energy_cost'] < min_energy:
                min_energy = solution['energy_cost']
                best_solution = solution
        
        return best_solution if best_solution else pareto_solutions[0]
    
    def calculate_annual_impact(self, daily_savings_kwh, num_greenhouses=1):
        """
        Calculate scaled impact for Track B submission
        
        Args:
            daily_savings_kwh: Energy saved per day per greenhouse
            num_greenhouses: Number of greenhouses in deployment scenario
            
        Returns:
            Dictionary with annual impact metrics
        """
        # Annual calculations
        days_per_year = 365
        annual_savings_kwh = daily_savings_kwh * days_per_year * num_greenhouses
        
        # Carbon impact
        annual_carbon_saved_kg = annual_savings_kwh * self.grid_carbon
        annual_carbon_saved_tons = annual_carbon_saved_kg / 1000
        
        # Cost savings
        annual_cost_savings = annual_savings_kwh * self.electricity_cost
        
        # Water savings (indirect: reduced cooling = less water evaporation)
        # Typical: 2L water per kWh cooling energy
        cooling_fraction = 0.3  # 30% of energy goes to cooling
        annual_water_saved_m3 = (annual_savings_kwh * cooling_fraction * 2) / 1000
        
        # Environmental equivalents
        cars_removed = annual_carbon_saved_tons / 4.6  # Avg car emits 4.6 tons/year
        trees_planted = annual_carbon_saved_tons * 45  # 1 tree absorbs ~22kg CO2/year
        
        return {
            'annual_energy_saved_kwh': annual_savings_kwh,
            'annual_carbon_saved_tons': annual_carbon_saved_tons,
            'annual_cost_savings_eur': annual_cost_savings,
            'annual_water_saved_m3': annual_water_saved_m3,
            'num_greenhouses': num_greenhouses,
            'cars_equivalent': cars_removed,
            'trees_equivalent': int(trees_planted)
        }
    
    def generate_impact_scenarios(self, daily_savings_kwh):
        """Generate low/medium/high impact scenarios for Track B"""
        scenarios = {
            'low': self.calculate_annual_impact(daily_savings_kwh, num_greenhouses=10),
            'medium': self.calculate_annual_impact(daily_savings_kwh, num_greenhouses=100),
            'high': self.calculate_annual_impact(daily_savings_kwh, num_greenhouses=1000)
        }
        return scenarios


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Greenhouse energy optimization for Track B')
    parser.add_argument('--model_path', type=str,
                       default='../pica_framework/models/trained_models/trained_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str,
                       default='../pica_framework/data/filtered_dates.csv',
                       help='Path to greenhouse data')
    parser.add_argument('--horizon', type=int, default=288,
                       help='Planning horizon (timesteps)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GREENHOUSE ENERGY OPTIMIZER - TRACK B")
    print("="*70)
    
    optimizer = GreenhouseEnergyOptimizer(model_path=args.model_path)
    
    # Load data
    if os.path.exists(args.data_path):
        data = pd.read_csv(args.data_path)
        print(f"✅ Loaded data from {args.data_path}")
        
        # Get 24 hours of control data (288 timesteps @ 5 min intervals)
        # Use actual column names from the dataset
        control_cols = ['Vent_ref', 'CO2_inj_ref', 'heat_ref', 'cool_ref']
        baseline_controls = data[control_cols].values[:args.horizon]
    else:
        print("⚠️  Data not found, using synthetic controls")
        # Generate synthetic baseline controls
        baseline_controls = np.random.uniform(0, 0.5, (args.horizon, 4))
    
    # Calculate baseline performance
    print("\n" + "-"*70)
    print("📊 BASELINE PERFORMANCE (Traditional Rule-Based Control)")
    print("-"*70)
    
    baseline = optimizer.calculate_baseline_energy(baseline_controls)
    
    print(f"Energy consumption:")
    print(f"  Total: {baseline['total_kwh']:.2f} kWh/day")
    print(f"  - Heating: {baseline['heating_kwh']:.2f} kWh")
    print(f"  - Cooling: {baseline['cooling_kwh']:.2f} kWh")
    print(f"  - Ventilation: {baseline['ventilation_kwh']:.2f} kWh")
    print(f"  - CO₂ injection: {baseline['co2_kwh']:.2f} kWh")
    print(f"\nOperating costs: €{baseline['cost_eur']:.2f}/day")
    print(f"Carbon footprint: {baseline['carbon_kg']:.2f} kg CO₂e/day")
    
    # Optimize controls
    print("\n" + "-"*70)
    print("🌱 OPTIMIZED PERFORMANCE (AI-Based Control)")
    print("-"*70)
    
    initial_state = torch.tensor([20.0, 75.0, 600.0, 10.0])
    target_state = torch.tensor([22.0, 75.0, 800.0, 11.0])
    
    optimized = optimizer.optimize_controls(initial_state, target_state, horizon=args.horizon)
    optimized_energy = optimizer.calculate_baseline_energy(optimized['controls'])
    
    print(f"Energy consumption:")
    print(f"  Total: {optimized_energy['total_kwh']:.2f} kWh/day")
    print(f"  - Heating: {optimized_energy['heating_kwh']:.2f} kWh")
    print(f"  - Cooling: {optimized_energy['cooling_kwh']:.2f} kWh")
    print(f"  - Ventilation: {optimized_energy['ventilation_kwh']:.2f} kWh")
    print(f"  - CO₂ injection: {optimized_energy['co2_kwh']:.2f} kWh")
    print(f"\nOperating costs: €{optimized_energy['cost_eur']:.2f}/day")
    print(f"Carbon footprint: {optimized_energy['carbon_kg']:.2f} kg CO₂e/day")
    
    # Calculate savings
    print("\n" + "-"*70)
    print("✅ DAILY SAVINGS")
    print("-"*70)
    
    daily_savings_kwh = baseline['total_kwh'] - optimized_energy['total_kwh']
    savings_percent = (daily_savings_kwh / baseline['total_kwh']) * 100
    cost_savings = baseline['cost_eur'] - optimized_energy['cost_eur']
    carbon_savings = baseline['carbon_kg'] - optimized_energy['carbon_kg']
    
    print(f"Energy: {daily_savings_kwh:.2f} kWh/day ({savings_percent:.1f}% reduction)")
    print(f"Cost: €{cost_savings:.2f}/day")
    print(f"Carbon: {carbon_savings:.2f} kg CO₂e/day")
    
    # Annual impact scenarios
    print("\n" + "="*70)
    print("🌍 ANNUAL IMPACT SCENARIOS")
    print("="*70)
    
    scenarios = optimizer.generate_impact_scenarios(daily_savings_kwh)
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n{scenario_name.upper()} SCENARIO ({scenario['num_greenhouses']} greenhouses):")
        print(f"  Energy saved: {scenario['annual_energy_saved_kwh']:,.0f} kWh/year")
        print(f"  Carbon avoided: {scenario['annual_carbon_saved_tons']:.1f} tons CO₂e/year")
        print(f"  Cost savings: €{scenario['annual_cost_savings_eur']:,.0f}/year")
        print(f"  Water saved: {scenario['annual_water_saved_m3']:.0f} m³/year")
        print(f"  Equivalent to:")
        print(f"    🚗 Removing {scenario['cars_equivalent']:.0f} cars from roads")
        print(f"    🌳 Planting {scenario['trees_equivalent']:,} trees")
    
    # Save results
    output_path = '../ecogrow/results/greenhouse_optimization_results.csv'
    results_df = pd.DataFrame([{
        'scenario': 'baseline',
        'energy_kwh_day': baseline['total_kwh'],
        'cost_eur_day': baseline['cost_eur'],
        'carbon_kg_day': baseline['carbon_kg']
    }, {
        'scenario': 'optimized',
        'energy_kwh_day': optimized_energy['total_kwh'],
        'cost_eur_day': optimized_energy['cost_eur'],
        'carbon_kg_day': optimized_energy['carbon_kg']
    }])
    
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
