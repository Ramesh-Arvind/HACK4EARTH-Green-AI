#!/usr/bin/env python3
"""
Multi-Objective Optimization for Greenhouse Control
===================================================
Pareto optimization framework balancing competing objectives:
- Minimize operational cost (€)
- Maximize crop yield (kg/m²)
- Minimize water consumption (L/m²)
- Minimize carbon emissions (kg CO₂e)

Uses NSGA-II (Non-dominated Sorting Genetic Algorithm II) to find
Pareto-optimal control policies satisfying greenhouse constraints.

Author: EcoGrow Team
Date: October 15, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import copy

# Set style
sns.set_style("whitegrid")


@dataclass
class GreenhouseConstraints:
    """Physical and operational constraints for greenhouse."""
    
    # Temperature constraints (°C)
    tair_min: float = 15.0
    tair_max: float = 30.0
    tair_optimal_min: float = 18.0
    tair_optimal_max: float = 24.0
    
    # Humidity constraints (%)
    rhair_min: float = 40.0
    rhair_max: float = 100.0
    rhair_optimal_min: float = 65.0
    rhair_optimal_max: float = 85.0
    
    # CO₂ constraints (ppm)
    co2_min: float = 380.0
    co2_max: float = 1500.0
    co2_optimal_min: float = 400.0
    co2_optimal_max: float = 1200.0
    
    # Control constraints
    ventlee_min: float = 0.0
    ventlee_max: float = 100.0
    pipelow_min: float = 20.0
    pipelow_max: float = 90.0
    assimlight_min: float = 0.0
    assimlight_max: float = 100.0
    co2_dosing_min: float = 0.0
    co2_dosing_max: float = 200.0  # kg/ha/h
    
    # PAR constraint (μmol/m²/s)
    par_min: float = 0.0
    par_max: float = 1000.0
    
    def __post_init__(self):
        """Validate constraints."""
        assert self.tair_min < self.tair_optimal_min < self.tair_optimal_max < self.tair_max
        assert self.rhair_min < self.rhair_optimal_min < self.rhair_optimal_max < self.rhair_max
        assert self.co2_min < self.co2_optimal_min < self.co2_optimal_max < self.co2_max


@dataclass
class ControlPolicy:
    """Greenhouse control policy (decision variables)."""
    
    # Temperature control
    tair_setpoint_day: float = 21.5  # °C
    tair_setpoint_night: float = 20.0  # °C
    
    # Heating control
    pipelow_base: float = 50.0  # °C
    heating_p_gain: float = 5.0  # Proportional gain
    
    # Ventilation control
    vent_temp_threshold: float = 22.0  # °C
    vent_rh_threshold: float = 85.0  # %
    vent_co2_threshold: float = 1000.0  # ppm
    
    # Lighting control
    par_target: float = 200.0  # μmol/m²/s
    lighting_hours_start: int = 6  # Hour (0-23)
    lighting_hours_end: int = 22  # Hour (0-23)
    
    # CO₂ enrichment
    co2_setpoint: float = 800.0  # ppm
    co2_enrichment_enabled: bool = True
    
    # Screen control
    screen_close_hour: int = 22  # Hour (0-23)
    screen_open_hour: int = 4  # Hour (0-23)
    
    def to_array(self) -> np.ndarray:
        """Convert policy to array for optimization."""
        return np.array([
            self.tair_setpoint_day,
            self.tair_setpoint_night,
            self.pipelow_base,
            self.heating_p_gain,
            self.vent_temp_threshold,
            self.vent_rh_threshold,
            self.vent_co2_threshold,
            self.par_target,
            float(self.lighting_hours_start),
            float(self.lighting_hours_end),
            self.co2_setpoint,
            float(self.co2_enrichment_enabled),
            float(self.screen_close_hour),
            float(self.screen_open_hour)
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ControlPolicy':
        """Create policy from array."""
        return cls(
            tair_setpoint_day=arr[0],
            tair_setpoint_night=arr[1],
            pipelow_base=arr[2],
            heating_p_gain=arr[3],
            vent_temp_threshold=arr[4],
            vent_rh_threshold=arr[5],
            vent_co2_threshold=arr[6],
            par_target=arr[7],
            lighting_hours_start=int(arr[8]),
            lighting_hours_end=int(arr[9]),
            co2_setpoint=arr[10],
            co2_enrichment_enabled=bool(arr[11] > 0.5),
            screen_close_hour=int(arr[12]),
            screen_open_hour=int(arr[13])
        )


@dataclass
class ObjectiveValues:
    """Multi-objective function values."""
    
    cost_eur_per_m2: float  # Operational cost (minimize)
    yield_kg_per_m2: float  # Crop yield (maximize)
    water_l_per_m2: float  # Water consumption (minimize)
    carbon_kg_per_m2: float  # Carbon emissions (minimize)
    
    # Constraint violations (penalty terms)
    constraint_violation: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to array for optimization (all objectives to minimize)."""
        return np.array([
            self.cost_eur_per_m2,
            -self.yield_kg_per_m2,  # Negative because we maximize yield
            self.water_l_per_m2,
            self.carbon_kg_per_m2,
            self.constraint_violation
        ])
    
    def dominates(self, other: 'ObjectiveValues') -> bool:
        """Check if this solution dominates another (Pareto dominance)."""
        self_arr = self.to_array()
        other_arr = other.to_array()
        
        # At least one objective better, none worse
        better = np.any(self_arr < other_arr)
        not_worse = np.all(self_arr <= other_arr)
        
        return better and not_worse


class GreenhouseSimulator:
    """
    Simplified greenhouse simulator for optimization.
    Based on baseline controller physics but faster for optimization.
    """
    
    def __init__(self, constraints: GreenhouseConstraints):
        """
        Initialize simulator.
        
        Parameters:
        -----------
        constraints : GreenhouseConstraints
            Physical constraints
        """
        self.constraints = constraints
        
        # Economic parameters (from Economics.pdf and EDA)
        self.elec_price_peak = 0.08  # €/kWh
        self.elec_price_offpeak = 0.04  # €/kWh
        self.heat_price = 0.0083  # €/MJ
        self.co2_price_tier1 = 0.08  # €/kg (0-50 kg/ha)
        self.co2_price_tier2 = 0.20  # €/kg (>50 kg/ha)
        
        # Carbon factors (from carbon-aware scheduler)
        self.carbon_heat = 0.056  # kg CO₂/MJ
        self.carbon_elec = 0.42  # kg CO₂/kWh (Germany grid)
        
        # Crop model parameters (simplified tomato growth)
        self.base_yield = 60.0  # kg/m²/year baseline
        self.temp_optimal = 21.0  # °C
        self.co2_optimal = 800.0  # ppm
        self.par_optimal = 200.0  # μmol/m²/s
        
        # Water use parameters
        self.base_water = 1000.0  # L/m²/year baseline
        self.transpiration_rate = 0.15  # L/m²/day average
        
    def simulate_week(self, policy: ControlPolicy, 
                     weather_scenario: str = 'winter') -> ObjectiveValues:
        """
        Simulate one week of greenhouse operation with given policy.
        
        Parameters:
        -----------
        policy : ControlPolicy
            Control policy to evaluate
        weather_scenario : str
            Weather scenario ('winter', 'spring', 'summer', 'autumn')
            
        Returns:
        --------
        objectives : ObjectiveValues
            Multi-objective function values
        """
        # Weather parameters (simplified)
        weather_params = {
            'winter': {'tout_avg': 5.0, 'iglob_avg': 100.0, 'rhout_avg': 85.0},
            'spring': {'tout_avg': 12.0, 'iglob_avg': 250.0, 'rhout_avg': 70.0},
            'summer': {'tout_avg': 20.0, 'iglob_avg': 400.0, 'rhout_avg': 60.0},
            'autumn': {'tout_avg': 10.0, 'iglob_avg': 150.0, 'rhout_avg': 80.0}
        }
        weather = weather_params[weather_scenario]
        
        # Simulation parameters
        timesteps = 7 * 24 * 12  # 7 days, 5-min intervals
        dt = 5.0 / 60.0  # 5 minutes in hours
        
        # Initialize state
        tair = policy.tair_setpoint_day
        rhair = 75.0
        co2air = policy.co2_setpoint
        
        # Accumulators
        total_heating_mj = 0.0
        total_electricity_kwh = 0.0
        total_co2_kg = 0.0
        total_water_l = 0.0
        
        # Constraint violation accumulator
        constraint_penalty = 0.0
        
        # Optimal hours accumulator (for yield)
        optimal_hours = 0.0
        
        # Simulate timesteps
        for step in range(timesteps):
            hour = int(step * dt) % 24
            
            # Day/night
            is_day = 6 <= hour < 18
            
            # Setpoint
            tair_setpoint = policy.tair_setpoint_day if is_day else policy.tair_setpoint_night
            
            # Heating (proportional control)
            temp_error = max(0, tair_setpoint - tair)
            heating_mj = policy.heating_p_gain * temp_error * dt
            
            # Add outdoor temperature dependency (from EDA: r=-0.85)
            outdoor_factor = max(0, 1.0 - weather['tout_avg'] / 25.0)
            heating_mj *= (0.5 + 0.5 * outdoor_factor)
            
            total_heating_mj += heating_mj
            
            # Update temperature (simplified)
            heat_gain = heating_mj / 0.4  # Rough conversion
            heat_loss = 0.1 * (tair - weather['tout_avg']) * dt
            solar_gain = weather['iglob_avg'] * 0.5 * dt / 1000.0
            tair += heat_gain - heat_loss + solar_gain
            
            # Ventilation (cooling, dehumidification)
            vent_needed = 0.0
            if tair > policy.vent_temp_threshold:
                vent_needed = max(vent_needed, 50.0)
            if rhair > policy.vent_rh_threshold:
                vent_needed = max(vent_needed, 40.0)
            if co2air > policy.vent_co2_threshold:
                vent_needed = max(vent_needed, 30.0)
            
            if vent_needed > 0:
                tair -= 0.5 * dt
                rhair -= 2.0 * dt
                co2air -= 50.0 * dt
            
            # Lighting (supplemental)
            par_natural = weather['iglob_avg'] * 2.0  # Rough PAR estimate
            par_deficit = max(0, policy.par_target - par_natural)
            
            if policy.lighting_hours_start <= hour < policy.lighting_hours_end and par_deficit > 0:
                # HPS lighting: 81 W/m² for 180 μmol/m²/s
                lighting_kwh = 0.081 * (par_deficit / 180.0) * dt
                total_electricity_kwh += lighting_kwh
            
            # CO₂ enrichment
            if policy.co2_enrichment_enabled and is_day and co2air < policy.co2_setpoint:
                co2_needed = policy.co2_setpoint - co2air
                co2_injection_kg = min(co2_needed / 1000.0, 0.5) * dt  # Rough conversion
                total_co2_kg += co2_injection_kg
                co2air += co2_injection_kg * 1000.0
            
            # Natural CO₂ depletion (photosynthesis)
            if is_day and par_natural > 50:
                co2air -= 20.0 * dt
            
            # Humidity dynamics
            transpiration = self.transpiration_rate * dt / 24.0
            total_water_l += transpiration
            rhair += transpiration * 5.0  # Rough humidity increase
            
            # Clamp state to physical bounds
            tair = np.clip(tair, self.constraints.tair_min, self.constraints.tair_max)
            rhair = np.clip(rhair, self.constraints.rhair_min, self.constraints.rhair_max)
            co2air = np.clip(co2air, self.constraints.co2_min, self.constraints.co2_max)
            
            # Check constraints (accumulate violations)
            if not (self.constraints.tair_optimal_min <= tair <= self.constraints.tair_optimal_max):
                constraint_penalty += abs(tair - 21.0) * dt
            if not (self.constraints.rhair_optimal_min <= rhair <= self.constraints.rhair_optimal_max):
                constraint_penalty += abs(rhair - 75.0) * 0.1 * dt
            if not (self.constraints.co2_optimal_min <= co2air <= self.constraints.co2_optimal_max):
                constraint_penalty += abs(co2air - 800.0) * 0.01 * dt
            
            # Count optimal hours (for yield)
            if (self.constraints.tair_optimal_min <= tair <= self.constraints.tair_optimal_max and
                self.constraints.rhair_optimal_min <= rhair <= self.constraints.rhair_optimal_max and
                self.constraints.co2_optimal_min <= co2air <= self.constraints.co2_optimal_max):
                optimal_hours += dt
        
        # Calculate costs
        cost_heating = total_heating_mj * self.heat_price
        cost_electricity = total_electricity_kwh * (0.6 * self.elec_price_peak + 0.4 * self.elec_price_offpeak)
        
        # CO₂ tiered pricing
        co2_ha = total_co2_kg * 16.0  # Convert from greenhouse (62.5m²) to ha (10000m²)
        if co2_ha <= 50:
            cost_co2 = total_co2_kg * self.co2_price_tier1
        else:
            cost_co2 = 50 / 16.0 * self.co2_price_tier1 + (total_co2_kg - 50 / 16.0) * self.co2_price_tier2
        
        total_cost = cost_heating + cost_electricity + cost_co2
        
        # Calculate carbon emissions
        carbon_heating = total_heating_mj * self.carbon_heat
        carbon_electricity = total_electricity_kwh * self.carbon_elec
        carbon_co2_injection = total_co2_kg  # CO₂ injection is carbon emission
        total_carbon = carbon_heating + carbon_electricity + carbon_co2_injection
        
        # Calculate yield (simplified growth model)
        # Yield depends on optimal hours ratio
        optimal_ratio = optimal_hours / (7 * 24)  # Fraction of week in optimal conditions
        
        # Yield factors
        temp_factor = np.exp(-((tair - self.temp_optimal) / 5.0)**2)  # Gaussian
        co2_factor = min(1.0, co2air / self.co2_optimal)
        par_factor = min(1.0, policy.par_target / self.par_optimal)
        
        # Weekly yield (rough estimate)
        weekly_yield = self.base_yield / 52.0 * optimal_ratio * temp_factor * co2_factor * par_factor
        
        # Weekly water
        weekly_water = total_water_l
        
        return ObjectiveValues(
            cost_eur_per_m2=total_cost,
            yield_kg_per_m2=weekly_yield,
            water_l_per_m2=weekly_water,
            carbon_kg_per_m2=total_carbon,
            constraint_violation=constraint_penalty
        )


class NSGAII:
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II).
    Multi-objective evolutionary algorithm for finding Pareto optimal solutions.
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 num_generations: int = 50,
                 crossover_prob: float = 0.9,
                 mutation_prob: float = 0.1,
                 mutation_strength: float = 0.2):
        """
        Initialize NSGA-II.
        
        Parameters:
        -----------
        population_size : int
            Population size (should be even)
        num_generations : int
            Number of generations to evolve
        crossover_prob : float
            Crossover probability
        mutation_prob : float
            Mutation probability
        mutation_strength : float
            Mutation strength (fraction of range)
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_strength = mutation_strength
        
        # Decision variable bounds (control policy parameters)
        self.bounds = np.array([
            [18.0, 24.0],   # tair_setpoint_day
            [16.0, 22.0],   # tair_setpoint_night
            [30.0, 70.0],   # pipelow_base
            [1.0, 10.0],    # heating_p_gain
            [20.0, 26.0],   # vent_temp_threshold
            [75.0, 95.0],   # vent_rh_threshold
            [800.0, 1200.0],  # vent_co2_threshold
            [150.0, 250.0],   # par_target
            [4.0, 8.0],     # lighting_hours_start
            [20.0, 23.0],   # lighting_hours_end
            [600.0, 1000.0],  # co2_setpoint
            [0.0, 1.0],     # co2_enrichment_enabled
            [20.0, 23.0],   # screen_close_hour
            [3.0, 6.0]      # screen_open_hour
        ])
        
        self.num_variables = len(self.bounds)
        
    def initialize_population(self) -> np.ndarray:
        """
        Initialize random population within bounds.
        
        Returns:
        --------
        population : np.ndarray [population_size, num_variables]
            Initial population
        """
        population = np.random.rand(self.population_size, self.num_variables)
        
        # Scale to bounds
        for i in range(self.num_variables):
            population[:, i] = (population[:, i] * (self.bounds[i, 1] - self.bounds[i, 0]) + 
                               self.bounds[i, 0])
        
        return population
    
    def evaluate_population(self, population: np.ndarray, 
                           simulator: GreenhouseSimulator,
                           weather_scenario: str = 'winter') -> np.ndarray:
        """
        Evaluate objectives for entire population.
        
        Parameters:
        -----------
        population : np.ndarray [population_size, num_variables]
            Population to evaluate
        simulator : GreenhouseSimulator
            Greenhouse simulator
        weather_scenario : str
            Weather scenario
            
        Returns:
        --------
        objectives : np.ndarray [population_size, num_objectives]
            Objective values (all minimization)
        """
        objectives = []
        
        for individual in population:
            policy = ControlPolicy.from_array(individual)
            obj = simulator.simulate_week(policy, weather_scenario)
            objectives.append(obj.to_array())
        
        return np.array(objectives)
    
    def fast_non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """
        Fast non-dominated sorting (NSGA-II).
        
        Parameters:
        -----------
        objectives : np.ndarray [population_size, num_objectives]
            Objective values
            
        Returns:
        --------
        fronts : List[List[int]]
            Pareto fronts (list of lists of indices)
        """
        population_size = len(objectives)
        
        # Domination sets and counts
        domination_sets = [[] for _ in range(population_size)]
        domination_counts = np.zeros(population_size, dtype=int)
        
        # First front
        fronts = [[]]
        
        # Find domination relationships
        for i in range(population_size):
            for j in range(i + 1, population_size):
                # Check if i dominates j
                i_dominates_j = (np.all(objectives[i] <= objectives[j]) and 
                                np.any(objectives[i] < objectives[j]))
                
                # Check if j dominates i
                j_dominates_i = (np.all(objectives[j] <= objectives[i]) and 
                                np.any(objectives[j] < objectives[i]))
                
                if i_dominates_j:
                    domination_sets[i].append(j)
                    domination_counts[j] += 1
                elif j_dominates_i:
                    domination_sets[j].append(i)
                    domination_counts[i] += 1
            
            # If not dominated, add to first front
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        current_front = 0
        while current_front < len(fronts) and fronts[current_front]:
            next_front = []
            
            for i in fronts[current_front]:
                for j in domination_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        # Remove empty fronts
        return [f for f in fronts if f]
    
    def calculate_crowding_distance(self, objectives: np.ndarray, front: List[int]) -> np.ndarray:
        """
        Calculate crowding distance for a front.
        
        Parameters:
        -----------
        objectives : np.ndarray [population_size, num_objectives]
            Objective values
        front : List[int]
            Indices of individuals in front
            
        Returns:
        --------
        distances : np.ndarray [len(front)]
            Crowding distances
        """
        num_individuals = len(front)
        num_objectives = objectives.shape[1]
        distances = np.zeros(num_individuals)
        
        if num_individuals <= 2:
            return np.full(num_individuals, np.inf)
        
        # For each objective
        for m in range(num_objectives):
            # Sort by objective
            sorted_indices = np.argsort(objectives[front, m])
            
            # Boundary points have infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Normalize by objective range
            obj_range = objectives[front[sorted_indices[-1]], m] - objectives[front[sorted_indices[0]], m]
            if obj_range == 0:
                continue
            
            # Crowding distance for interior points
            for i in range(1, num_individuals - 1):
                distances[sorted_indices[i]] += (
                    (objectives[front[sorted_indices[i + 1]], m] - 
                     objectives[front[sorted_indices[i - 1]], m]) / obj_range
                )
        
        return distances
    
    def tournament_selection(self, population: np.ndarray, 
                            fronts: List[List[int]], 
                            crowding_distances: Dict[int, float],
                            tournament_size: int = 2) -> np.ndarray:
        """
        Tournament selection based on rank and crowding distance.
        
        Parameters:
        -----------
        population : np.ndarray
            Current population
        fronts : List[List[int]]
            Pareto fronts
        crowding_distances : Dict[int, float]
            Crowding distances for all individuals
        tournament_size : int
            Tournament size
            
        Returns:
        --------
        selected : np.ndarray
            Selected individual
        """
        # Create rank map
        ranks = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
        
        # Random tournament
        candidates = np.random.choice(self.population_size, tournament_size, replace=False)
        
        # Select best by rank, then crowding distance
        best = candidates[0]
        for candidate in candidates[1:]:
            if ranks[candidate] < ranks[best]:
                best = candidate
            elif ranks[candidate] == ranks[best]:
                if crowding_distances[candidate] > crowding_distances[best]:
                    best = candidate
        
        return population[best].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulated binary crossover (SBX).
        
        Parameters:
        -----------
        parent1, parent2 : np.ndarray
            Parent individuals
            
        Returns:
        --------
        offspring1, offspring2 : np.ndarray
            Offspring individuals
        """
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        offspring1 = np.zeros_like(parent1)
        offspring2 = np.zeros_like(parent2)
        
        eta_c = 20.0  # Distribution index for crossover
        
        for i in range(self.num_variables):
            if abs(parent1[i] - parent2[i]) < 1e-9:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
                continue
            
            # SBX
            u = np.random.rand()
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta_c + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
            
            offspring1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            offspring2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
            
            # Clamp to bounds
            offspring1[i] = np.clip(offspring1[i], self.bounds[i, 0], self.bounds[i, 1])
            offspring2[i] = np.clip(offspring2[i], self.bounds[i, 0], self.bounds[i, 1])
        
        return offspring1, offspring2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Polynomial mutation.
        
        Parameters:
        -----------
        individual : np.ndarray
            Individual to mutate
            
        Returns:
        --------
        mutated : np.ndarray
            Mutated individual
        """
        mutated = individual.copy()
        eta_m = 20.0  # Distribution index for mutation
        
        for i in range(self.num_variables):
            if np.random.rand() < self.mutation_prob:
                u = np.random.rand()
                
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                
                mutated[i] += delta * (self.bounds[i, 1] - self.bounds[i, 0]) * self.mutation_strength
                mutated[i] = np.clip(mutated[i], self.bounds[i, 0], self.bounds[i, 1])
        
        return mutated
    
    def optimize(self, simulator: GreenhouseSimulator, 
                weather_scenario: str = 'winter',
                verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """
        Run NSGA-II optimization.
        
        Parameters:
        -----------
        simulator : GreenhouseSimulator
            Greenhouse simulator
        weather_scenario : str
            Weather scenario
        verbose : bool
            Print progress
            
        Returns:
        --------
        pareto_population : np.ndarray
            Pareto-optimal decision variables
        pareto_objectives : np.ndarray
            Pareto-optimal objective values
        history_fronts : List[List[int]]
            History of first fronts (for visualization)
        """
        # Initialize
        population = self.initialize_population()
        history_fronts = []
        
        if verbose:
            print(f"🧬 NSGA-II Optimization")
            print(f"   Population: {self.population_size}")
            print(f"   Generations: {self.num_generations}")
            print(f"   Weather: {weather_scenario}")
            print()
        
        # Evolution
        for gen in range(self.num_generations):
            # Evaluate objectives
            objectives = self.evaluate_population(population, simulator, weather_scenario)
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(objectives)
            
            # Save first front
            history_fronts.append(fronts[0])
            
            # Calculate crowding distances
            crowding_distances = {}
            for front in fronts:
                distances = self.calculate_crowding_distance(objectives, front)
                for idx, dist in zip(front, distances):
                    crowding_distances[idx] = dist
            
            if verbose and (gen % 10 == 0 or gen == self.num_generations - 1):
                print(f"   Gen {gen:3d}: {len(fronts[0])} solutions in Pareto front")
            
            # Create offspring
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = self.tournament_selection(population, fronts, crowding_distances)
                parent2 = self.tournament_selection(population, fronts, crowding_distances)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                offspring.append(child1)
                if len(offspring) < self.population_size:
                    offspring.append(child2)
            
            offspring = np.array(offspring)
            
            # Combine parent and offspring
            combined_population = np.vstack([population, offspring])
            combined_objectives = np.vstack([
                objectives,
                self.evaluate_population(offspring, simulator, weather_scenario)
            ])
            
            # Select next generation
            combined_fronts = self.fast_non_dominated_sort(combined_objectives)
            next_population = []
            next_indices = []
            
            for front in combined_fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front)
                    next_indices.extend(front)
                else:
                    # Need to select partial front based on crowding distance
                    remaining = self.population_size - len(next_population)
                    distances = self.calculate_crowding_distance(combined_objectives, front)
                    sorted_front = [front[i] for i in np.argsort(distances)[::-1]]
                    next_population.extend(sorted_front[:remaining])
                    next_indices.extend(sorted_front[:remaining])
                    break
            
            population = combined_population[next_indices]
        
        # Final evaluation
        objectives = self.evaluate_population(population, simulator, weather_scenario)
        fronts = self.fast_non_dominated_sort(objectives)
        
        # Extract Pareto front
        pareto_indices = fronts[0]
        pareto_population = population[pareto_indices]
        pareto_objectives = objectives[pareto_indices]
        
        if verbose:
            print()
            print(f"✅ Optimization complete!")
            print(f"   Pareto front: {len(pareto_indices)} solutions")
            print()
        
        return pareto_population, pareto_objectives, history_fronts


class ParetoAnalyzer:
    """Analyze and visualize Pareto-optimal solutions."""
    
    def __init__(self, pareto_population: np.ndarray, pareto_objectives: np.ndarray):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        pareto_population : np.ndarray
            Pareto-optimal decision variables
        pareto_objectives : np.ndarray
            Pareto-optimal objective values
        """
        self.pareto_population = pareto_population
        self.pareto_objectives = pareto_objectives
        
        # Objective names
        self.objective_names = ['Cost (€/m²)', 'Yield (kg/m²)', 'Water (L/m²)', 
                               'Carbon (kg CO₂e/m²)', 'Constraint Violation']
        
    def create_summary(self) -> pd.DataFrame:
        """
        Create summary DataFrame of Pareto solutions.
        
        Returns:
        --------
        df : pd.DataFrame
            Summary table
        """
        # Convert objectives (negate yield back to positive)
        objectives_display = self.pareto_objectives.copy()
        objectives_display[:, 1] = -objectives_display[:, 1]
        
        # Create dataframe
        df = pd.DataFrame(objectives_display, columns=self.objective_names)
        
        # Add solution index
        df.insert(0, 'Solution', range(len(df)))
        
        return df
    
    def find_extreme_solutions(self) -> Dict[str, int]:
        """
        Find extreme solutions (best for each objective).
        
        Returns:
        --------
        extremes : Dict[str, int]
            Dictionary mapping objective name to solution index
        """
        extremes = {}
        
        # Cost (minimize)
        extremes['Min Cost'] = np.argmin(self.pareto_objectives[:, 0])
        
        # Yield (maximize, stored as negative)
        extremes['Max Yield'] = np.argmin(self.pareto_objectives[:, 1])
        
        # Water (minimize)
        extremes['Min Water'] = np.argmin(self.pareto_objectives[:, 2])
        
        # Carbon (minimize)
        extremes['Min Carbon'] = np.argmin(self.pareto_objectives[:, 3])
        
        return extremes
    
    def find_balanced_solution(self) -> int:
        """
        Find balanced solution (closest to ideal point).
        
        Returns:
        --------
        balanced_idx : int
            Index of balanced solution
        """
        # Normalize objectives to [0, 1]
        obj_min = self.pareto_objectives.min(axis=0)
        obj_max = self.pareto_objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1  # Avoid division by zero
        
        objectives_normalized = (self.pareto_objectives - obj_min) / obj_range
        
        # Ideal point (all zeros for minimization)
        ideal = np.zeros(objectives_normalized.shape[1])
        
        # Find closest solution (Euclidean distance)
        distances = np.linalg.norm(objectives_normalized - ideal, axis=1)
        balanced_idx = np.argmin(distances)
        
        return balanced_idx
    
    def visualize_pareto_front(self, save_path: Optional[str] = None):
        """
        Visualize Pareto front (multi-panel scatter plots).
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save figure
        """
        # Create figure with 6 subplots (4 objective pairs + policy + summary)
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Convert objectives for display (negate yield)
        objectives_display = self.pareto_objectives.copy()
        objectives_display[:, 1] = -objectives_display[:, 1]
        
        # Color by constraint violation
        colors = objectives_display[:, 4]
        
        # Plot 1: Cost vs Yield
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(objectives_display[:, 0], objectives_display[:, 1], 
                              c=colors, cmap='RdYlGn_r', s=50, alpha=0.7)
        ax1.set_xlabel('Cost (€/m²)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Yield (kg/m²)', fontsize=11, fontweight='bold')
        ax1.set_title('Cost vs Yield Trade-off', fontsize=12, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Cost vs Carbon
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(objectives_display[:, 0], objectives_display[:, 3], 
                   c=colors, cmap='RdYlGn_r', s=50, alpha=0.7)
        ax2.set_xlabel('Cost (€/m²)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Carbon (kg CO₂e/m²)', fontsize=11, fontweight='bold')
        ax2.set_title('Cost vs Carbon Trade-off', fontsize=12, fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Carbon vs Yield
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(objectives_display[:, 3], objectives_display[:, 1], 
                   c=colors, cmap='RdYlGn_r', s=50, alpha=0.7)
        ax3.set_xlabel('Carbon (kg CO₂e/m²)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Yield (kg/m²)', fontsize=11, fontweight='bold')
        ax3.set_title('Carbon vs Yield Trade-off', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Water vs Carbon
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(objectives_display[:, 2], objectives_display[:, 3], 
                   c=colors, cmap='RdYlGn_r', s=50, alpha=0.7)
        ax4.set_xlabel('Water (L/m²)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Carbon (kg CO₂e/m²)', fontsize=11, fontweight='bold')
        ax4.set_title('Water vs Carbon Trade-off', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Plot 5: Cost vs Water
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.scatter(objectives_display[:, 0], objectives_display[:, 2], 
                   c=colors, cmap='RdYlGn_r', s=50, alpha=0.7)
        ax5.set_xlabel('Cost (€/m²)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Water (L/m²)', fontsize=11, fontweight='bold')
        ax5.set_title('Cost vs Water Trade-off', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)
        
        # Plot 6: Yield vs Water
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(objectives_display[:, 1], objectives_display[:, 2], 
                   c=colors, cmap='RdYlGn_r', s=50, alpha=0.7)
        ax6.set_xlabel('Yield (kg/m²)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Water (L/m²)', fontsize=11, fontweight='bold')
        ax6.set_title('Yield vs Water Trade-off', fontsize=12, fontweight='bold')
        ax6.grid(alpha=0.3)
        
        # Plot 7: Parallel coordinates (normalized objectives)
        ax7 = fig.add_subplot(gs[2, :])
        
        # Normalize objectives
        obj_min = objectives_display[:, :4].min(axis=0)
        obj_max = objectives_display[:, :4].max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1
        objectives_normalized = (objectives_display[:, :4] - obj_min) / obj_range
        
        # Plot lines
        for i in range(len(objectives_normalized)):
            ax7.plot(range(4), objectives_normalized[i], alpha=0.3, color='blue')
        
        ax7.set_xticks(range(4))
        ax7.set_xticklabels(['Cost', 'Yield', 'Water', 'Carbon'], fontsize=11, fontweight='bold')
        ax7.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
        ax7.set_title('Pareto Solutions (Parallel Coordinates)', fontsize=12, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter1, ax=[ax1, ax2, ax3, ax4, ax5, ax6], 
                           location='right', shrink=0.8, pad=0.02)
        cbar.set_label('Constraint Violation', fontsize=11, fontweight='bold')
        
        plt.suptitle('Multi-Objective Pareto Front Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Pareto front visualization saved: {save_path}")
        
        plt.show()


def main():
    """Demonstration of multi-objective optimization."""
    print("="*70)
    print("🎯 MULTI-OBJECTIVE GREENHOUSE OPTIMIZATION")
    print("="*70)
    print()
    
    # Initialize components
    constraints = GreenhouseConstraints()
    simulator = GreenhouseSimulator(constraints)
    
    print("📋 Optimization Setup:")
    print(f"   Objectives: 4 (cost, yield, water, carbon)")
    print(f"   Constraints: Temperature {constraints.tair_optimal_min}-{constraints.tair_optimal_max}°C, "
          f"RH {constraints.rhair_optimal_min}-{constraints.rhair_optimal_max}%, "
          f"CO₂ {constraints.co2_optimal_min}-{constraints.co2_optimal_max} ppm")
    print(f"   Decision variables: 14 (control policy parameters)")
    print()
    
    # Run NSGA-II
    optimizer = NSGAII(population_size=100, num_generations=50)
    pareto_population, pareto_objectives, history = optimizer.optimize(
        simulator, weather_scenario='winter', verbose=True
    )
    
    # Analyze results
    analyzer = ParetoAnalyzer(pareto_population, pareto_objectives)
    
    # Create summary
    summary_df = analyzer.create_summary()
    print("📊 Pareto Front Summary (first 10 solutions):")
    print(summary_df.head(10).to_string(index=False))
    print(f"   ... ({len(summary_df)} solutions total)")
    print()
    
    # Find extreme and balanced solutions
    extremes = analyzer.find_extreme_solutions()
    balanced_idx = analyzer.find_balanced_solution()
    
    print("🏆 Notable Solutions:")
    print()
    
    for name, idx in extremes.items():
        obj = pareto_objectives[idx]
        print(f"   {name} (Solution {idx}):")
        print(f"      Cost: €{obj[0]:.3f}/m², Yield: {-obj[1]:.3f} kg/m², "
              f"Water: {obj[2]:.2f} L/m², Carbon: {obj[3]:.3f} kg CO₂e/m²")
    
    print()
    print(f"   Balanced Solution (Solution {balanced_idx}):")
    obj_bal = pareto_objectives[balanced_idx]
    print(f"      Cost: €{obj_bal[0]:.3f}/m², Yield: {-obj_bal[1]:.3f} kg/m², "
          f"Water: {obj_bal[2]:.2f} L/m², Carbon: {obj_bal[3]:.3f} kg CO₂e/m²")
    print()
    
    # Save results
    summary_df.to_csv('../results/pareto_solutions.csv', index=False)
    print("✅ Pareto solutions saved: ../results/pareto_solutions.csv")
    
    # Save control policies
    policies_data = []
    for i, individual in enumerate(pareto_population):
        policy = ControlPolicy.from_array(individual)
        policy_dict = {
            'solution_id': i,
            'tair_setpoint_day': policy.tair_setpoint_day,
            'tair_setpoint_night': policy.tair_setpoint_night,
            'pipelow_base': policy.pipelow_base,
            'par_target': policy.par_target,
            'co2_setpoint': policy.co2_setpoint
        }
        policies_data.append(policy_dict)
    
    policies_df = pd.DataFrame(policies_data)
    policies_df.to_csv('../results/pareto_control_policies.csv', index=False)
    print("✅ Control policies saved: ../results/pareto_control_policies.csv")
    
    # Visualize
    analyzer.visualize_pareto_front(save_path='../results/pareto_front_analysis.png')
    
    print()
    print("="*70)
    print("✅ MULTI-OBJECTIVE OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
