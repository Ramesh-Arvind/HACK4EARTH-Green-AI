#!/usr/bin/env python3
"""
Baseline Rule-Based Greenhouse Controller
==========================================
Implements historical control strategy based on EDA findings from Phase 2.
Simulates greenhouse operation for 1 week or 1 month with simple rule-based control.

Author: EcoGrow Team
Date: October 15, 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class BaselineController:
    """
    Rule-based greenhouse controller using historical control strategies.
    Based on EDA findings from Reference Group performance.
    """
    
    def __init__(self, start_date='2020-01-01', season='winter'):
        """
        Initialize baseline controller with season-dependent parameters.
        
        Parameters:
        -----------
        start_date : str
            Start date for simulation (format: 'YYYY-MM-DD')
        season : str
            Season for control parameters ('winter', 'spring', 'summer', 'autumn')
        """
        self.start_date = pd.to_datetime(start_date)
        self.season = season
        self.timestep = 5  # minutes (matching dataset resolution)
        
        # Control setpoints from EDA findings
        self.setpoints = self._get_seasonal_setpoints()
        
        # Physical parameters (from Phase 1)
        self.greenhouse_area = 62.5  # m² growing area
        self.cover_transmissivity = 0.5
        self.screen_transmissivity = 0.75
        
        # Energy parameters
        self.hps_lamp_power = 81  # W/m²
        self.heating_efficiency = 0.95  # boiler efficiency
        
        # Economic parameters (from Economics.pdf)
        self.elec_price_peak = 0.08  # €/kWh (07:00-23:00)
        self.elec_price_offpeak = 0.04  # €/kWh (23:00-07:00)
        self.heat_price = 0.0083  # €/MJ
        self.co2_price_tier1 = 0.08  # €/kg (0-12 kg/m²)
        self.co2_price_tier2 = 0.20  # €/kg (>12 kg/m²)
        
        # Carbon emission factors
        self.heat_carbon = 0.056  # kg CO₂/MJ (natural gas)
        self.elec_carbon = 0.42  # kg CO₂/kWh (Germany grid 2020)
        
        # History tracking
        self.history = {
            'timestamp': [],
            'Tair': [], 'Rhair': [], 'CO2air': [], 'Tot_PAR': [],
            'Tout': [], 'Rhout': [], 'Iglob': [], 'PARout': [],
            'PipeLow': [], 'PipeGrow': [], 'VentLee': [], 'Ventwind': [],
            'AssimLight': [], 'EnScr': [], 'BlackScr': [], 'co2_dos': [],
            'Heat_cons': [], 'ElecHigh': [], 'ElecLow': [], 'CO2_cons': [],
            'cost_heating': [], 'cost_elec': [], 'cost_co2': [], 'cost_total': [],
            'carbon_heating': [], 'carbon_elec': [], 'carbon_co2': [], 'carbon_total': []
        }
        
        # Initialize state
        self.current_state = self._initialize_state()
        
    def _get_seasonal_setpoints(self):
        """
        Get season-specific control setpoints based on EDA findings.
        
        Returns:
        --------
        dict : Control setpoints for temperature, humidity, CO2, etc.
        """
        # Base setpoints from EDA (winter/spring average)
        base = {
            'Tair_day': 21.5,      # °C (daytime, from EDA mean 21.25)
            'Tair_night': 20.0,    # °C (nighttime, energy saving)
            'Rhair_target': 74.0,  # % (from EDA mean 73.74)
            'CO2_day': 800.0,      # ppm (enrichment during photosynthesis)
            'CO2_night': 450.0,    # ppm (minimal dosing)
            'PAR_target': 180.0,   # µmol/m²/s (supplemental lighting target)
            'VentLee_max': 100.0,  # % (maximum ventilation)
            'PipeLow_base': 46.0,  # °C (from EDA mean 46.15)
            'PipeGrow_base': 5.0,  # °C (from EDA mean 5.76)
        }
        
        # Season-specific adjustments
        adjustments = {
            'winter': {
                'Tair_day': 21.5,
                'Tair_night': 20.0,
                'PAR_target': 200.0,  # Higher supplemental lighting
                'PipeLow_base': 50.0,  # More heating
            },
            'spring': {
                'Tair_day': 22.0,
                'Tair_night': 20.5,
                'PAR_target': 150.0,  # Less supplemental lighting
                'PipeLow_base': 42.0,  # Less heating
            },
            'summer': {
                'Tair_day': 23.0,
                'Tair_night': 21.0,
                'PAR_target': 100.0,  # Minimal supplemental lighting
                'PipeLow_base': 35.0,  # Minimal heating
            },
            'autumn': {
                'Tair_day': 22.0,
                'Tair_night': 20.0,
                'PAR_target': 180.0,
                'PipeLow_base': 45.0,
            }
        }
        
        setpoints = base.copy()
        setpoints.update(adjustments.get(self.season, {}))
        return setpoints
    
    def _initialize_state(self):
        """Initialize greenhouse state variables."""
        return {
            'Tair': self.setpoints['Tair_day'],
            'Rhair': self.setpoints['Rhair_target'],
            'CO2air': self.setpoints['CO2_night'],
            'Tot_PAR': 0.0,
            'cumulative_heat': 0.0,
            'cumulative_elec': 0.0,
            'cumulative_co2': 0.0,
            'cumulative_cost': 0.0,
            'cumulative_carbon': 0.0,
        }
    
    def generate_weather(self, duration_days=7):
        """
        Generate synthetic weather data based on season.
        
        Parameters:
        -----------
        duration_days : int
            Number of days to simulate
            
        Returns:
        --------
        pd.DataFrame : Weather data with 5-minute resolution
        """
        n_steps = int(duration_days * 24 * 60 / self.timestep)
        timestamps = [self.start_date + timedelta(minutes=i*self.timestep) for i in range(n_steps)]
        
        # Seasonal weather parameters (based on EDA)
        weather_params = {
            'winter': {'Tout_mean': 5.0, 'Tout_std': 3.0, 'Iglob_max': 300, 'cloud_prob': 0.7},
            'spring': {'Tout_mean': 12.0, 'Tout_std': 4.0, 'Iglob_max': 600, 'cloud_prob': 0.5},
            'summer': {'Tout_mean': 18.0, 'Tout_std': 3.5, 'Iglob_max': 800, 'cloud_prob': 0.3},
            'autumn': {'Tout_mean': 10.0, 'Tout_std': 3.5, 'Iglob_max': 400, 'cloud_prob': 0.6},
        }
        
        params = weather_params[self.season]
        
        weather_data = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60.0
            
            # Temperature with diurnal cycle
            Tout_diurnal = 2.0 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 18:00
            Tout = params['Tout_mean'] + Tout_diurnal + np.random.normal(0, 1.0)
            
            # Solar radiation (sunrise ~6:00, sunset ~18:00)
            if 6 <= hour <= 18:
                # Parabolic profile with cloud cover
                sun_elevation = np.sin(np.pi * (hour - 6) / 12)
                clear_sky_rad = params['Iglob_max'] * sun_elevation
                cloud_factor = 0.3 if np.random.random() < params['cloud_prob'] else 1.0
                Iglob = clear_sky_rad * cloud_factor + np.random.normal(0, 20)
            else:
                Iglob = 0.0
            
            Iglob = max(0, Iglob)
            
            # Outdoor humidity (higher at night)
            Rhout = 85.0 - 10.0 * sun_elevation if 6 <= hour <= 18 else 90.0
            Rhout += np.random.normal(0, 3.0)
            Rhout = np.clip(Rhout, 40, 100)
            
            # PAR from solar radiation (0.45 conversion factor)
            PARout = Iglob * 4.6 if Iglob > 0 else 0  # W/m² to µmol/m²/s
            
            # Wind speed
            Windsp = max(0, np.random.gamma(2, 1.5))  # Gamma distribution, mean ~3 m/s
            
            weather_data.append({
                'timestamp': ts,
                'Tout': Tout,
                'Rhout': Rhout,
                'Iglob': Iglob,
                'PARout': PARout,
                'Windsp': Windsp
            })
        
        return pd.DataFrame(weather_data)
    
    def control_heating(self, Tair, Tout, hour):
        """
        Rule-based heating control.
        
        Parameters:
        -----------
        Tair : float
            Current indoor temperature (°C)
        Tout : float
            Outdoor temperature (°C)
        hour : float
            Hour of day (0-24)
            
        Returns:
        --------
        tuple : (PipeLow temp, PipeGrow temp, heat consumption MJ/m²/day)
        """
        # Target temperature (day/night)
        if 8 <= hour <= 20:
            T_target = self.setpoints['Tair_day']
        else:
            T_target = self.setpoints['Tair_night']
        
        T_error = T_target - Tair
        
        # Proportional control with outdoor compensation
        # Based on EDA: Heat_cons = -0.16 * Tout + 2.81
        base_heat = max(0, -0.16 * Tout + 2.81)  # MJ/m²/day
        
        # Adjust based on temperature error
        if T_error > 2.0:
            PipeLow = self.setpoints['PipeLow_base'] + 10.0
            PipeGrow = self.setpoints['PipeGrow_base'] + 5.0
            heat_factor = 1.5
        elif T_error > 0.5:
            PipeLow = self.setpoints['PipeLow_base'] + 5.0
            PipeGrow = self.setpoints['PipeGrow_base'] + 2.0
            heat_factor = 1.2
        elif T_error > -0.5:
            PipeLow = self.setpoints['PipeLow_base']
            PipeGrow = self.setpoints['PipeGrow_base']
            heat_factor = 1.0
        else:
            PipeLow = self.setpoints['PipeLow_base'] - 5.0
            PipeGrow = 0.0
            heat_factor = 0.7
        
        # Clip pipe temperatures
        PipeLow = np.clip(PipeLow, 35, 70)
        PipeGrow = np.clip(PipeGrow, 0, 60)
        
        # Heat consumption (MJ/m²/day) -> convert to per timestep
        heat_cons = base_heat * heat_factor * (self.timestep / (24 * 60))
        
        return PipeLow, PipeGrow, heat_cons
    
    def control_ventilation(self, Tair, Rhair, CO2air, Tout, hour):
        """
        Rule-based ventilation control.
        
        Parameters:
        -----------
        Tair : float
            Indoor temperature (°C)
        Rhair : float
            Indoor humidity (%)
        CO2air : float
            Indoor CO₂ (ppm)
        Tout : float
            Outdoor temperature (°C)
        hour : float
            Hour of day
            
        Returns:
        --------
        tuple : (VentLee %, Ventwind %)
        """
        VentLee = 0.0
        Ventwind = 0.0
        
        # Temperature-based ventilation (cooling)
        if 8 <= hour <= 20:
            T_target = self.setpoints['Tair_day']
        else:
            T_target = self.setpoints['Tair_night']
        
        T_error = Tair - T_target
        
        if T_error > 3.0:
            VentLee = 80.0
        elif T_error > 1.5:
            VentLee = 40.0
        elif T_error > 0.5:
            VentLee = 10.0
        
        # Humidity-based ventilation (dehumidification)
        if Rhair > 85.0:
            VentLee = max(VentLee, 60.0)
        elif Rhair > 80.0:
            VentLee = max(VentLee, 30.0)
        
        # CO₂ conservation (close vents during enrichment)
        if 8 <= hour <= 18 and CO2air > 700:
            VentLee = min(VentLee, 5.0)  # Keep mostly closed
        
        # Don't ventilate when outdoor is colder (winter)
        if Tout < 10.0 and T_error < 1.0:
            VentLee = min(VentLee, 10.0)
        
        # Wind-side ventilation (similar to lee-side)
        Ventwind = VentLee * 0.8  # Slightly less than lee-side
        
        return VentLee, Ventwind
    
    def control_lighting(self, Tot_PAR, PARout, hour):
        """
        Rule-based supplemental lighting control.
        
        Parameters:
        -----------
        Tot_PAR : float
            Current total PAR (µmol/m²/s)
        PARout : float
            Outdoor PAR (µmol/m²/s)
        hour : float
            Hour of day
            
        Returns:
        --------
        tuple : (AssimLight %, electricity consumption kWh/m²/day)
        """
        AssimLight = 0.0
        
        # Only operate during photoperiod (6:00-22:00)
        if 6 <= hour <= 22:
            PAR_deficit = self.setpoints['PAR_target'] - PARout
            
            if PAR_deficit > 150:
                AssimLight = 100.0  # Full power
            elif PAR_deficit > 75:
                AssimLight = 75.0
            elif PAR_deficit > 30:
                AssimLight = 50.0
            elif PAR_deficit > 10:
                AssimLight = 25.0
        
        # Electricity consumption (HPS lamps: 81 W/m²)
        # Convert to kWh/m²/day for timestep
        elec_cons = (AssimLight / 100.0) * self.hps_lamp_power * (self.timestep / 60) / 1000  # kWh/m²
        
        return AssimLight, elec_cons
    
    def control_co2(self, CO2air, Tot_PAR, VentLee, hour):
        """
        Rule-based CO₂ enrichment control.
        
        Parameters:
        -----------
        CO2air : float
            Current CO₂ concentration (ppm)
        Tot_PAR : float
            Current PAR (µmol/m²/s)
        VentLee : float
            Ventilation opening (%)
        hour : float
            Hour of day
            
        Returns:
        --------
        tuple : (co2_dos kg/ha/hour, CO2 consumption kg/m²/day)
        """
        co2_dos = 0.0
        
        # Only dose during daytime with photosynthesis
        if 8 <= hour <= 18 and Tot_PAR > 50:
            # Target CO₂ (from EDA: 800 ppm during day)
            CO2_target = self.setpoints['CO2_day']
            CO2_deficit = CO2_target - CO2air
            
            # Only dose if deficit and vents mostly closed
            if CO2_deficit > 100 and VentLee < 20:
                # Dosing rate proportional to deficit
                co2_dos = 150.0  # kg/ha/hour (typical rate)
            elif CO2_deficit > 50 and VentLee < 10:
                co2_dos = 80.0
            elif CO2_deficit > 20 and VentLee < 5:
                co2_dos = 40.0
        
        # Convert to kg/m²/day for timestep
        # 1 ha = 10,000 m², so kg/ha/hour -> kg/m²/day
        co2_cons = co2_dos * (self.timestep / 60) / 10000  # kg/m²
        
        return co2_dos, co2_cons
    
    def control_screens(self, hour, Iglob, Tair):
        """
        Rule-based screen control (energy screen and blackout screen).
        
        Parameters:
        -----------
        hour : float
            Hour of day
        Iglob : float
            Solar radiation (W/m²)
        Tair : float
            Indoor temperature (°C)
            
        Returns:
        --------
        tuple : (EnScr %, BlackScr %)
        """
        EnScr = 0.0
        BlackScr = 0.0
        
        # Energy screen (nighttime insulation)
        if hour < 6 or hour > 20:
            EnScr = 80.0  # Mostly closed at night for insulation
        elif Iglob < 50:
            EnScr = 50.0  # Partially closed on cloudy days
        
        # Blackout screen (photoperiod control)
        if hour < 4 or hour > 22:
            BlackScr = 100.0  # Full darkness at night
        
        return EnScr, BlackScr
    
    def update_state(self, weather, controls, heat_cons, elec_cons, co2_cons):
        """
        Update greenhouse state based on weather and controls.
        Simple energy balance model.
        
        Parameters:
        -----------
        weather : dict
            Weather conditions
        controls : dict
            Control actions
        heat_cons : float
            Heating consumption (MJ/m²)
        elec_cons : float
            Electricity consumption (kWh/m²)
        co2_cons : float
            CO₂ consumption (kg/m²)
        """
        # Simple energy balance for temperature
        # Based on EDA: Tair = 0.23 * PipeLow + 10.8
        T_from_heating = 0.23 * controls['PipeLow'] + 10.8
        T_from_solar = weather['Iglob'] * 0.005  # Solar gain
        T_loss_vent = controls['VentLee'] * 0.05 * (self.current_state['Tair'] - weather['Tout'])
        
        self.current_state['Tair'] = 0.7 * self.current_state['Tair'] + 0.3 * (
            T_from_heating + T_from_solar - T_loss_vent
        )
        
        # Humidity balance
        ventilation_factor = 1.0 - controls['VentLee'] / 100.0
        self.current_state['Rhair'] = (
            0.8 * self.current_state['Rhair'] +
            0.2 * (weather['Rhout'] * (1 - ventilation_factor) + 
                   self.setpoints['Rhair_target'] * ventilation_factor)
        )
        
        # CO₂ balance
        co2_injection = co2_cons * 1e6 / (44 * 2.5)  # Convert kg/m² to ppm increment
        co2_ventilation_loss = controls['VentLee'] / 100.0 * (self.current_state['CO2air'] - 400)
        co2_photosynthesis = min(50, controls['Tot_PAR'] * 0.2)  # PAR-dependent uptake
        
        self.current_state['CO2air'] = self.current_state['CO2air'] + co2_injection - co2_ventilation_loss - co2_photosynthesis
        self.current_state['CO2air'] = np.clip(self.current_state['CO2air'], 380, 1500)
        
        # Total PAR (outdoor + artificial)
        par_from_lamps = (controls['AssimLight'] / 100.0) * 200  # HPS contributes ~200 µmol/m²/s at 100%
        self.current_state['Tot_PAR'] = weather['PARout'] * self.cover_transmissivity + par_from_lamps
        
        # Update cumulative metrics
        self.current_state['cumulative_heat'] += heat_cons
        self.current_state['cumulative_elec'] += elec_cons
        self.current_state['cumulative_co2'] += co2_cons
    
    def calculate_economics(self, heat_cons, elec_cons, co2_cons, hour):
        """
        Calculate costs and carbon emissions.
        
        Parameters:
        -----------
        heat_cons : float
            Heating consumption (MJ/m²)
        elec_cons : float
            Electricity consumption (kWh/m²)
        co2_cons : float
            CO₂ consumption (kg/m²)
        hour : float
            Hour of day (for peak/off-peak pricing)
            
        Returns:
        --------
        dict : Costs and emissions breakdown
        """
        # Costs
        cost_heating = heat_cons * self.heat_price
        
        # Peak/off-peak electricity pricing
        if 7 <= hour <= 23:
            cost_elec = elec_cons * self.elec_price_peak
        else:
            cost_elec = elec_cons * self.elec_price_offpeak
        
        # CO₂ tiered pricing
        if self.current_state['cumulative_co2'] < 12.0:
            cost_co2 = co2_cons * self.co2_price_tier1
        else:
            cost_co2 = co2_cons * self.co2_price_tier2
        
        cost_total = cost_heating + cost_elec + cost_co2
        
        # Carbon emissions
        carbon_heating = heat_cons * self.heat_carbon
        carbon_elec = elec_cons * self.elec_carbon
        carbon_co2 = co2_cons  # Direct release
        carbon_total = carbon_heating + carbon_elec + carbon_co2
        
        return {
            'cost_heating': cost_heating,
            'cost_elec': cost_elec,
            'cost_co2': cost_co2,
            'cost_total': cost_total,
            'carbon_heating': carbon_heating,
            'carbon_elec': carbon_elec,
            'carbon_co2': carbon_co2,
            'carbon_total': carbon_total
        }
    
    def simulate(self, duration_days=7, verbose=True):
        """
        Run baseline simulation for specified duration.
        
        Parameters:
        -----------
        duration_days : int
            Number of days to simulate
        verbose : bool
            Print progress messages
            
        Returns:
        --------
        pd.DataFrame : Complete simulation history
        """
        if verbose:
            print(f"🌱 Starting Baseline Simulation")
            print(f"   Season: {self.season}")
            print(f"   Duration: {duration_days} days")
            print(f"   Start: {self.start_date}")
            print(f"   Timestep: {self.timestep} minutes")
            print()
        
        # Generate weather data
        weather_df = self.generate_weather(duration_days)
        
        # Simulation loop
        for idx, weather_row in weather_df.iterrows():
            timestamp = weather_row['timestamp']
            hour = timestamp.hour + timestamp.minute / 60.0
            
            # Get weather conditions
            weather = {
                'Tout': weather_row['Tout'],
                'Rhout': weather_row['Rhout'],
                'Iglob': weather_row['Iglob'],
                'PARout': weather_row['PARout'],
                'Windsp': weather_row['Windsp']
            }
            
            # Apply control rules
            PipeLow, PipeGrow, heat_cons = self.control_heating(
                self.current_state['Tair'], weather['Tout'], hour
            )
            
            VentLee, Ventwind = self.control_ventilation(
                self.current_state['Tair'], self.current_state['Rhair'],
                self.current_state['CO2air'], weather['Tout'], hour
            )
            
            AssimLight, elec_cons = self.control_lighting(
                self.current_state['Tot_PAR'], weather['PARout'], hour
            )
            
            co2_dos, co2_cons = self.control_co2(
                self.current_state['CO2air'], self.current_state['Tot_PAR'],
                VentLee, hour
            )
            
            EnScr, BlackScr = self.control_screens(hour, weather['Iglob'], self.current_state['Tair'])
            
            # Package controls
            controls = {
                'PipeLow': PipeLow, 'PipeGrow': PipeGrow,
                'VentLee': VentLee, 'Ventwind': Ventwind,
                'AssimLight': AssimLight, 'EnScr': EnScr,
                'BlackScr': BlackScr, 'co2_dos': co2_dos,
                'Tot_PAR': self.current_state['Tot_PAR']
            }
            
            # Update state
            self.update_state(weather, controls, heat_cons, elec_cons, co2_cons)
            
            # Calculate economics
            economics = self.calculate_economics(heat_cons, elec_cons, co2_cons, hour)
            
            # Update cumulative costs and emissions
            self.current_state['cumulative_cost'] += economics['cost_total']
            self.current_state['cumulative_carbon'] += economics['carbon_total']
            
            # Record history
            self.history['timestamp'].append(timestamp)
            self.history['Tair'].append(self.current_state['Tair'])
            self.history['Rhair'].append(self.current_state['Rhair'])
            self.history['CO2air'].append(self.current_state['CO2air'])
            self.history['Tot_PAR'].append(self.current_state['Tot_PAR'])
            
            self.history['Tout'].append(weather['Tout'])
            self.history['Rhout'].append(weather['Rhout'])
            self.history['Iglob'].append(weather['Iglob'])
            self.history['PARout'].append(weather['PARout'])
            
            self.history['PipeLow'].append(PipeLow)
            self.history['PipeGrow'].append(PipeGrow)
            self.history['VentLee'].append(VentLee)
            self.history['Ventwind'].append(Ventwind)
            self.history['AssimLight'].append(AssimLight)
            self.history['EnScr'].append(EnScr)
            self.history['BlackScr'].append(BlackScr)
            self.history['co2_dos'].append(co2_dos)
            
            self.history['Heat_cons'].append(heat_cons)
            self.history['ElecHigh'].append(elec_cons if 7 <= hour <= 23 else 0)
            self.history['ElecLow'].append(elec_cons if hour < 7 or hour > 23 else 0)
            self.history['CO2_cons'].append(co2_cons)
            
            self.history['cost_heating'].append(economics['cost_heating'])
            self.history['cost_elec'].append(economics['cost_elec'])
            self.history['cost_co2'].append(economics['cost_co2'])
            self.history['cost_total'].append(economics['cost_total'])
            
            self.history['carbon_heating'].append(economics['carbon_heating'])
            self.history['carbon_elec'].append(economics['carbon_elec'])
            self.history['carbon_co2'].append(economics['carbon_co2'])
            self.history['carbon_total'].append(economics['carbon_total'])
            
            # Progress update
            if verbose and idx % 2880 == 0:  # Every day (288 steps/day * 10)
                day = idx // 288 + 1
                print(f"   Day {day}/{duration_days}: "
                      f"Tair={self.current_state['Tair']:.1f}°C, "
                      f"Cost=€{self.current_state['cumulative_cost']:.2f}/m², "
                      f"Carbon={self.current_state['cumulative_carbon']:.2f} kg CO₂e/m²")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.history)
        
        if verbose:
            print()
            self.print_summary(df, duration_days)
        
        return df
    
    def print_summary(self, df, duration_days):
        """Print simulation summary statistics."""
        print("="*70)
        print("📊 BASELINE SIMULATION SUMMARY")
        print("="*70)
        
        # Aggregate daily values
        df_daily = df.resample('D', on='timestamp').mean()
        
        # Resource consumption
        total_heat = df['Heat_cons'].sum()
        total_elec = (df['ElecHigh'].sum() + df['ElecLow'].sum())
        total_co2 = df['CO2_cons'].sum()
        
        print(f"\n🔥 Resource Consumption ({duration_days} days):")
        print(f"   Heating:      {total_heat:.2f} MJ/m² ({total_heat/duration_days:.2f} MJ/m²/day)")
        print(f"   Electricity:  {total_elec:.2f} kWh/m² ({total_elec/duration_days:.2f} kWh/m²/day)")
        print(f"   CO₂:          {total_co2:.4f} kg/m² ({total_co2/duration_days:.4f} kg/m²/day)")
        
        # Economics
        total_cost = df['cost_total'].sum()
        cost_breakdown = {
            'Heating': df['cost_heating'].sum(),
            'Electricity': df['cost_elec'].sum(),
            'CO₂': df['cost_co2'].sum()
        }
        
        print(f"\n💰 Economics:")
        print(f"   Total Cost:   €{total_cost:.2f}/m² (€{total_cost/duration_days:.2f}/m²/day)")
        for component, cost in cost_breakdown.items():
            pct = 100 * cost / total_cost
            print(f"   - {component:12s} €{cost:.2f} ({pct:.1f}%)")
        
        # Carbon footprint
        total_carbon = df['carbon_total'].sum()
        carbon_breakdown = {
            'Heating': df['carbon_heating'].sum(),
            'Electricity': df['carbon_elec'].sum(),
            'CO₂ injection': df['carbon_co2'].sum()
        }
        
        print(f"\n🌱 Carbon Footprint:")
        print(f"   Total:        {total_carbon:.2f} kg CO₂e/m² ({total_carbon/duration_days:.2f} kg CO₂e/m²/day)")
        for component, carbon in carbon_breakdown.items():
            pct = 100 * carbon / total_carbon
            print(f"   - {component:12s} {carbon:.2f} kg CO₂e ({pct:.1f}%)")
        
        # Climate control performance
        print(f"\n🌡️ Climate Control Performance:")
        print(f"   Temperature:  {df['Tair'].mean():.1f} ± {df['Tair'].std():.1f}°C "
              f"(range: {df['Tair'].min():.1f}-{df['Tair'].max():.1f}°C)")
        print(f"   Humidity:     {df['Rhair'].mean():.1f} ± {df['Rhair'].std():.1f}% "
              f"(range: {df['Rhair'].min():.1f}-{df['Rhair'].max():.1f}%)")
        print(f"   CO₂:          {df['CO2air'].mean():.0f} ± {df['CO2air'].std():.0f} ppm "
              f"(range: {df['CO2air'].min():.0f}-{df['CO2air'].max():.0f} ppm)")
        print(f"   PAR:          {df['Tot_PAR'].mean():.1f} ± {df['Tot_PAR'].std():.1f} µmol/m²/s "
              f"(range: {df['Tot_PAR'].min():.1f}-{df['Tot_PAR'].max():.1f})")
        
        print("="*70)
    
    def plot_results(self, df, save_path=None):
        """
        Create comprehensive visualization of simulation results.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Simulation history
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(5, 1, figsize=(16, 20))
        
        # 1. Climate variables
        ax = axes[0]
        ax.plot(df['timestamp'], df['Tair'], label='Tair (indoor)', color='red', linewidth=1.5)
        ax.plot(df['timestamp'], df['Tout'], label='Tout (outdoor)', color='blue', alpha=0.6)
        ax.axhline(self.setpoints['Tair_day'], color='red', linestyle='--', alpha=0.3, label='Target (day)')
        ax.axhline(self.setpoints['Tair_night'], color='darkred', linestyle='--', alpha=0.3, label='Target (night)')
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title('Climate Control Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Humidity and CO₂
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        ax2.plot(df['timestamp'], df['Rhair'], label='RH (indoor)', color='cyan', linewidth=1.5)
        ax2_twin.plot(df['timestamp'], df['CO2air'], label='CO₂', color='green', linewidth=1.5)
        ax2.set_ylabel('Relative Humidity (%)', fontsize=12, color='cyan')
        ax2_twin.set_ylabel('CO₂ (ppm)', fontsize=12, color='green')
        ax2.tick_params(axis='y', labelcolor='cyan')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2.set_title('Humidity & CO₂ Control', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Lighting (PAR)
        ax = axes[2]
        ax.fill_between(df['timestamp'], 0, df['PARout'], alpha=0.3, color='yellow', label='Solar PAR')
        ax.plot(df['timestamp'], df['Tot_PAR'], label='Total PAR', color='orange', linewidth=1.5)
        ax.axhline(self.setpoints['PAR_target'], color='orange', linestyle='--', alpha=0.5, label='Target PAR')
        ax.set_ylabel('PAR (µmol/m²/s)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title('Light Management', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Energy consumption
        ax = axes[3]
        # Convert to daily values
        df_daily = df.resample('D', on='timestamp').sum()
        ax.plot(df_daily.index, df_daily['Heat_cons'], label='Heating (MJ/m²/day)', 
                color='red', marker='o', linewidth=2)
        ax_twin = ax.twinx()
        ax_twin.plot(df_daily.index, df_daily['ElecHigh'] + df_daily['ElecLow'], 
                     label='Electricity (kWh/m²/day)', color='blue', marker='s', linewidth=2)
        ax.set_ylabel('Heating (MJ/m²/day)', fontsize=12, color='red')
        ax_twin.set_ylabel('Electricity (kWh/m²/day)', fontsize=12, color='blue')
        ax.tick_params(axis='y', labelcolor='red')
        ax_twin.tick_params(axis='y', labelcolor='blue')
        ax.set_title('Daily Energy Consumption', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 5. Economics and carbon
        ax = axes[4]
        cumulative_cost = df['cost_total'].cumsum()
        cumulative_carbon = df['carbon_total'].cumsum()
        ax.plot(df['timestamp'], cumulative_cost, label='Cost (€/m²)', color='green', linewidth=2)
        ax_twin = ax.twinx()
        ax_twin.plot(df['timestamp'], cumulative_carbon, label='Carbon (kg CO₂e/m²)', 
                     color='brown', linewidth=2)
        ax.set_ylabel('Cumulative Cost (€/m²)', fontsize=12, color='green')
        ax_twin.set_ylabel('Cumulative Carbon (kg CO₂e/m²)', fontsize=12, color='brown')
        ax.tick_params(axis='y', labelcolor='green')
        ax_twin.tick_params(axis='y', labelcolor='brown')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title('Economics & Environmental Impact', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure saved: {save_path}")
        
        plt.show()
    
    def save_results(self, df, output_dir='../results'):
        """
        Save simulation results to CSV and JSON.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Simulation history
        output_dir : str
            Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed CSV
        csv_path = output_path / f'baseline_simulation_{self.season}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Detailed results saved: {csv_path}")
        
        # Save summary JSON
        duration_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        summary = {
            'simulation_params': {
                'season': self.season,
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'duration_days': duration_days,
                'timestep_minutes': self.timestep,
                'greenhouse_area_m2': self.greenhouse_area
            },
            'resource_consumption': {
                'heating_MJ_m2': float(df['Heat_cons'].sum()),
                'heating_MJ_m2_day': float(df['Heat_cons'].sum() / duration_days),
                'electricity_kWh_m2': float(df['ElecHigh'].sum() + df['ElecLow'].sum()),
                'electricity_kWh_m2_day': float((df['ElecHigh'].sum() + df['ElecLow'].sum()) / duration_days),
                'co2_kg_m2': float(df['CO2_cons'].sum()),
                'co2_kg_m2_day': float(df['CO2_cons'].sum() / duration_days)
            },
            'economics': {
                'total_cost_eur_m2': float(df['cost_total'].sum()),
                'cost_heating_eur_m2': float(df['cost_heating'].sum()),
                'cost_electricity_eur_m2': float(df['cost_elec'].sum()),
                'cost_co2_eur_m2': float(df['cost_co2'].sum()),
                'daily_cost_eur_m2': float(df['cost_total'].sum() / duration_days)
            },
            'carbon_footprint': {
                'total_kgCO2e_m2': float(df['carbon_total'].sum()),
                'heating_kgCO2e_m2': float(df['carbon_heating'].sum()),
                'electricity_kgCO2e_m2': float(df['carbon_elec'].sum()),
                'co2_injection_kgCO2e_m2': float(df['carbon_co2'].sum()),
                'daily_kgCO2e_m2': float(df['carbon_total'].sum() / duration_days)
            },
            'climate_performance': {
                'temperature_mean_C': float(df['Tair'].mean()),
                'temperature_std_C': float(df['Tair'].std()),
                'temperature_range': [float(df['Tair'].min()), float(df['Tair'].max())],
                'humidity_mean_pct': float(df['Rhair'].mean()),
                'humidity_std_pct': float(df['Rhair'].std()),
                'co2_mean_ppm': float(df['CO2air'].mean()),
                'co2_std_ppm': float(df['CO2air'].std()),
                'par_mean_umol': float(df['Tot_PAR'].mean()),
                'par_std_umol': float(df['Tot_PAR'].std())
            }
        }
        
        json_path = output_path / f'baseline_summary_{self.season}.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Summary saved: {json_path}")


def main():
    """Main execution function."""
    print("="*70)
    print("🌿 BASELINE GREENHOUSE CONTROLLER")
    print("   Rule-Based Control Strategy Simulation")
    print("="*70)
    print()
    
    # Simulation parameters
    SEASONS = ['winter']  # Start with winter (can extend to all seasons)
    DURATION_DAYS = 7  # 1 week simulation
    
    for season in SEASONS:
        print(f"\n{'='*70}")
        print(f"Running {season.upper()} simulation")
        print(f"{'='*70}\n")
        
        # Initialize controller
        controller = BaselineController(
            start_date='2020-01-15',  # Mid-winter
            season=season
        )
        
        # Run simulation
        df = controller.simulate(duration_days=DURATION_DAYS, verbose=True)
        
        # Save results
        controller.save_results(df, output_dir='../results/baseline')
        
        # Create visualization
        fig_path = f'../results/baseline/baseline_simulation_{season}.png'
        controller.plot_results(df, save_path=fig_path)
        
        print(f"\n✅ {season.upper()} simulation complete!\n")
    
    print("="*70)
    print("🎉 ALL SIMULATIONS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
