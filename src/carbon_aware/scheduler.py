#!/usr/bin/env python3
"""
Carbon-Aware Scheduling Module for Greenhouse Control
=====================================================
Optimizes greenhouse control actions based on grid carbon intensity
and electricity pricing to minimize environmental impact and costs.

Features:
- Real-time grid carbon intensity profiles (Germany/Netherlands)
- Time-of-use electricity pricing (peak/off-peak)
- Execution window selection for non-urgent operations
- Energy shifting strategies for load balancing

Author: EcoGrow Team
Date: October 15, 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")


class GridCarbonIntensityProfile:
    """
    Manages grid carbon intensity profiles for different regions and times.
    Based on real-world data from electricity grids.
    """
    
    def __init__(self, region: str = 'DE'):
        """
        Initialize carbon intensity profile.
        
        Parameters:
        -----------
        region : str
            Region code ('DE'=Germany, 'NL'=Netherlands, 'EU'=European average)
        """
        self.region = region
        self.carbon_profiles = self._load_carbon_profiles()
        
    def _load_carbon_profiles(self) -> Dict:
        """
        Load region-specific carbon intensity profiles.
        
        Returns:
        --------
        profiles : Dict
            Hourly carbon intensity profiles (kg CO₂/kWh) for each region
        """
        # Based on Electricity Maps / European Environmental Agency data (2020-2024)
        profiles = {
            'DE': {  # Germany
                'annual_average': 0.42,  # kg CO₂/kWh (used in EDA)
                'winter_average': 0.48,  # Higher coal usage in winter
                'summer_average': 0.36,  # More renewables in summer
                'hourly_pattern': self._germany_hourly_profile(),
                'renewable_fraction': 0.42  # 42% renewables (2020)
            },
            'NL': {  # Netherlands
                'annual_average': 0.38,
                'winter_average': 0.44,
                'summer_average': 0.32,
                'hourly_pattern': self._netherlands_hourly_profile(),
                'renewable_fraction': 0.35
            },
            'EU': {  # European average
                'annual_average': 0.35,
                'winter_average': 0.40,
                'summer_average': 0.30,
                'hourly_pattern': self._eu_hourly_profile(),
                'renewable_fraction': 0.38
            }
        }
        
        return profiles
    
    def _germany_hourly_profile(self) -> np.ndarray:
        """
        Germany's typical hourly carbon intensity pattern.
        
        Returns:
        --------
        pattern : np.ndarray [24]
            Normalized carbon intensity by hour (multiply by daily average)
        """
        # Based on historical data: high carbon during peak hours (coal/gas),
        # lower during night (more wind/nuclear baseload)
        pattern = np.array([
            0.90,  # 00:00 - Night, low demand, more baseload
            0.85,  # 01:00
            0.80,  # 02:00 - Minimum demand
            0.82,  # 03:00
            0.88,  # 04:00 - Starting to ramp up
            0.95,  # 05:00 - Morning ramp
            1.10,  # 06:00 - Peak starts, coal plants activate
            1.20,  # 07:00 - Morning peak
            1.15,  # 08:00
            1.10,  # 09:00 - Solar starts contributing (if sunny)
            1.00,  # 10:00 - Solar peak hours
            0.95,  # 11:00
            0.92,  # 12:00 - Midday, max solar
            0.90,  # 13:00
            0.95,  # 14:00
            1.00,  # 15:00 - Solar declining
            1.10,  # 16:00 - Evening ramp
            1.25,  # 17:00 - Evening peak starts
            1.30,  # 18:00 - Peak demand, high carbon
            1.28,  # 19:00
            1.20,  # 20:00 - Peak declining
            1.10,  # 21:00
            1.00,  # 22:00
            0.95,  # 23:00
        ])
        
        return pattern
    
    def _netherlands_hourly_profile(self) -> np.ndarray:
        """Netherlands hourly carbon intensity (similar to Germany but more gas)."""
        pattern = np.array([
            0.88, 0.83, 0.78, 0.80, 0.86, 0.93,
            1.08, 1.18, 1.12, 1.05, 0.98, 0.92,
            0.90, 0.88, 0.92, 0.98, 1.08, 1.22,
            1.28, 1.25, 1.18, 1.08, 0.98, 0.93
        ])
        return pattern
    
    def _eu_hourly_profile(self) -> np.ndarray:
        """European average hourly carbon intensity."""
        pattern = np.array([
            0.92, 0.87, 0.82, 0.84, 0.90, 0.97,
            1.05, 1.15, 1.10, 1.03, 0.96, 0.90,
            0.88, 0.86, 0.90, 0.96, 1.05, 1.18,
            1.24, 1.22, 1.15, 1.05, 0.95, 0.90
        ])
        return pattern
    
    def get_carbon_intensity(self, timestamp: datetime, season: str = 'winter') -> float:
        """
        Get carbon intensity at specific timestamp.
        
        Parameters:
        -----------
        timestamp : datetime
            Target timestamp
        season : str
            Season ('winter', 'spring', 'summer', 'autumn')
            
        Returns:
        --------
        carbon_intensity : float
            Carbon intensity in kg CO₂/kWh
        """
        profile = self.carbon_profiles[self.region]
        hour = timestamp.hour
        
        # Get seasonal base
        if season in ['winter', 'autumn']:
            base_carbon = profile['winter_average']
        else:
            base_carbon = profile['summer_average']
        
        # Apply hourly pattern
        hourly_factor = profile['hourly_pattern'][hour]
        carbon_intensity = base_carbon * hourly_factor
        
        return carbon_intensity
    
    def get_daily_profile(self, date: datetime, season: str = 'winter') -> pd.DataFrame:
        """
        Get full daily carbon intensity profile.
        
        Parameters:
        -----------
        date : datetime
            Target date
        season : str
            Season
            
        Returns:
        --------
        df : pd.DataFrame
            Hourly carbon intensity profile
        """
        profile = self.carbon_profiles[self.region]
        
        # Get seasonal base
        if season in ['winter', 'autumn']:
            base_carbon = profile['winter_average']
        else:
            base_carbon = profile['summer_average']
        
        # Create hourly timestamps
        timestamps = [date + timedelta(hours=h) for h in range(24)]
        carbon_intensities = base_carbon * profile['hourly_pattern']
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'hour': range(24),
            'carbon_intensity': carbon_intensities,
            'renewable_fraction': profile['renewable_fraction']
        })
        
        return df
    
    def find_cleanest_windows(self, start_time: datetime, duration_hours: int = 4,
                             season: str = 'winter', num_windows: int = 3) -> List[Dict]:
        """
        Find cleanest time windows for energy-intensive operations.
        
        Parameters:
        -----------
        start_time : datetime
            Search start time
        duration_hours : int
            Required window duration (hours)
        season : str
            Season
        num_windows : int
            Number of windows to return
            
        Returns:
        --------
        windows : List[Dict]
            List of cleanest time windows with carbon intensity
        """
        # Get 24-hour profile
        df = self.get_daily_profile(start_time, season)
        
        # Calculate rolling average for window duration
        df['rolling_carbon'] = df['carbon_intensity'].rolling(
            window=duration_hours, min_periods=duration_hours
        ).mean()
        
        # Find windows with lowest carbon
        df_sorted = df.dropna().sort_values('rolling_carbon')
        
        windows = []
        for i in range(min(num_windows, len(df_sorted))):
            row = df_sorted.iloc[i]
            windows.append({
                'start_time': row['timestamp'],
                'end_time': row['timestamp'] + timedelta(hours=duration_hours),
                'avg_carbon_intensity': row['rolling_carbon'],
                'carbon_reduction_pct': 100 * (1 - row['rolling_carbon'] / df['carbon_intensity'].mean())
            })
        
        return windows


class ElectricityPricingProfile:
    """
    Manages time-of-use electricity pricing.
    Based on Economics.pdf parameters.
    """
    
    def __init__(self, country: str = 'NL'):
        """
        Initialize pricing profile.
        
        Parameters:
        -----------
        country : str
            Country code ('NL'=Netherlands, 'DE'=Germany, 'EU'=European average)
        """
        self.country = country
        self.pricing = self._load_pricing_profiles()
        
    def _load_pricing_profiles(self) -> Dict:
        """
        Load country-specific electricity pricing.
        
        Returns:
        --------
        pricing : Dict
            Time-of-use pricing profiles (€/kWh)
        """
        # From Economics.pdf and European electricity market data (2020)
        pricing = {
            'NL': {
                'peak_hours': (7, 23),  # 07:00-23:00
                'peak_price': 0.08,     # €/kWh (from Economics.pdf)
                'offpeak_price': 0.04,  # €/kWh (from Economics.pdf)
                'super_offpeak_price': 0.02,  # €/kWh (02:00-05:00, special rate)
                'weekend_discount': 0.9,  # 10% discount on weekends
                'renewable_tariff': 0.06  # Fixed price for 100% renewable contracts
            },
            'DE': {
                'peak_hours': (7, 23),
                'peak_price': 0.10,
                'offpeak_price': 0.05,
                'super_offpeak_price': 0.025,
                'weekend_discount': 0.85,
                'renewable_tariff': 0.08
            },
            'EU': {
                'peak_hours': (7, 23),
                'peak_price': 0.09,
                'offpeak_price': 0.045,
                'super_offpeak_price': 0.0225,
                'weekend_discount': 0.875,
                'renewable_tariff': 0.07
            }
        }
        
        return pricing
    
    def get_price(self, timestamp: datetime, tariff_type: str = 'tou') -> float:
        """
        Get electricity price at specific timestamp.
        
        Parameters:
        -----------
        timestamp : datetime
            Target timestamp
        tariff_type : str
            Tariff type ('tou'=time-of-use, 'flat'=flat rate, 'renewable'=green)
            
        Returns:
        --------
        price : float
            Electricity price in €/kWh
        """
        pricing = self.pricing[self.country]
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        if tariff_type == 'renewable':
            return pricing['renewable_tariff']
        
        if tariff_type == 'flat':
            # Average of peak and off-peak
            return (pricing['peak_price'] + pricing['offpeak_price']) / 2
        
        # Time-of-use pricing
        peak_start, peak_end = pricing['peak_hours']
        
        if 2 <= hour < 5:
            # Super off-peak (night)
            price = pricing['super_offpeak_price']
        elif peak_start <= hour < peak_end:
            # Peak hours
            price = pricing['peak_price']
        else:
            # Off-peak
            price = pricing['offpeak_price']
        
        # Weekend discount
        if is_weekend:
            price *= pricing['weekend_discount']
        
        return price
    
    def get_daily_profile(self, date: datetime) -> pd.DataFrame:
        """
        Get full daily pricing profile.
        
        Parameters:
        -----------
        date : datetime
            Target date
            
        Returns:
        --------
        df : pd.DataFrame
            Hourly pricing profile
        """
        timestamps = [date + timedelta(hours=h) for h in range(24)]
        prices_tou = [self.get_price(ts, 'tou') for ts in timestamps]
        prices_flat = [self.get_price(ts, 'flat') for ts in timestamps]
        prices_renewable = [self.get_price(ts, 'renewable') for ts in timestamps]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'hour': range(24),
            'price_tou': prices_tou,
            'price_flat': prices_flat,
            'price_renewable': prices_renewable
        })
        
        return df
    
    def find_cheapest_windows(self, start_time: datetime, duration_hours: int = 4,
                             num_windows: int = 3) -> List[Dict]:
        """
        Find cheapest time windows for energy-intensive operations.
        
        Parameters:
        -----------
        start_time : datetime
            Search start time
        duration_hours : int
            Required window duration (hours)
        num_windows : int
            Number of windows to return
            
        Returns:
        --------
        windows : List[Dict]
            List of cheapest time windows with pricing
        """
        # Get 24-hour profile
        df = self.get_daily_profile(start_time)
        
        # Calculate rolling average for window duration
        df['rolling_price'] = df['price_tou'].rolling(
            window=duration_hours, min_periods=duration_hours
        ).mean()
        
        # Find windows with lowest price
        df_sorted = df.dropna().sort_values('rolling_price')
        
        windows = []
        for i in range(min(num_windows, len(df_sorted))):
            row = df_sorted.iloc[i]
            windows.append({
                'start_time': row['timestamp'],
                'end_time': row['timestamp'] + timedelta(hours=duration_hours),
                'avg_price': row['rolling_price'],
                'cost_savings_pct': 100 * (1 - row['rolling_price'] / df['price_tou'].mean())
            })
        
        return windows


class CarbonAwareScheduler:
    """
    Carbon-aware scheduler that optimizes greenhouse control decisions
    based on grid carbon intensity and electricity pricing.
    """
    
    def __init__(self, region: str = 'DE', country: str = 'NL'):
        """
        Initialize carbon-aware scheduler.
        
        Parameters:
        -----------
        region : str
            Grid region for carbon intensity
        country : str
            Country for electricity pricing
        """
        self.carbon_profile = GridCarbonIntensityProfile(region)
        self.pricing_profile = ElectricityPricingProfile(country)
        
        # Scheduler parameters
        self.optimization_objective = 'balanced'  # 'cost', 'carbon', 'balanced'
        self.lambda_cost = 0.5  # Weight for cost in balanced optimization
        self.lambda_carbon = 0.5  # Weight for carbon in balanced optimization
        
    def evaluate_execution_window(self, start_time: datetime, duration_hours: int,
                                 energy_kwh: float, season: str = 'winter') -> Dict:
        """
        Evaluate cost and carbon for executing operation in given window.
        
        Parameters:
        -----------
        start_time : datetime
            Window start time
        duration_hours : int
            Operation duration
        energy_kwh : float
            Total energy consumption (kWh)
        season : str
            Season
            
        Returns:
        --------
        evaluation : Dict
            Cost, carbon, and combined score
        """
        # Average carbon intensity in window
        df_carbon = self.carbon_profile.get_daily_profile(start_time, season)
        hour_start = start_time.hour
        hour_end = (hour_start + duration_hours) % 24
        
        if hour_end > hour_start:
            carbon_window = df_carbon.iloc[hour_start:hour_end]['carbon_intensity'].mean()
        else:
            # Wraps around midnight
            carbon_window = pd.concat([
                df_carbon.iloc[hour_start:],
                df_carbon.iloc[:hour_end]
            ])['carbon_intensity'].mean()
        
        # Average price in window
        df_price = self.pricing_profile.get_daily_profile(start_time)
        if hour_end > hour_start:
            price_window = df_price.iloc[hour_start:hour_end]['price_tou'].mean()
        else:
            price_window = pd.concat([
                df_price.iloc[hour_start:],
                df_price.iloc[:hour_end]
            ])['price_tou'].mean()
        
        # Calculate totals
        total_cost = energy_kwh * price_window
        total_carbon = energy_kwh * carbon_window
        
        # Combined score (normalized to 0-100, lower is better)
        avg_carbon = df_carbon['carbon_intensity'].mean()
        avg_price = df_price['price_tou'].mean()
        
        carbon_score = 100 * (carbon_window / avg_carbon)
        cost_score = 100 * (price_window / avg_price)
        
        combined_score = (
            self.lambda_carbon * carbon_score +
            self.lambda_cost * cost_score
        )
        
        return {
            'start_time': start_time,
            'end_time': start_time + timedelta(hours=duration_hours),
            'duration_hours': duration_hours,
            'energy_kwh': energy_kwh,
            'avg_carbon_intensity': carbon_window,
            'total_carbon_kg': total_carbon,
            'avg_price_eur_kwh': price_window,
            'total_cost_eur': total_cost,
            'carbon_score': carbon_score,
            'cost_score': cost_score,
            'combined_score': combined_score
        }
    
    def find_optimal_window(self, reference_time: datetime, duration_hours: int,
                           energy_kwh: float, season: str = 'winter',
                           look_ahead_hours: int = 24) -> Dict:
        """
        Find optimal execution window within look-ahead period.
        
        Parameters:
        -----------
        reference_time : datetime
            Current/reference time
        duration_hours : int
            Operation duration
        energy_kwh : float
            Energy consumption
        season : str
            Season
        look_ahead_hours : int
            How far ahead to search (hours)
            
        Returns:
        --------
        optimal_window : Dict
            Best window with evaluation metrics
        """
        # Evaluate all possible windows
        evaluations = []
        
        for offset in range(0, look_ahead_hours - duration_hours + 1):
            start_time = reference_time + timedelta(hours=offset)
            eval_result = self.evaluate_execution_window(
                start_time, duration_hours, energy_kwh, season
            )
            evaluations.append(eval_result)
        
        # Find window with lowest combined score
        evaluations_sorted = sorted(evaluations, key=lambda x: x['combined_score'])
        optimal_window = evaluations_sorted[0]
        
        # Calculate improvement vs immediate execution
        immediate_eval = evaluations[0]
        optimal_window['carbon_reduction_pct'] = 100 * (
            1 - optimal_window['total_carbon_kg'] / immediate_eval['total_carbon_kg']
        )
        optimal_window['cost_savings_pct'] = 100 * (
            1 - optimal_window['total_cost_eur'] / immediate_eval['total_cost_eur']
        )
        optimal_window['combined_improvement_pct'] = 100 * (
            1 - optimal_window['combined_score'] / immediate_eval['combined_score']
        )
        
        return optimal_window
    
    def create_decision_log(self, reference_time: datetime, optimal_window: Dict,
                           immediate_eval: Dict, save_path: Optional[str] = None) -> Dict:
        """
        Create decision log for carbon-aware scheduling decision.
        
        Parameters:
        -----------
        reference_time : datetime
            Decision time
        optimal_window : Dict
            Chosen optimal window
        immediate_eval : Dict
            Evaluation if executed immediately
        save_path : str, optional
            Path to save decision log JSON
            
        Returns:
        --------
        decision_log : Dict
            Complete decision log
        """
        delay_hours = (optimal_window['start_time'] - reference_time).total_seconds() / 3600
        
        decision_log = {
            'decision_metadata': {
                'decision_time': reference_time.isoformat(),
                'scheduler_objective': self.optimization_objective,
                'lambda_cost': self.lambda_cost,
                'lambda_carbon': self.lambda_carbon
            },
            'naive_execution': {
                'start_time': immediate_eval['start_time'].isoformat(),
                'end_time': immediate_eval['end_time'].isoformat(),
                'total_cost_eur': immediate_eval['total_cost_eur'],
                'total_carbon_kg': immediate_eval['total_carbon_kg'],
                'avg_carbon_intensity': immediate_eval['avg_carbon_intensity'],
                'avg_price': immediate_eval['avg_price_eur_kwh']
            },
            'optimal_execution': {
                'start_time': optimal_window['start_time'].isoformat(),
                'end_time': optimal_window['end_time'].isoformat(),
                'total_cost_eur': optimal_window['total_cost_eur'],
                'total_carbon_kg': optimal_window['total_carbon_kg'],
                'avg_carbon_intensity': optimal_window['avg_carbon_intensity'],
                'avg_price': optimal_window['avg_price_eur_kwh'],
                'delay_hours': delay_hours
            },
            'improvement': {
                'carbon_reduction_kg': immediate_eval['total_carbon_kg'] - optimal_window['total_carbon_kg'],
                'carbon_reduction_pct': optimal_window['carbon_reduction_pct'],
                'cost_savings_eur': immediate_eval['total_cost_eur'] - optimal_window['total_cost_eur'],
                'cost_savings_pct': optimal_window['cost_savings_pct'],
                'combined_improvement_pct': optimal_window['combined_improvement_pct']
            },
            'decision': 'defer' if delay_hours > 0 else 'execute_immediately',
            'rationale': self._generate_rationale(optimal_window, immediate_eval, delay_hours)
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(decision_log, f, indent=2)
            print(f"✅ Decision log saved: {save_path}")
        
        return decision_log
    
    def _generate_rationale(self, optimal: Dict, immediate: Dict, delay_hours: float) -> str:
        """Generate human-readable rationale for scheduling decision."""
        if delay_hours < 0.1:
            return "Execute immediately - current window is optimal (low carbon intensity and favorable pricing)."
        
        carbon_reduction = optimal['carbon_reduction_pct']
        cost_savings = optimal['cost_savings_pct']
        
        rationale = f"Defer execution by {delay_hours:.1f} hours. "
        
        if carbon_reduction > 10 and cost_savings > 10:
            rationale += f"This reduces carbon emissions by {carbon_reduction:.1f}% and costs by {cost_savings:.1f}%."
        elif carbon_reduction > 10:
            rationale += f"This reduces carbon emissions by {carbon_reduction:.1f}% (primary optimization goal)."
        elif cost_savings > 10:
            rationale += f"This reduces costs by {cost_savings:.1f}% (primary optimization goal)."
        else:
            rationale += "Minor improvements in both carbon and cost achieved."
        
        return rationale
    
    def visualize_daily_optimization(self, date: datetime, season: str = 'winter',
                                    save_path: Optional[str] = None):
        """
        Visualize carbon intensity and pricing profiles for a day.
        
        Parameters:
        -----------
        date : datetime
            Target date
        season : str
            Season
        save_path : str, optional
            Path to save figure
        """
        # Get daily profiles
        df_carbon = self.carbon_profile.get_daily_profile(date, season)
        df_price = self.pricing_profile.get_daily_profile(date)
        
        # Merge data
        df = pd.merge(df_carbon, df_price, on=['timestamp', 'hour'])
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Carbon intensity plot
        ax1 = axes[0]
        ax1.plot(df['hour'], df['carbon_intensity'], 'o-', color='brown', linewidth=2, markersize=6)
        ax1.axhline(df['carbon_intensity'].mean(), color='brown', linestyle='--', alpha=0.5, label='Daily average')
        ax1.fill_between(df['hour'], df['carbon_intensity'], alpha=0.3, color='brown')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Carbon Intensity (kg CO₂/kWh)', fontsize=12, color='brown')
        ax1.set_title(f'Grid Carbon Intensity Profile - {self.carbon_profile.region} ({season})', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.tick_params(axis='y', labelcolor='brown')
        
        # Highlight cleanest windows
        cleanest_idx = df['carbon_intensity'].nsmallest(4).index
        for idx in cleanest_idx:
            ax1.axvspan(df.loc[idx, 'hour'] - 0.5, df.loc[idx, 'hour'] + 0.5, 
                       alpha=0.2, color='green')
        
        # Pricing plot
        ax2 = axes[1]
        ax2.plot(df['hour'], df['price_tou'], 's-', color='darkgreen', linewidth=2, 
                markersize=6, label='Time-of-Use')
        ax2.plot(df['hour'], df['price_flat'], '--', color='gray', linewidth=1.5, 
                alpha=0.7, label='Flat Rate')
        ax2.plot(df['hour'], df['price_renewable'], '-.', color='lightgreen', linewidth=1.5, 
                alpha=0.7, label='100% Renewable')
        ax2.axhline(df['price_tou'].mean(), color='darkgreen', linestyle=':', alpha=0.5)
        ax2.fill_between(df['hour'], df['price_tou'], alpha=0.3, color='darkgreen')
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Electricity Price (€/kWh)', fontsize=12, color='darkgreen')
        ax2.set_title(f'Electricity Pricing Profile - {self.pricing_profile.country}', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.tick_params(axis='y', labelcolor='darkgreen')
        
        # Highlight cheapest windows
        cheapest_idx = df['price_tou'].nsmallest(4).index
        for idx in cheapest_idx:
            ax2.axvspan(df.loc[idx, 'hour'] - 0.5, df.loc[idx, 'hour'] + 0.5, 
                       alpha=0.2, color='green')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Figure saved: {save_path}")
        
        plt.show()


def main():
    """Demonstration of carbon-aware scheduling."""
    print("="*70)
    print("🌍 CARBON-AWARE SCHEDULING MODULE")
    print("="*70)
    print()
    
    # Initialize scheduler
    scheduler = CarbonAwareScheduler(region='DE', country='NL')
    print(f"✅ Scheduler initialized:")
    print(f"   Region: {scheduler.carbon_profile.region}")
    print(f"   Country: {scheduler.pricing_profile.country}")
    print(f"   Objective: {scheduler.optimization_objective}")
    print()
    
    # Example: Schedule a high-energy operation
    reference_time = datetime(2020, 1, 15, 14, 0)  # Mid-afternoon
    duration_hours = 4
    energy_kwh = 50.0  # High-power heating/lighting operation
    season = 'winter'
    
    print(f"📋 Scheduling Task:")
    print(f"   Reference time: {reference_time}")
    print(f"   Duration: {duration_hours} hours")
    print(f"   Energy consumption: {energy_kwh} kWh")
    print(f"   Season: {season}")
    print()
    
    # Evaluate immediate execution
    immediate_eval = scheduler.evaluate_execution_window(
        reference_time, duration_hours, energy_kwh, season
    )
    
    print(f"🔴 Immediate Execution (Naive):")
    print(f"   Carbon: {immediate_eval['total_carbon_kg']:.2f} kg CO₂")
    print(f"   Cost: €{immediate_eval['total_cost_eur']:.2f}")
    print(f"   Carbon intensity: {immediate_eval['avg_carbon_intensity']:.3f} kg CO₂/kWh")
    print(f"   Price: €{immediate_eval['avg_price_eur_kwh']:.3f}/kWh")
    print()
    
    # Find optimal window
    optimal_window = scheduler.find_optimal_window(
        reference_time, duration_hours, energy_kwh, season, look_ahead_hours=24
    )
    
    print(f"🟢 Optimal Execution (Carbon-Aware):")
    print(f"   Start: {optimal_window['start_time']}")
    print(f"   Carbon: {optimal_window['total_carbon_kg']:.2f} kg CO₂")
    print(f"   Cost: €{optimal_window['total_cost_eur']:.2f}")
    print(f"   Carbon intensity: {optimal_window['avg_carbon_intensity']:.3f} kg CO₂/kWh")
    print(f"   Price: €{optimal_window['avg_price_eur_kwh']:.3f}/kWh")
    print()
    
    print(f"💡 Improvement:")
    print(f"   Carbon reduction: {optimal_window['carbon_reduction_pct']:.1f}%")
    print(f"   Cost savings: {optimal_window['cost_savings_pct']:.1f}%")
    print(f"   Combined improvement: {optimal_window['combined_improvement_pct']:.1f}%")
    print()
    
    # Create decision log
    decision_log = scheduler.create_decision_log(
        reference_time, optimal_window, immediate_eval,
        save_path='../results/carbon_aware_decision.json'
    )
    
    print(f"📝 Decision: {decision_log['decision'].upper()}")
    print(f"   Rationale: {decision_log['rationale']}")
    print()
    
    # Visualize daily profiles
    print("📊 Generating visualization...")
    scheduler.visualize_daily_optimization(
        reference_time, season,
        save_path='../results/carbon_aware_profiles.png'
    )
    
    print()
    print("="*70)
    print("✅ CARBON-AWARE SCHEDULING DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
