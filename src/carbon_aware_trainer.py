"""
Carbon-Aware Scheduler for HACK4EARTH Green AI Challenge
Track A: Schedule training during low-carbon grid hours
"""

import requests
import time
from datetime import datetime, timedelta
import json


class CarbonAwareScheduler:
    """Schedule training during low-carbon grid hours"""
    
    def __init__(self, region='DE'):
        """
        Initialize scheduler
        
        Args:
            region: Region code (DE=Germany, FR=France, etc.)
        """
        self.region = region
        self.api_url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
        
        # Fallback data for Germany (g CO₂/kWh by hour)
        self.germany_hourly_intensity = {
            0: 550, 1: 560, 2: 570, 3: 580, 4: 570, 5: 550,
            6: 520, 7: 480, 8: 420, 9: 350, 10: 280, 11: 250,
            12: 240, 13: 250, 14: 260, 15: 280, 16: 320, 17: 380,
            18: 450, 19: 520, 20: 540, 21: 550, 22: 540, 23: 530
        }
        
    def get_carbon_intensity(self):
        """
        Get current grid carbon intensity (g CO₂/kWh)
        
        Returns:
            Carbon intensity in g CO₂/kWh
        """
        # Try to fetch from API
        try:
            headers = {
                'auth-token': 'YOUR_API_TOKEN_HERE'  # Replace with actual token if available
            }
            params = {'zone': self.region}
            response = requests.get(self.api_url, headers=headers, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                intensity = data.get('carbonIntensity', None)
                if intensity is not None:
                    print(f"📡 Real-time carbon intensity: {intensity:.0f} g CO₂/kWh")
                    return intensity
        except Exception as e:
            print(f"⚠️  API request failed: {e}, using fallback data")
        
        # Fallback to time-based estimates
        hour = datetime.now().hour
        
        if self.region == 'DE':  # Germany
            intensity = self.germany_hourly_intensity.get(hour, 420)
        else:
            # Generic fallback
            if 10 <= hour <= 16:  # Solar peak hours
                intensity = 250
            else:
                intensity = 550
        
        print(f"🕐 Hour {hour}: Carbon intensity ~{intensity} g CO₂/kWh (estimated)")
        return intensity
    
    def should_train_now(self, threshold=350):
        """
        Determine if now is a good time to train
        
        Args:
            threshold: Maximum acceptable carbon intensity (g CO₂/kWh)
            
        Returns:
            Boolean indicating whether to train
        """
        intensity = self.get_carbon_intensity()
        return intensity < threshold
    
    def get_next_clean_window(self, threshold=350, hours_ahead=24):
        """
        Find next clean energy window
        
        Args:
            threshold: Maximum acceptable carbon intensity
            hours_ahead: How many hours to look ahead
            
        Returns:
            Datetime of next clean window, or None if none found
        """
        current_hour = datetime.now().hour
        
        for i in range(hours_ahead):
            check_hour = (current_hour + i) % 24
            intensity = self.germany_hourly_intensity.get(check_hour, 420)
            
            if intensity < threshold:
                time_until = i
                next_window = datetime.now() + timedelta(hours=time_until)
                return next_window, intensity
        
        return None, None
    
    def schedule_training(self, train_function, max_wait_hours=12, threshold=350):
        """
        Wait for clean energy window and then train
        
        Args:
            train_function: Function to call when clean energy is available
            max_wait_hours: Maximum hours to wait (0 = train immediately)
            threshold: Carbon intensity threshold (g CO₂/kWh)
            
        Returns:
            Result from train_function
        """
        if max_wait_hours == 0:
            print("⚡ Training immediately (carbon-awareness disabled)")
            return train_function()
        
        print(f"🌱 Carbon-aware scheduling enabled (threshold: {threshold} g CO₂/kWh)")
        
        start_time = datetime.now()
        check_interval = 1800  # Check every 30 minutes
        
        while True:
            elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
            
            if elapsed_hours >= max_wait_hours:
                print(f"⏰ Max wait time ({max_wait_hours}h) reached, training anyway...")
                break
            
            if self.should_train_now(threshold):
                print("✅ Clean energy window detected! Starting training...")
                break
            else:
                intensity = self.get_carbon_intensity()
                next_window, next_intensity = self.get_next_clean_window(threshold)
                
                if next_window:
                    print(f"⏳ Current: {intensity} g/kWh (too high)")
                    print(f"   Next clean window: {next_window.strftime('%H:%M')} "
                          f"({next_intensity} g/kWh)")
                else:
                    print(f"⏳ Current: {intensity} g/kWh (waiting...)")
                
                print(f"   Checking again in {check_interval/60:.0f} minutes...")
                time.sleep(check_interval)
        
        return train_function()
    
    def estimate_carbon_savings(self, energy_kwh, wait_for_clean=True):
        """
        Estimate carbon savings from carbon-aware scheduling
        
        Args:
            energy_kwh: Energy consumed by training (kWh)
            wait_for_clean: Whether carbon-aware scheduling is used
            
        Returns:
            Dictionary with carbon metrics
        """
        if wait_for_clean:
            # Train during solar peak (low intensity)
            clean_intensity = 250  # g CO₂/kWh
            carbon_kg = energy_kwh * clean_intensity / 1000
            mode = "carbon-aware"
        else:
            # Train immediately (average intensity)
            avg_intensity = 420  # g CO₂/kWh
            carbon_kg = energy_kwh * avg_intensity / 1000
            mode = "immediate"
        
        baseline_carbon = energy_kwh * 420 / 1000  # Always compare to average
        savings_kg = baseline_carbon - carbon_kg
        savings_pct = (savings_kg / baseline_carbon * 100) if baseline_carbon > 0 else 0
        
        return {
            'energy_kwh': energy_kwh,
            'carbon_kg': carbon_kg,
            'baseline_carbon_kg': baseline_carbon,
            'savings_kg': savings_kg,
            'savings_pct': savings_pct,
            'mode': mode
        }


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Carbon-aware training scheduler')
    parser.add_argument('--region', type=str, default='DE', help='Region code')
    parser.add_argument('--threshold', type=int, default=350, 
                       help='Carbon intensity threshold (g CO₂/kWh)')
    parser.add_argument('--max_wait', type=int, default=12, 
                       help='Maximum hours to wait (0=train immediately)')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check current carbon intensity')
    
    args = parser.parse_args()
    
    scheduler = CarbonAwareScheduler(region=args.region)
    
    if args.check_only:
        # Just check current intensity
        print("\n" + "="*60)
        print("CURRENT CARBON INTENSITY")
        print("="*60)
        intensity = scheduler.get_carbon_intensity()
        
        if intensity < args.threshold:
            print(f"✅ Good time to train! ({intensity} < {args.threshold})")
        else:
            print(f"⏳ Consider waiting ({intensity} > {args.threshold})")
        
        next_window, next_intensity = scheduler.get_next_clean_window(args.threshold)
        if next_window:
            print(f"\n🌱 Next clean window: {next_window.strftime('%H:%M')} "
                  f"({next_intensity} g CO₂/kWh)")
        
    else:
        # Schedule training
        def mock_training():
            print("\n🚀 Starting training...")
            time.sleep(2)  # Simulate training
            print("✅ Training complete!")
            return {"status": "success", "energy_kwh": 0.5}
        
        print("\n" + "="*60)
        print("CARBON-AWARE TRAINING SCHEDULER")
        print("="*60)
        
        result = scheduler.schedule_training(
            mock_training,
            max_wait_hours=args.max_wait,
            threshold=args.threshold
        )
        
        # Estimate savings
        if result and 'energy_kwh' in result:
            print("\n" + "="*60)
            print("CARBON SAVINGS ESTIMATE")
            print("="*60)
            
            savings = scheduler.estimate_carbon_savings(
                energy_kwh=result['energy_kwh'],
                wait_for_clean=(args.max_wait > 0)
            )
            
            print(f"Energy consumed: {savings['energy_kwh']:.3f} kWh")
            print(f"Carbon emitted: {savings['carbon_kg']*1000:.1f} g CO₂e ({savings['mode']})")
            
            if savings['savings_kg'] > 0:
                print(f"✅ Savings: {savings['savings_kg']*1000:.1f} g CO₂e "
                      f"({savings['savings_pct']:.1f}% reduction)")
            else:
                print(f"Baseline: {savings['baseline_carbon_kg']*1000:.1f} g CO₂e")
