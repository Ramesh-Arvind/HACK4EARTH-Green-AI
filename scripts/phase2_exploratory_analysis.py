"""
Phase 2: Exploratory Data Analysis (EDA)
Wageningen Autonomous Greenhouse Challenge Dataset

This script performs comprehensive EDA including:
- Statistical profiling
- Time-series visualization
- Correlation & causality analysis
- Economic context analysis
- Crop yield relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_PATH = Path("/home/rnaa/paper_5_pica_whatif/ecogrow/data/AutonomousGreenhouseChallenge_edition2")
REFERENCE_PATH = BASE_PATH / "Reference"
OUTPUT_PATH = Path("/home/rnaa/paper_5_pica_whatif/ecogrow/results")
FIGURES_PATH = OUTPUT_PATH / "eda_figures"
FIGURES_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
print("Wageningen Autonomous Greenhouse Challenge - 2nd Edition (2019)")
print("=" * 80)

# ============================================================================
# LOAD DATASETS
# ============================================================================

print("\n📊 Loading datasets...")

# Load GreenhouseClimate
df_climate = pd.read_csv(REFERENCE_PATH / "GreenhouseClimate.csv")
time_col = [col for col in df_climate.columns if col.startswith('%')][0]
df_climate['datetime'] = pd.TimedeltaIndex(df_climate[time_col], unit='D') + pd.Timestamp('1899-12-30')
df_climate = df_climate.sort_values('datetime').reset_index(drop=True)

# Load Resources
df_resource = pd.read_csv(REFERENCE_PATH / "Resources.csv")
time_col_res = [col for col in df_resource.columns if 'Time' in col][0]
df_resource['datetime'] = pd.TimedeltaIndex(df_resource[time_col_res], unit='D') + pd.Timestamp('1899-12-30')
df_resource = df_resource.sort_values('datetime').reset_index(drop=True)

# Load Weather (in subfolder)
weather_path = BASE_PATH / "Weather" / "Weather.csv"
df_weather = pd.read_csv(weather_path)
time_col_weather = [col for col in df_weather.columns if 'time' in col.lower()][0]
df_weather['datetime'] = pd.TimedeltaIndex(df_weather[time_col_weather], unit='D') + pd.Timestamp('1899-12-30')

# Load Production
df_production = pd.read_csv(REFERENCE_PATH / "Production.csv")
time_col_prod = [col for col in df_production.columns if 'time' in col.lower()][0]
df_production['datetime'] = pd.TimedeltaIndex(df_production[time_col_prod], unit='D') + pd.Timestamp('1899-12-30')

# Load TomQuality
df_quality = pd.read_csv(REFERENCE_PATH / "TomQuality.csv")
time_col_qual = [col for col in df_quality.columns if 'time' in col.lower()][0]
df_quality['datetime'] = pd.TimedeltaIndex(df_quality[time_col_qual], unit='D') + pd.Timestamp('1899-12-30')

print(f"✅ Climate data: {len(df_climate):,} records")
print(f"✅ Resource data: {len(df_resource)} records")
print(f"✅ Weather data: {len(df_weather):,} records")
print(f"✅ Production data: {len(df_production)} records")
print(f"✅ Quality data: {len(df_quality)} records")

# ============================================================================
# 1. STATISTICAL PROFILING
# ============================================================================

print("\n" + "=" * 80)
print("1. STATISTICAL PROFILING")
print("=" * 80)

# Climate variables statistics - convert to numeric first
climate_vars = ['Tair', 'Rhair', 'CO2air', 'HumDef', 'VentLee', 'Ventwind', 
                'AssimLight', 'PipeLow', 'PipeGrow', 'Tot_PAR']

# Convert to numeric, coerce errors to NaN
for col in climate_vars:
    df_climate[col] = pd.to_numeric(df_climate[col], errors='coerce')

climate_stats = df_climate[climate_vars].describe().T
climate_stats['missing_pct'] = (df_climate[climate_vars].isnull().sum() / len(df_climate) * 100)

# Calculate outliers safely
outlier_counts = []
for var in climate_vars:
    data = df_climate[var].dropna()
    if len(data) > 0:
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = sum((data < lower) | (data > upper))
        outlier_counts.append(int(outliers))
    else:
        outlier_counts.append(0)

climate_stats['outliers'] = outlier_counts

print("\n📊 Climate Variables Statistics:")
print(climate_stats[['mean', 'std', 'min', 'max', 'missing_pct', 'outliers']].to_string())

# Resource variables statistics
resource_vars = ['Heat_cons', 'ElecHigh', 'ElecLow', 'CO2_cons', 'Irr', 'Drain']
resource_stats = df_resource[resource_vars].describe().T
resource_stats['missing_pct'] = (df_resource[resource_vars].isnull().sum() / len(df_resource) * 100)

print("\n📊 Resource Consumption Statistics:")
print(resource_stats[['mean', 'std', 'min', 'max', 'missing_pct']].to_string())

# Weather variables statistics
weather_vars = ['Tout', 'Rhout', 'Iglob', 'Windsp', 'PARout']
weather_stats = df_weather[weather_vars].describe().T

print("\n📊 Weather Variables Statistics:")
print(weather_stats[['mean', 'std', 'min', 'max']].to_string())

# Save statistics
stats_summary = {
    "climate": climate_stats.to_dict(),
    "resources": resource_stats.to_dict(),
    "weather": weather_stats.to_dict()
}

with open(OUTPUT_PATH / "statistical_profile.json", 'w') as f:
    json.dump(stats_summary, f, indent=2, default=str)

# ============================================================================
# 2. TIME-SERIES VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("2. TIME-SERIES VISUALIZATION")
print("=" * 80)

# Figure 1: Daily Energy Patterns
print("\n📈 Creating Figure 1: Daily Energy Patterns...")

fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Heating consumption
axes[0].plot(df_resource['datetime'], df_resource['Heat_cons'], label='Heating', color='red', linewidth=1)
axes[0].set_ylabel('Heating (MJ/m²/day)', fontsize=12)
axes[0].set_title('Daily Energy Consumption Patterns - Reference Group', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Electricity consumption
axes[1].plot(df_resource['datetime'], df_resource['ElecHigh'], label='Peak hours', color='orange', linewidth=1)
axes[1].plot(df_resource['datetime'], df_resource['ElecLow'], label='Off-peak hours', color='blue', linewidth=1)
axes[1].set_ylabel('Electricity (kWh/m²/day)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# CO₂ consumption
axes[2].plot(df_resource['datetime'], df_resource['CO2_cons'], label='CO₂', color='green', linewidth=1)
axes[2].set_ylabel('CO₂ (kg/m²/day)', fontsize=12)
axes[2].set_xlabel('Date', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig1_daily_energy_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig1_daily_energy_patterns.png")

# Figure 2: Climate Variables Time Series
print("\n📈 Creating Figure 2: Climate Variables...")

# Sample every 12 points (hourly from 5-min data)
df_climate_hourly = df_climate.iloc[::12, :]

fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Temperature
axes[0].plot(df_climate_hourly['datetime'], df_climate_hourly['Tair'], label='Indoor', color='red', linewidth=0.8)
axes[0].plot(df_weather.iloc[::12, :]['datetime'], df_weather.iloc[::12, :]['Tout'], label='Outdoor', color='blue', linewidth=0.8)
axes[0].set_ylabel('Temperature (°C)', fontsize=12)
axes[0].set_title('Greenhouse Climate Time Series (Hourly)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Humidity
axes[1].plot(df_climate_hourly['datetime'], df_climate_hourly['Rhair'], label='Indoor RH', color='green', linewidth=0.8)
axes[1].set_ylabel('Humidity (%)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# CO₂
axes[2].plot(df_climate_hourly['datetime'], df_climate_hourly['CO2air'], label='CO₂', color='purple', linewidth=0.8)
axes[2].set_ylabel('CO₂ (ppm)', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# PAR
axes[3].plot(df_climate_hourly['datetime'], df_climate_hourly['Tot_PAR'], label='Total PAR', color='orange', linewidth=0.8)
axes[3].set_ylabel('PAR (µmol/m²/s)', fontsize=12)
axes[3].set_xlabel('Date', fontsize=12)
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig2_climate_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig2_climate_timeseries.png")

# Figure 3: Seasonal Trends
print("\n📈 Creating Figure 3: Seasonal Trends...")

# Aggregate by week
df_resource['week'] = df_resource['datetime'].dt.isocalendar().week
weekly_avg = df_resource.groupby('week')[resource_vars].mean()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(weekly_avg.index, weekly_avg['Heat_cons'], marker='o', linewidth=2, markersize=6, color='red')
axes[0, 0].set_ylabel('Heating (MJ/m²/day)', fontsize=12)
axes[0, 0].set_title('Weekly Average Energy Consumption', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(weekly_avg.index, weekly_avg['ElecHigh'] + weekly_avg['ElecLow'], marker='o', linewidth=2, markersize=6, color='orange')
axes[0, 1].set_ylabel('Electricity (kWh/m²/day)', fontsize=12)
axes[0, 1].set_title('Total Electricity Consumption', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(weekly_avg.index, weekly_avg['CO2_cons'], marker='o', linewidth=2, markersize=6, color='green')
axes[1, 0].set_ylabel('CO₂ (kg/m²/day)', fontsize=12)
axes[1, 0].set_xlabel('Week of Year', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(weekly_avg.index, weekly_avg['Irr'], marker='o', linewidth=2, markersize=6, color='blue')
axes[1, 1].plot(weekly_avg.index, weekly_avg['Drain'], marker='s', linewidth=2, markersize=6, color='cyan')
axes[1, 1].set_ylabel('Water (L/m²/day)', fontsize=12)
axes[1, 1].set_xlabel('Week of Year', fontsize=12)
axes[1, 1].legend(['Irrigation', 'Drainage'])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig3_seasonal_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig3_seasonal_trends.png")

# Figure 4: Diurnal Patterns
print("\n📈 Creating Figure 4: Diurnal Patterns...")

# Extract hour of day
df_climate_sample = df_climate.iloc[::12, :].copy()  # Hourly sampling
df_climate_sample['hour'] = df_climate_sample['datetime'].dt.hour

# Average by hour
hourly_avg = df_climate_sample.groupby('hour')[['Tair', 'CO2air', 'Tot_PAR', 'PipeLow']].mean()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(hourly_avg.index, hourly_avg['Tair'], marker='o', linewidth=2, markersize=6, color='red')
axes[0, 0].set_ylabel('Temperature (°C)', fontsize=12)
axes[0, 0].set_title('Diurnal Patterns (Average by Hour)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(hourly_avg.index, hourly_avg['CO2air'], marker='o', linewidth=2, markersize=6, color='green')
axes[0, 1].set_ylabel('CO₂ (ppm)', fontsize=12)
axes[0, 1].set_xlabel('Hour of Day', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(hourly_avg.index, hourly_avg['Tot_PAR'], marker='o', linewidth=2, markersize=6, color='orange')
axes[1, 0].set_ylabel('PAR (µmol/m²/s)', fontsize=12)
axes[1, 0].set_xlabel('Hour of Day', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(hourly_avg.index, hourly_avg['PipeLow'], marker='o', linewidth=2, markersize=6, color='purple')
axes[1, 1].set_ylabel('Heating Pipe (°C)', fontsize=12)
axes[1, 1].set_xlabel('Hour of Day', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig4_diurnal_patterns.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig4_diurnal_patterns.png")

# ============================================================================
# 3. CORRELATION & CAUSALITY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("3. CORRELATION & CAUSALITY ANALYSIS")
print("=" * 80)

# Merge datasets for correlation analysis
df_climate_daily = df_climate.copy()
df_climate_daily['date'] = df_climate_daily['datetime'].dt.date
df_climate_daily_avg = df_climate_daily.groupby('date')[climate_vars].mean().reset_index()

df_resource['date'] = df_resource['datetime'].dt.date

df_merged = df_resource.merge(df_climate_daily_avg, on='date', how='inner')

# Calculate correlation matrix
corr_vars = ['Heat_cons', 'ElecHigh', 'ElecLow', 'Tair', 'CO2air', 'Tot_PAR', 'VentLee', 'PipeLow']
corr_matrix = df_merged[corr_vars].corr()

# Figure 5: Correlation Heatmap
print("\n📈 Creating Figure 5: Correlation Heatmap...")

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 10})
ax.set_title('Correlation Matrix: Climate & Energy Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig5_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig5_correlation_heatmap.png")

# Print key correlations
print("\n🔍 Key Correlations:")
print(f"   Heat_cons vs Tair: {corr_matrix.loc['Heat_cons', 'Tair']:.3f}")
print(f"   ElecHigh vs Tot_PAR: {corr_matrix.loc['ElecHigh', 'Tot_PAR']:.3f}")
print(f"   VentLee vs Tair: {corr_matrix.loc['VentLee', 'Tair']:.3f}")

# Save correlation matrix
corr_matrix.to_csv(OUTPUT_PATH / 'correlation_matrix.csv')

# Figure 6: Scatter Plots (Key Relationships)
print("\n📈 Creating Figure 6: Key Relationships...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Heating vs Temperature (merge with weather)
df_weather_daily = df_weather.copy()
df_weather_daily['date'] = df_weather_daily['datetime'].dt.date
df_weather_daily_avg = df_weather_daily.groupby('date')['Tout'].mean().reset_index()
df_merged_weather = df_resource.merge(df_weather_daily_avg, on='date', how='inner')

axes[0, 0].scatter(df_merged_weather['Tout'], df_merged_weather['Heat_cons'], alpha=0.6, s=50)
axes[0, 0].set_xlabel('Outdoor Temperature (°C)', fontsize=12)
axes[0, 0].set_ylabel('Heating Consumption (MJ/m²/day)', fontsize=12)
axes[0, 0].set_title('Heating vs Outdoor Temperature', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(df_merged_weather['Tout'].dropna(), df_merged_weather['Heat_cons'].dropna(), 1)
p = np.poly1d(z)
x_line = np.linspace(df_merged_weather['Tout'].min(), df_merged_weather['Tout'].max(), 100)
axes[0, 0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
axes[0, 0].legend()

# Electricity vs PAR
axes[0, 1].scatter(df_merged['Tot_PAR'], df_merged['ElecHigh'] + df_merged['ElecLow'], alpha=0.6, s=50, color='orange')
axes[0, 1].set_xlabel('Total PAR (µmol/m²/s)', fontsize=12)
axes[0, 1].set_ylabel('Total Electricity (kWh/m²/day)', fontsize=12)
axes[0, 1].set_title('Electricity vs PAR (Lighting)', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# CO₂ vs Ventilation
axes[1, 0].scatter(df_merged['VentLee'], df_merged['CO2air'], alpha=0.6, s=50, color='green')
axes[1, 0].set_xlabel('Ventilation Opening (%)', fontsize=12)
axes[1, 0].set_ylabel('CO₂ Concentration (ppm)', fontsize=12)
axes[1, 0].set_title('CO₂ vs Ventilation', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Temperature vs Heating Pipe
axes[1, 1].scatter(df_merged['PipeLow'], df_merged['Tair'], alpha=0.6, s=50, color='red')
axes[1, 1].set_xlabel('Heating Pipe Temperature (°C)', fontsize=12)
axes[1, 1].set_ylabel('Air Temperature (°C)', fontsize=12)
axes[1, 1].set_title('Air Temp vs Heating Pipe', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig6_scatter_relationships.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig6_scatter_relationships.png")

# ============================================================================
# 4. ECONOMIC CONTEXT ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("4. ECONOMIC CONTEXT ANALYSIS")
print("=" * 80)

# Calculate costs (from Economics.pdf)
COST_HEAT = 0.0083  # €/MJ
COST_ELEC_PEAK = 0.08  # €/kWh
COST_ELEC_OFFPEAK = 0.04  # €/kWh
COST_CO2_TIER1 = 0.08  # €/kg (first 12 kg/m²)
COST_CO2_TIER2 = 0.20  # €/kg (additional)

df_resource['cost_heating'] = df_resource['Heat_cons'] * COST_HEAT
df_resource['cost_elec_peak'] = df_resource['ElecHigh'] * COST_ELEC_PEAK
df_resource['cost_elec_offpeak'] = df_resource['ElecLow'] * COST_ELEC_OFFPEAK

# CO₂ cost (tiered pricing)
df_resource['cumsum_co2'] = df_resource['CO2_cons'].cumsum()
df_resource['cost_co2'] = df_resource.apply(
    lambda row: (min(row['CO2_cons'], max(0, 12 - row['cumsum_co2'] + row['CO2_cons'])) * COST_CO2_TIER1 +
                 max(0, row['CO2_cons'] - max(0, 12 - row['cumsum_co2'] + row['CO2_cons'])) * COST_CO2_TIER2),
    axis=1
)

df_resource['total_cost'] = (df_resource['cost_heating'] + df_resource['cost_elec_peak'] + 
                              df_resource['cost_elec_offpeak'] + df_resource['cost_co2'])

# Calculate CO₂ emissions
EMISSION_HEAT = 0.056  # kg CO₂/MJ (natural gas)
EMISSION_ELEC = 0.42  # kg CO₂/kWh (Germany grid 2020)

df_resource['carbon_heating'] = df_resource['Heat_cons'] * EMISSION_HEAT
df_resource['carbon_elec'] = (df_resource['ElecHigh'] + df_resource['ElecLow']) * EMISSION_ELEC
df_resource['carbon_co2'] = df_resource['CO2_cons']  # Direct CO₂ injection
df_resource['total_carbon'] = df_resource['carbon_heating'] + df_resource['carbon_elec'] + df_resource['carbon_co2']

print(f"\n💰 Economic Analysis:")
print(f"   Total heating cost: €{df_resource['cost_heating'].sum():.2f}")
print(f"   Total electricity cost: €{(df_resource['cost_elec_peak'] + df_resource['cost_elec_offpeak']).sum():.2f}")
print(f"   Total CO₂ cost: €{df_resource['cost_co2'].sum():.2f}")
print(f"   TOTAL COST: €{df_resource['total_cost'].sum():.2f}")

print(f"\n🌱 Carbon Footprint:")
print(f"   Heating emissions: {df_resource['carbon_heating'].sum():.2f} kg CO₂e")
print(f"   Electricity emissions: {df_resource['carbon_elec'].sum():.2f} kg CO₂e")
print(f"   CO₂ injection: {df_resource['carbon_co2'].sum():.2f} kg CO₂")
print(f"   TOTAL EMISSIONS: {df_resource['total_carbon'].sum():.2f} kg CO₂e")

# Figure 7: Cost & Carbon Time Series
print("\n📈 Creating Figure 7: Cost & Carbon Analysis...")

fig, axes = plt.subplots(3, 1, figsize=(15, 10))

# Daily costs
axes[0].plot(df_resource['datetime'], df_resource['cost_heating'], label='Heating', linewidth=1.5)
axes[0].plot(df_resource['datetime'], df_resource['cost_elec_peak'] + df_resource['cost_elec_offpeak'], 
             label='Electricity', linewidth=1.5)
axes[0].plot(df_resource['datetime'], df_resource['cost_co2'], label='CO₂', linewidth=1.5)
axes[0].set_ylabel('Daily Cost (€/m²)', fontsize=12)
axes[0].set_title('Economic & Environmental Analysis', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative cost
axes[1].plot(df_resource['datetime'], df_resource['total_cost'].cumsum(), linewidth=2, color='red')
axes[1].set_ylabel('Cumulative Cost (€/m²)', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Daily carbon emissions
axes[2].plot(df_resource['datetime'], df_resource['carbon_heating'], label='Heating', linewidth=1.5)
axes[2].plot(df_resource['datetime'], df_resource['carbon_elec'], label='Electricity', linewidth=1.5)
axes[2].set_ylabel('Carbon (kg CO₂e/m²/day)', fontsize=12)
axes[2].set_xlabel('Date', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig7_economic_carbon_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig7_economic_carbon_analysis.png")

# Save economic data
economic_summary = {
    "total_cost_eur_m2": float(df_resource['total_cost'].sum()),
    "total_carbon_kg_co2e_m2": float(df_resource['total_carbon'].sum()),
    "breakdown_cost": {
        "heating": float(df_resource['cost_heating'].sum()),
        "electricity": float((df_resource['cost_elec_peak'] + df_resource['cost_elec_offpeak']).sum()),
        "co2": float(df_resource['cost_co2'].sum())
    },
    "breakdown_carbon": {
        "heating": float(df_resource['carbon_heating'].sum()),
        "electricity": float(df_resource['carbon_elec'].sum()),
        "co2_injection": float(df_resource['carbon_co2'].sum())
    }
}

with open(OUTPUT_PATH / 'economic_analysis.json', 'w') as f:
    json.dump(economic_summary, f, indent=2)

# ============================================================================
# 5. CROP YIELD RELATIONSHIPS
# ============================================================================

print("\n" + "=" * 80)
print("5. CROP YIELD RELATIONSHIPS")
print("=" * 80)

print(f"\n📊 Production Data:")
print(f"   Total harvests: {len(df_production)}")
print(f"   Total yield (Class A): {df_production['ProdA'].sum():.2f} kg/m²")
print(f"   Total yield (Class B): {df_production['ProdB'].sum():.2f} kg/m²")
print(f"   Total yield: {(df_production['ProdA'] + df_production['ProdB']).sum():.2f} kg/m²")

# Calculate yield per kg energy
total_energy_MJ = df_resource['Heat_cons'].sum() + (df_resource['ElecHigh'] + df_resource['ElecLow']).sum() * 3.6
total_yield_kg = (df_production['ProdA'] + df_production['ProdB']).sum() * 62.5  # Growing area
energy_per_kg = total_energy_MJ / total_yield_kg

print(f"\n⚡ Energy Efficiency:")
print(f"   Total energy: {total_energy_MJ:.2f} MJ")
print(f"   Total yield: {total_yield_kg:.2f} kg")
print(f"   Energy per kg tomato: {energy_per_kg:.2f} MJ/kg")

# Calculate economic efficiency
cost_per_kg = df_resource['total_cost'].sum() / (df_production['ProdA'] + df_production['ProdB']).sum()
print(f"\n💰 Economic Efficiency:")
print(f"   Resource cost per kg tomato: €{cost_per_kg:.2f}/kg")

# Assuming tomato price (average from Economics.pdf)
AVG_PRICE_CLASS_A = 3.5  # €/kg (mid-range Brix)
AVG_PRICE_CLASS_B = AVG_PRICE_CLASS_A / 2

income = (df_production['ProdA'].sum() * AVG_PRICE_CLASS_A + 
          df_production['ProdB'].sum() * AVG_PRICE_CLASS_B)
net_profit = income - df_resource['total_cost'].sum()

print(f"   Estimated income: €{income:.2f}/m²")
print(f"   Net profit: €{net_profit:.2f}/m²")
print(f"   Profit margin: {(net_profit/income)*100:.1f}%")

# Figure 8: Cumulative Yield & Quality
print("\n📈 Creating Figure 8: Yield & Quality...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Cumulative yield
axes[0, 0].plot(df_production['datetime'], df_production['ProdA'].cumsum(), 
                marker='o', linewidth=2, markersize=6, label='Class A', color='green')
axes[0, 0].plot(df_production['datetime'], df_production['ProdB'].cumsum(), 
                marker='s', linewidth=2, markersize=6, label='Class B', color='orange')
axes[0, 0].set_ylabel('Cumulative Yield (kg/m²)', fontsize=12)
axes[0, 0].set_title('Crop Yield & Quality Analysis', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Yield per harvest
total_prod = df_production['ProdA'] + df_production['ProdB']
axes[0, 1].bar(df_production['datetime'], df_production['ProdA'], label='Class A', color='green', alpha=0.7)
axes[0, 1].bar(df_production['datetime'], df_production['ProdB'], bottom=df_production['ProdA'], 
               label='Class B', color='orange', alpha=0.7)
axes[0, 1].set_ylabel('Yield per Harvest (kg/m²)', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Quality metrics (Brix, Flavor) - handle whitespace in column names
tss_col = [col for col in df_quality.columns if 'TSS' in col or 'tss' in col][0]
flavour_col = [col for col in df_quality.columns if 'Flavour' in col or 'flavour' in col.lower()][0]

axes[1, 0].plot(df_quality['datetime'], df_quality[tss_col], marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 0].set_ylabel('Brix (°Brix)', fontsize=12)
axes[1, 0].set_xlabel('Date', fontsize=12)
axes[1, 0].set_title('Total Soluble Solids (TSS)', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(df_quality['datetime'], df_quality[flavour_col], marker='s', linewidth=2, markersize=8, color='red')
axes[1, 1].set_ylabel('Flavor Score (0-100)', fontsize=12)
axes[1, 1].set_xlabel('Date', fontsize=12)
axes[1, 1].set_title('Flavor Quality', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_PATH / 'fig8_yield_quality.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Saved: fig8_yield_quality.png")

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE - SUMMARY")
print("=" * 80)

eda_summary = {
    "completion_date": datetime.now().isoformat(),
    
    "statistical_profile": {
        "climate_variables": len(climate_vars),
        "resource_variables": len(resource_vars),
        "weather_variables": len(weather_vars)
    },
    
    "economic_analysis": economic_summary,
    
    "yield_analysis": {
        "total_yield_kg_m2": float((df_production['ProdA'] + df_production['ProdB']).sum()),
        "class_a_kg_m2": float(df_production['ProdA'].sum()),
        "class_b_kg_m2": float(df_production['ProdB'].sum()),
        "energy_per_kg_MJ": float(energy_per_kg),
        "cost_per_kg_eur": float(cost_per_kg),
        "net_profit_eur_m2": float(net_profit)
    },
    
    "key_correlations": {
        "heating_vs_temp": float(corr_matrix.loc['Heat_cons', 'Tair']),
        "elec_vs_PAR": float(corr_matrix.loc['ElecHigh', 'Tot_PAR']),
        "vent_vs_temp": float(corr_matrix.loc['VentLee', 'Tair'])
    },
    
    "figures_created": [
        "fig1_daily_energy_patterns.png",
        "fig2_climate_timeseries.png",
        "fig3_seasonal_trends.png",
        "fig4_diurnal_patterns.png",
        "fig5_correlation_heatmap.png",
        "fig6_scatter_relationships.png",
        "fig7_economic_carbon_analysis.png",
        "fig8_yield_quality.png"
    ]
}

with open(OUTPUT_PATH / 'phase2_eda_summary.json', 'w') as f:
    json.dump(eda_summary, f, indent=2)

print(f"\n✅ Figures created: {len(eda_summary['figures_created'])}")
print(f"✅ Economic analysis: €{economic_summary['total_cost_eur_m2']:.2f}/m² total cost")
print(f"✅ Carbon footprint: {economic_summary['total_carbon_kg_co2e_m2']:.2f} kg CO₂e/m²")
print(f"✅ Yield: {eda_summary['yield_analysis']['total_yield_kg_m2']:.2f} kg/m²")
print(f"✅ Net profit: €{eda_summary['yield_analysis']['net_profit_eur_m2']:.2f}/m²")

print(f"\n📁 All results saved to: {OUTPUT_PATH}")
print(f"📁 All figures saved to: {FIGURES_PATH}")

print("\n" + "=" * 80)
print("✨ EDA COMPLETE - READY FOR PUBLICATION")
print("=" * 80)
