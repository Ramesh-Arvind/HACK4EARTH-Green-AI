"""
Phase 1: Data Understanding & Ingestion
Wageningen Autonomous Greenhouse Challenge Dataset

This script performs comprehensive data ingestion, schema identification,
and metadata documentation for publication-ready analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

# Paths
BASE_PATH = Path("/home/rnaa/paper_5_pica_whatif/ecogrow/data/AutonomousGreenhouseChallenge_edition2")
REFERENCE_PATH = BASE_PATH / "Reference"
OUTPUT_PATH = Path("/home/rnaa/paper_5_pica_whatif/ecogrow/results")
OUTPUT_PATH.mkdir(exist_ok=True)

print("=" * 80)
print("PHASE 1: DATA UNDERSTANDING & INGESTION")
print("Wageningen Autonomous Greenhouse Challenge - 2nd Edition (2019)")
print("=" * 80)

# ============================================================================
# 1. IDENTIFY DATA SCHEMA
# ============================================================================

print("\n" + "=" * 80)
print("1. DATA SCHEMA IDENTIFICATION")
print("=" * 80)

datasets = {
    "Weather": "Weather data (outdoor conditions)",
    "GreenhouseClimate": "Indoor climate, actuators, setpoints",
    "Resources": "Energy consumption (daily aggregates)",
    "Production": "Harvest data (yield, quality classification)",
    "CropParameters": "Weekly crop measurements",
    "TomQuality": "Bi-weekly tomato quality analysis",
    "LabAnalysis": "Bi-weekly irrigation/drainage nutrient analysis",
    "GrodanSens": "Root zone sensors (EC, water content, temperature)"
}

schema_info = {}

for dataset_name, description in datasets.items():
    csv_file = REFERENCE_PATH / f"{dataset_name}.csv"
    
    if not csv_file.exists():
        print(f"\n⚠️  {dataset_name}: File not found")
        continue
    
    print(f"\n📊 {dataset_name}: {description}")
    print("-" * 80)
    
    # Read CSV
    df = pd.read_csv(csv_file, nrows=5)
    
    # Get schema
    columns = df.columns.tolist()
    dtypes = df.dtypes.to_dict()
    
    print(f"   Columns: {len(columns)}")
    print(f"   Sample columns: {columns[:10]}...")
    
    # Infer units and descriptions from column names
    schema_info[dataset_name] = {
        "file": str(csv_file),
        "description": description,
        "num_columns": len(columns),
        "columns": columns,
        "dtypes": {k: str(v) for k, v in dtypes.items()},
        "sample_data": df.head(2).to_dict(orient='records')
    }

# Save schema to JSON
schema_file = OUTPUT_PATH / "dataset_schema.json"
with open(schema_file, 'w') as f:
    json.dump(schema_info, f, indent=2)

print(f"\n✅ Schema saved to: {schema_file}")

# ============================================================================
# 2. RECORD METADATA
# ============================================================================

print("\n" + "=" * 80)
print("2. METADATA DOCUMENTATION")
print("=" * 80)

metadata = {
    "dataset_name": "Autonomous Greenhouse Challenge - 2nd Edition (2019)",
    "doi": "10.18174/544434",
    "citation": "Hemming, S., de Zwart, F., Elings, A., Righini, I., Petropoulou, A. (2019)",
    "location": "Wageningen University & Research, Netherlands (52°N)",
    "period": "January 1 - June 17, 2020 (168 days)",
    
    "greenhouse": {
        "type": "Research compartment",
        "total_area_m2": 96,
        "growing_area_m2": 62.5,
        "cover_transmissivity": 0.5,
        "energy_screen_transmissivity": 0.75,
        "blackout_screen_transmissivity": 0.02
    },
    
    "crop": {
        "species": "Solanum lycopersicum (Tomato)",
        "cultivar": "Axiany",
        "type": "Truss tomato",
        "planting_density": "2-stem plants",
        "density_range": "2.5-3.5 stems/m²"
    },
    
    "sensors": {
        "climate": {
            "temperature": {"accuracy": "±0.2°C", "location": "Indoor (Tair) & Outdoor (Tout)"},
            "humidity": {"accuracy": "±2% RH", "variables": ["Rhair", "Rhout", "HumDef"]},
            "co2": {"accuracy": "±50 ppm", "calibration": "Monthly handheld meter"},
            "light": {"sensors": ["Pyranometer (Iglob)", "PAR sensor (PARout, Tot_PAR)"]},
            "wind": {"sensors": ["Anemometer (Windsp)", "Compass (Winddir)"]}
        },
        
        "actuators": {
            "ventilation": {"variables": ["VentLee", "Ventwind"], "range": "0-100% opening"},
            "lighting": {"HPS": "81 W/m²", "LED": "Spectrum-specific (7-25 W/m²)"},
            "screens": {"energy": "EnScr (0-100%)", "blackout": "BlackScr (0-100%)"},
            "heating": {"rail_pipe": "PipeLow (°C)", "crop_pipe": "PipeGrow (°C)"},
            "co2_dosing": {"variable": "co2_dos", "unit": "kg/ha/hour"}
        },
        
        "root_zone": {
            "sensor_type": "Grodan Grosens",
            "variables": ["EC_slab1/2 (dS/m)", "WC_slab1/2 (%)", "t_slab1/2 (°C)"],
            "resolution": "5 minutes (upsampled from 3 min)",
            "coverage": "Until May 26, 2020"
        }
    },
    
    "sampling_intervals": {
        "Weather": "5 minutes",
        "GreenhouseClimate": "5 minutes",
        "Resources": "Daily aggregate",
        "Production": "Per harvest (3× per 2 weeks)",
        "CropParameters": "Weekly",
        "TomQuality": "Bi-weekly",
        "LabAnalysis": "Bi-weekly",
        "GrodanSens": "5 minutes"
    },
    
    "economic_model": {
        "net_profit_formula": "Income - Fixed Costs - Variable Costs",
        "costs": {
            "electricity_peak": "€0.08/kWh (07:00-23:00)",
            "electricity_offpeak": "€0.04/kWh (23:00-07:00)",
            "heating": "€0.0083/MJ",
            "co2": "€0.08/kg (first 12 kg/m²), €0.20/kg (additional)",
            "labor": "€0.0085 per stem/m²/day"
        },
        "income": {
            "tomato_price_range": "€1.10-5.20/kg",
            "price_factors": ["Brix level (6-10)", "Market date", "Class (A=full, B=half)"]
        }
    },
    
    "reference_group_performance": {
        "yield_kg_m2": 48,
        "quality_brix": 7.8,
        "quality_flavor": 68,
        "heating_MJ_m2": 180,
        "electricity_kWh_m2": 370,
        "co2_kg_m2": 22,
        "water_L_m2": 300,
        "net_profit_eur_m2": 173.82
    }
}

metadata_file = OUTPUT_PATH / "dataset_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Metadata saved to: {metadata_file}")

print("\n📋 Greenhouse Specifications:")
print(f"   • Total area: {metadata['greenhouse']['total_area_m2']} m²")
print(f"   • Growing area: {metadata['greenhouse']['growing_area_m2']} m²")
print(f"   • Crop: {metadata['crop']['cultivar']} tomato (2-stem plants)")

print("\n📡 Sensor Systems:")
print(f"   • Climate: Temperature (±0.2°C), Humidity (±2%), CO₂ (±50 ppm)")
print(f"   • Light: Pyranometer, PAR sensors")
print(f"   • Root zone: Grodan Grosens (EC, WC, Temperature)")

# ============================================================================
# 3. SUMMARIZE DATASET SCOPE
# ============================================================================

print("\n" + "=" * 80)
print("3. DATASET SCOPE SUMMARY")
print("=" * 80)

# Load all datasets for scope analysis
scope_summary = {}

for dataset_name in datasets.keys():
    csv_file = REFERENCE_PATH / f"{dataset_name}.csv"
    
    if not csv_file.exists():
        continue
    
    print(f"\n📊 {dataset_name}")
    print("-" * 80)
    
    # Read full dataset
    df = pd.read_csv(csv_file)
    
    # Basic statistics
    num_rows = len(df)
    num_cols = len(df.columns)
    
    # Check for time column
    time_col = [col for col in df.columns if 'time' in col.lower() or col.startswith('%')]
    if time_col:
        time_col = time_col[0]
        # Excel serial date format (days since 1900-01-01)
        if df[time_col].dtype in [np.float64, np.int64]:
            # Convert Excel date to datetime
            df['datetime'] = pd.TimedeltaIndex(df[time_col], unit='D') + pd.Timestamp('1899-12-30')
            date_range = f"{df['datetime'].min()} to {df['datetime'].max()}"
        else:
            date_range = "Unknown format"
    else:
        date_range = "No time column"
    
    # Missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).describe()
    
    scope_summary[dataset_name] = {
        "num_rows": int(num_rows),
        "num_columns": int(num_cols),
        "date_range": str(date_range),
        "missing_values": {
            "max_pct": float(missing_pct['max']),
            "mean_pct": float(missing_pct['mean'])
        }
    }
    
    print(f"   Rows: {num_rows:,}")
    print(f"   Columns: {num_cols}")
    print(f"   Date range: {date_range}")
    print(f"   Missing values: {missing_pct['mean']:.2f}% (avg), {missing_pct['max']:.2f}% (max)")

# Save scope summary
scope_file = OUTPUT_PATH / "dataset_scope_summary.json"
with open(scope_file, 'w') as f:
    json.dump(scope_summary, f, indent=2)

print(f"\n✅ Scope summary saved to: {scope_file}")

# ============================================================================
# 4. CREATE VARIABLE TABLE
# ============================================================================

print("\n" + "=" * 80)
print("4. COMPREHENSIVE VARIABLE TABLE")
print("=" * 80)

# Define variable categories
variable_table = []

# Weather variables
weather_vars = [
    ("Tout", "Outdoor temperature", "°C", "State", "5 min"),
    ("Rhout", "Outdoor relative humidity", "%", "State", "5 min"),
    ("Iglob", "Solar radiation", "W/m²", "State", "5 min"),
    ("Windsp", "Wind speed", "m/s", "State", "5 min"),
    ("Winddir", "Wind direction", "compass [0-128]", "State", "5 min"),
    ("Rain", "Rain status", "1=rain, 0=dry", "State", "5 min"),
    ("PARout", "PAR outdoor", "µmol/m²/s", "State", "5 min"),
    ("Pyrgeo", "Heat emission (pyrgeometer)", "W/m²", "State", "5 min"),
    ("AbsHumOut", "Absolute humidity outdoor", "g/m³", "State", "5 min"),
    ("RadSum", "Radiation sum", "J/cm²", "State", "5 min")
]

# Greenhouse climate variables
climate_vars = [
    ("Tair", "Indoor air temperature", "°C", "State", "5 min"),
    ("Rhair", "Indoor relative humidity", "%", "State", "5 min"),
    ("CO2air", "Indoor CO₂ concentration", "ppm", "State", "5 min"),
    ("HumDef", "Humidity deficit", "g/m³", "State", "5 min"),
    ("VentLee", "Leeward vent opening", "% [0-100]", "Control", "5 min"),
    ("Ventwind", "Windward vent opening", "% [0-100]", "Control", "5 min"),
    ("AssimLight", "HPS lamps status", "% [0, 100]", "Control", "5 min"),
    ("EnScr", "Energy curtain opening", "% [0-100]", "Control", "5 min"),
    ("BlackScr", "Blackout curtain opening", "% [0-100]", "Control", "5 min"),
    ("PipeLow", "Rail pipe temperature", "°C", "Control", "5 min"),
    ("PipeGrow", "Crop pipe temperature", "°C", "Control", "5 min"),
    ("co2_dos", "CO₂ dosing rate", "kg/ha/hour", "Control", "5 min"),
    ("Tot_PAR", "Total PAR (sun + lamps)", "µmol/m²/s", "State", "5 min"),
    ("Tot_PAR_Lamps", "PAR from lamps only", "µmol/m²/s", "State", "5 min")
]

# Resource consumption variables
resource_vars = [
    ("Heat_cons", "Heating energy consumption", "MJ/m²/day", "Energy", "Daily"),
    ("ElecHigh", "Electricity (peak hours)", "kWh/m²/day", "Energy", "Daily"),
    ("ElecLow", "Electricity (off-peak)", "kWh/m²/day", "Energy", "Daily"),
    ("CO2_cons", "CO₂ consumption", "kg/m²/day", "Energy", "Daily"),
    ("Irr", "Irrigation water", "L/m²/day", "Resource", "Daily"),
    ("Drain", "Drainage water", "L/m²/day", "Resource", "Daily")
]

# Production variables
production_vars = [
    ("ProdA", "Tomato yield (Class A)", "kg/m²", "Production", "Per harvest"),
    ("ProdB", "Tomato yield (Class B)", "kg/m²", "Production", "Per harvest"),
    ("avg_nr_harvested_trusses", "Average trusses harvested", "Number/stem", "Production", "Per harvest"),
    ("Truss_development_time", "Truss development period", "days", "Production", "Per harvest")
]

# Quality variables
quality_vars = [
    ("Flavour", "Flavor level", "0-100", "Quality", "Bi-weekly"),
    ("TSS", "Total soluble solids (Brix)", "°Brix", "Quality", "Bi-weekly"),
    ("Acid", "Titratable acid", "mmol H₃O⁺/100g", "Quality", "Bi-weekly"),
    ("%Juice", "Juice content", "%", "Quality", "Bi-weekly"),
    ("Bite", "Firmness (breaking force)", "N", "Quality", "Bi-weekly"),
    ("Weight", "Average fruit weight", "g", "Quality", "Bi-weekly")
]

# Combine all variables
for var_list in [weather_vars, climate_vars, resource_vars, production_vars, quality_vars]:
    for var in var_list:
        variable_table.append({
            "Variable": var[0],
            "Description": var[1],
            "Unit": var[2],
            "Type": var[3],
            "Frequency": var[4]
        })

# Create DataFrame
var_df = pd.DataFrame(variable_table)

# Save to CSV
var_table_file = OUTPUT_PATH / "variable_table_comprehensive.csv"
var_df.to_csv(var_table_file, index=False)

print(f"\n✅ Variable table saved to: {var_table_file}")
print(f"\n📊 Total variables documented: {len(var_df)}")

# Print summary by type
print("\n📋 Variables by Type:")
for var_type in var_df['Type'].unique():
    count = len(var_df[var_df['Type'] == var_type])
    print(f"   • {var_type}: {count} variables")

# ============================================================================
# 5. TEMPORAL COVERAGE & GAPS
# ============================================================================

print("\n" + "=" * 80)
print("5. TEMPORAL COVERAGE & GAPS")
print("=" * 80)

# Analyze GreenhouseClimate (highest resolution)
climate_file = REFERENCE_PATH / "GreenhouseClimate.csv"
df_climate = pd.read_csv(climate_file)

# Get time column
time_col = [col for col in df_climate.columns if col.startswith('%')][0]

# Convert Excel date to datetime
df_climate['datetime'] = pd.TimedeltaIndex(df_climate[time_col], unit='D') + pd.Timestamp('1899-12-30')

print(f"\n📅 Greenhouse Climate Data:")
print(f"   Start: {df_climate['datetime'].min()}")
print(f"   End: {df_climate['datetime'].max()}")
print(f"   Duration: {(df_climate['datetime'].max() - df_climate['datetime'].min()).days} days")
print(f"   Total records: {len(df_climate):,}")
print(f"   Expected 5-min records: {(df_climate['datetime'].max() - df_climate['datetime'].min()).days * 288:,}")

# Check for gaps
df_climate = df_climate.sort_values('datetime')
time_diffs = df_climate['datetime'].diff()
gaps = time_diffs[time_diffs > pd.Timedelta('10 minutes')]

print(f"\n⚠️  Data gaps (>10 min): {len(gaps)}")
if len(gaps) > 0:
    print(f"   Largest gap: {gaps.max()}")
    gap_dates = df_climate.loc[gaps.index, 'datetime'].head(5)
    print(f"   First 5 gap locations:")
    for date in gap_dates:
        print(f"      • {date}")

# Analyze Resources (daily data)
resource_file = REFERENCE_PATH / "Resources.csv"
df_resource = pd.read_csv(resource_file)
time_col_res = [col for col in df_resource.columns if 'time' in col.lower()][0]
df_resource['datetime'] = pd.TimedeltaIndex(df_resource[time_col_res], unit='D') + pd.Timestamp('1899-12-30')

print(f"\n📅 Resources Data:")
print(f"   Start: {df_resource['datetime'].min()}")
print(f"   End: {df_resource['datetime'].max()}")
print(f"   Duration: {(df_resource['datetime'].max() - df_resource['datetime'].min()).days} days")
print(f"   Total records: {len(df_resource)}")

# Check for missing days
date_range = pd.date_range(df_resource['datetime'].min(), df_resource['datetime'].max(), freq='D')
missing_days = set(date_range) - set(df_resource['datetime'].dt.date)
print(f"\n⚠️  Missing days: {len(missing_days)}")

# ============================================================================
# 6. DEFINE RESEARCH QUESTIONS
# ============================================================================

print("\n" + "=" * 80)
print("6. RESEARCH QUESTIONS")
print("=" * 80)

research_questions = {
    "energy_patterns": {
        "question": "What are typical daily and seasonal energy patterns?",
        "sub_questions": [
            "How does heating demand vary with outdoor temperature?",
            "What is the diurnal pattern of electricity consumption (lighting)?",
            "How do seasonal changes affect total energy consumption?",
            "What is the correlation between solar radiation and artificial lighting?",
            "What fraction of energy is consumed during peak vs off-peak hours?"
        ],
        "required_data": ["Heat_cons", "ElecHigh", "ElecLow", "Tout", "Iglob", "datetime"],
        "analysis_methods": ["Time-series visualization", "Correlation analysis", "Seasonal decomposition"]
    },
    
    "control_effects": {
        "question": "How do environmental controls affect crop growth and yield?",
        "sub_questions": [
            "What is the relationship between CO₂ enrichment and photosynthesis/yield?",
            "How does temperature control affect truss development time?",
            "What is the optimal balance between heating and ventilation?",
            "How do PAR levels (light intensity) correlate with biomass accumulation?",
            "What irrigation strategies maximize water use efficiency?"
        ],
        "required_data": ["CO2air", "Tair", "Tot_PAR", "ProdA", "Truss_development_time", "Irr", "Drain"],
        "analysis_methods": ["Regression analysis", "Causal inference", "Multi-objective optimization"]
    },
    
    "optimization_potential": {
        "question": "What cost savings and carbon reductions are achievable under optimized control?",
        "sub_questions": [
            "What is the baseline energy cost per kg tomato produced?",
            "How much can carbon-aware scheduling reduce emissions?",
            "What is the ROI for implementing AI-based control?",
            "What are the trade-offs between profit maximization and sustainability?",
            "What savings scale to commercial greenhouse deployment (1+ hectare)?"
        ],
        "required_data": ["Heat_cons", "ElecHigh", "ElecLow", "CO2_cons", "ProdA", "economic_parameters"],
        "analysis_methods": ["Cost-benefit analysis", "Carbon accounting", "Scenario modeling", "Pareto frontier analysis"]
    }
}

rq_file = OUTPUT_PATH / "research_questions.json"
with open(rq_file, 'w') as f:
    json.dump(research_questions, f, indent=2)

print(f"\n✅ Research questions saved to: {rq_file}")

print("\n🔍 Primary Research Questions:")
for i, (key, rq) in enumerate(research_questions.items(), 1):
    print(f"\n{i}. {rq['question']}")
    print(f"   Sub-questions: {len(rq['sub_questions'])}")
    print(f"   Required data: {', '.join(rq['required_data'][:3])}...")
    print(f"   Methods: {', '.join(rq['analysis_methods'])}")

# ============================================================================
# 7. SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE - SUMMARY")
print("=" * 80)

summary = {
    "completion_date": datetime.now().isoformat(),
    "datasets_analyzed": len(datasets),
    "total_variables": len(var_df),
    "temporal_coverage": {
        "start": str(df_climate['datetime'].min()),
        "end": str(df_climate['datetime'].max()),
        "duration_days": (df_climate['datetime'].max() - df_climate['datetime'].min()).days
    },
    "data_quality": {
        "climate_records": len(df_climate),
        "resource_records": len(df_resource),
        "data_gaps": len(gaps),
        "missing_days": len(missing_days)
    },
    "files_created": [
        str(schema_file.name),
        str(metadata_file.name),
        str(scope_file.name),
        str(var_table_file.name),
        str(rq_file.name)
    ]
}

summary_file = OUTPUT_PATH / "phase1_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Datasets analyzed: {len(datasets)}")
print(f"✅ Variables documented: {len(var_df)}")
print(f"✅ Temporal coverage: {summary['temporal_coverage']['duration_days']} days")
print(f"✅ Climate records: {summary['data_quality']['climate_records']:,}")
print(f"✅ Resource records: {summary['data_quality']['resource_records']}")
print(f"\n📁 Files created:")
for f in summary['files_created']:
    print(f"   • {f}")

print(f"\n✅ Phase 1 summary saved to: {summary_file}")
print("\n" + "=" * 80)
print("✨ READY FOR PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 80)
