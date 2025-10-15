# EcoGrow Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Installation

```bash
cd ecogrow
pip install -r requirements.txt
```

### Step 2: Run Demo

```bash
python demo.py
```

This will show you:
- ✅ Track A: Model optimization results (67% energy reduction)
- ✅ Track B: Greenhouse optimization results (35% energy reduction)

### Step 3: Run Benchmarks

#### Track A - Measure Model Carbon Footprint

```bash
# Baseline model
python src/baseline_benchmark.py --config baseline --num_runs 100

# Optimized model
python src/baseline_benchmark.py --config optimized --num_runs 100
```

Results saved to `results/evidence.csv`

#### Track B - Optimize Greenhouse Energy

```bash
python src/greenhouse_optimizer.py
```

Results saved to `results/greenhouse_optimization_results.csv`

## 📊 View Results

```bash
# View carbon footprint data
cat results/evidence.csv

# View greenhouse optimization
cat results/greenhouse_optimization_results.csv
```

## 🔧 Advanced Usage

### Carbon-Aware Training

```python
from src.carbon_aware_trainer import CarbonAwareScheduler

scheduler = CarbonAwareScheduler(region='DE')

# Check current carbon intensity
intensity = scheduler.get_carbon_intensity()
print(f"Current grid carbon: {intensity} g CO₂/kWh")

# Schedule training during clean energy window
scheduler.schedule_training(
    train_function=your_training_fn,
    max_wait_hours=12,
    threshold=350  # g CO₂/kWh
)
```

### Model Quantization

```python
from src.optimized_model import QuantizedPICAModel

# Load your model
model = YourModel()

# Quantize
optimizer = QuantizedPICAModel(model)
quantized_model = optimizer.quantize_model()

# Measure performance
results = optimizer.measure_optimized(test_data, num_runs=100)
print(f"Energy saved: {results['energy_kwh']:.3f} kWh")
```

### Greenhouse Optimization

```python
from src.greenhouse_optimizer import GreenhouseEnergyOptimizer

optimizer = GreenhouseEnergyOptimizer()

# Define states
initial_state = torch.tensor([20.0, 75.0, 600.0, 10.0])  # [T, H, CO2, B]
target_state = torch.tensor([22.0, 75.0, 800.0, 11.0])

# Optimize
result = optimizer.optimize_controls(
    initial_state=initial_state,
    target_state=target_state,
    horizon=288  # 24 hours @ 5-min intervals
)

print(f"Energy cost: {result['energy_cost']:.2f} kWh")
```

## 📚 Documentation

- **[FOOTPRINT.md](FOOTPRINT.md)** - Track A carbon analysis
- **[IMPACT.md](IMPACT.md)** - Track B sustainability impact
- **[README.md](README.md)** - Full documentation

## 🆘 Troubleshooting

### Model not found?
Make sure you have trained model in `../pica_framework/models/trained_models/trained_model.pth`

### Import errors?
Check that you're running from the `ecogrow/` directory and have installed all requirements.

### Low measurements?
This is expected! The system is optimized for efficiency. Check that results show 60-70% reduction compared to baseline.

## 🎯 Next Steps

1. **Customize:** Modify parameters in source files
2. **Integrate:** Connect to your greenhouse HVAC system
3. **Scale:** Deploy to multiple greenhouses
4. **Monitor:** Track long-term energy and carbon savings

## 📧 Need Help?

- Check documentation in `FOOTPRINT.md` and `IMPACT.md`
- Open an issue on GitHub
- Contact: ecogrow@example.com

---

**Happy optimizing! 🌱**
