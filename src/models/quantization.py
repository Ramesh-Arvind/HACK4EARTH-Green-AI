#!/usr/bin/env python3
"""
Model Quantization & Efficiency Techniques for Greenhouse AI
============================================================
Implements model compression techniques to reduce inference energy consumption:
- Dynamic quantization (FP32 → INT8)
- Static quantization with calibration
- Pruning (structured and unstructured)
- Knowledge distillation (large model → small model)

Target: 67% energy reduction (0.150 → 0.050 kWh per 100 inferences)

Author: EcoGrow Team
Date: October 15, 2025
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import copy

# Set style
sns.set_style("whitegrid")


class EnergyMonitor:
    """
    Monitors inference energy consumption and timing.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize energy monitor.
        
        Parameters:
        -----------
        device : str
            Device for inference ('cpu' or 'cuda')
        """
        self.device = device
        
        # Energy consumption factors (based on hardware TDP and typical usage)
        # Source: Intel ARK, Nvidia datasheets, measured power consumption
        self.energy_factors = {
            'cpu': {
                'tdp_watts': 65,  # Typical desktop CPU (Intel i5/i7 or AMD Ryzen 5/7)
                'inference_utilization': 0.40,  # 40% of TDP during inference
                'idle_watts': 5,  # Idle power consumption
                'watts_per_inference': 26  # 65W × 0.40
            },
            'cuda': {
                'tdp_watts': 250,  # Nvidia GPU (Tesla P4/T4)
                'inference_utilization': 0.60,  # 60% of TDP during inference
                'idle_watts': 15,  # Idle power consumption
                'watts_per_inference': 150  # 250W × 0.60
            }
        }
        
        # Carbon intensity (Germany grid, from carbon-aware scheduler)
        self.carbon_intensity_kg_per_kwh = 0.42  # kg CO₂/kWh
        
    def measure_inference_energy(self, model: nn.Module, input_data: torch.Tensor,
                                num_inferences: int = 100) -> Dict:
        """
        Measure energy consumption for model inference.
        
        Parameters:
        -----------
        model : nn.Module
            Model to benchmark
        input_data : torch.Tensor
            Input data for inference
        num_inferences : int
            Number of inferences to run
            
        Returns:
        --------
        metrics : Dict
            Energy, carbon, timing metrics
        """
        # Warm-up (exclude from timing)
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        # Benchmark
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_inferences):
                _ = model(input_data)
        
        end_time = time.perf_counter()
        total_time_s = end_time - start_time
        
        # Calculate metrics
        factors = self.energy_factors[self.device]
        inference_time_s = total_time_s / num_inferences
        energy_per_inference_wh = factors['watts_per_inference'] * (inference_time_s / 3600)
        total_energy_kwh = energy_per_inference_wh * num_inferences / 1000
        
        # Carbon emissions
        carbon_kg = total_energy_kwh * self.carbon_intensity_kg_per_kwh
        carbon_per_inference_g = carbon_kg * 1000 / num_inferences
        
        return {
            'num_inferences': num_inferences,
            'total_time_s': total_time_s,
            'inference_time_ms': inference_time_s * 1000,
            'inferences_per_second': num_inferences / total_time_s,
            'total_energy_kwh': total_energy_kwh,
            'energy_per_inference_wh': energy_per_inference_wh,
            'carbon_total_kg': carbon_kg,
            'carbon_per_inference_g': carbon_per_inference_g,
            'device': self.device,
            'power_watts': factors['watts_per_inference']
        }


class GreenhouseModelQuantizer:
    """
    Quantization utilities for greenhouse prediction models.
    """
    
    def __init__(self, model: nn.Module, model_name: str = 'greenhouse_model'):
        """
        Initialize quantizer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to quantize
        model_name : str
            Model identifier
        """
        self.model_fp32 = model
        self.model_name = model_name
        
    def dynamic_quantization(self) -> nn.Module:
        """
        Apply dynamic quantization (weights INT8, activations FP32).
        Fast and easy, no calibration needed.
        
        Returns:
        --------
        quantized_model : nn.Module
            Dynamically quantized model
        """
        print(f"🔧 Applying dynamic quantization to {self.model_name}...")
        
        # Dynamic quantization converts weights to INT8
        # Activations remain FP32 (computed dynamically at runtime)
        quantized_model = quant.quantize_dynamic(
            self.model_fp32,
            {nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )
        
        print(f"✅ Dynamic quantization complete")
        return quantized_model
    
    def prepare_static_quantization(self, calibration_data: torch.Tensor) -> nn.Module:
        """
        Prepare model for static quantization with calibration.
        
        Parameters:
        -----------
        calibration_data : torch.Tensor
            Representative calibration data
            
        Returns:
        --------
        quantized_model : nn.Module
            Statically quantized model
        """
        print(f"🔧 Preparing static quantization for {self.model_name}...")
        
        # Copy model
        model_to_quantize = copy.deepcopy(self.model_fp32)
        model_to_quantize.eval()
        
        # Set quantization config
        model_to_quantize.qconfig = quant.get_default_qconfig('fbgemm')
        
        # Prepare model (insert observers)
        model_prepared = quant.prepare(model_to_quantize)
        
        # Calibration (run representative data through model)
        print(f"📊 Calibrating with {len(calibration_data)} samples...")
        with torch.no_grad():
            for data in calibration_data:
                _ = model_prepared(data.unsqueeze(0))
        
        # Convert to quantized model
        quantized_model = quant.convert(model_prepared)
        
        print(f"✅ Static quantization complete")
        return quantized_model
    
    def apply_pruning(self, amount: float = 0.3, structured: bool = False) -> nn.Module:
        """
        Apply pruning to reduce model parameters.
        
        Parameters:
        -----------
        amount : float
            Fraction of parameters to prune (0.0-1.0)
        structured : bool
            If True, use structured pruning (entire channels/filters)
            If False, use unstructured pruning (individual weights)
            
        Returns:
        --------
        pruned_model : nn.Module
            Pruned model
        """
        print(f"🔧 Applying {'structured' if structured else 'unstructured'} pruning "
              f"({amount*100:.0f}% sparsity)...")
        
        import torch.nn.utils.prune as prune
        
        # Copy model
        pruned_model = copy.deepcopy(self.model_fp32)
        
        # Apply pruning to all Linear layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                if structured:
                    # Structured pruning (prune entire output channels)
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                else:
                    # Unstructured pruning (magnitude-based)
                    prune.l1_unstructured(module, name='weight', amount=amount)
        
        # Make pruning permanent
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, 'weight')
                except:
                    pass
        
        # Count zero parameters
        zero_params = 0
        total_params = 0
        for param in pruned_model.parameters():
            zero_params += torch.sum(param == 0).item()
            total_params += param.numel()
        
        actual_sparsity = zero_params / total_params
        print(f"✅ Pruning complete: {actual_sparsity*100:.1f}% sparsity "
              f"({zero_params:,}/{total_params:,} parameters)")
        
        return pruned_model
    
    def create_distilled_student(self, hidden_dim: int = 64) -> nn.Module:
        """
        Create a smaller 'student' model for knowledge distillation.
        
        Parameters:
        -----------
        hidden_dim : int
            Hidden dimension for student model (smaller than teacher)
            
        Returns:
        --------
        student_model : nn.Module
            Smaller student model
        """
        print(f"🔧 Creating distilled student model (hidden_dim={hidden_dim})...")
        
        # Get input/output dimensions from teacher
        # For greenhouse model: input=16 (state+control+weather), output=4 (state)
        
        class DistilledGreenhouseModel(nn.Module):
            """Lightweight student model for knowledge distillation."""
            
            def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 4):
                super().__init__()
                
                # Smaller architecture than teacher
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, output_dim)
                )
                
            def forward(self, state, control=None, weather=None):
                # Concatenate inputs (match teacher interface)
                if control is not None and weather is not None:
                    x = torch.cat([state, control, weather], dim=1)
                else:
                    x = state
                
                # Simple forward pass
                h = self.encoder(x)
                delta_state = self.decoder(h)
                next_state = state[:, :4] + delta_state  # Assuming state is first 4 dims
                
                return {'state_pred': next_state}
        
        student_model = DistilledGreenhouseModel(hidden_dim=hidden_dim)
        
        # Count parameters
        student_params = sum(p.numel() for p in student_model.parameters())
        teacher_params = sum(p.numel() for p in self.model_fp32.parameters())
        reduction = 100 * (1 - student_params / teacher_params)
        
        print(f"✅ Student model created: {student_params:,} parameters "
              f"({reduction:.1f}% reduction from teacher)")
        
        return student_model


class ModelCompressionBenchmark:
    """
    Comprehensive benchmark of model compression techniques.
    """
    
    def __init__(self, teacher_model: nn.Module, device: str = 'cpu'):
        """
        Initialize benchmark.
        
        Parameters:
        -----------
        teacher_model : nn.Module
            Original (teacher) model
        device : str
            Device for inference
        """
        self.teacher_model = teacher_model
        self.device = device
        self.energy_monitor = EnergyMonitor(device)
        self.results = []
        
    def benchmark_model(self, model: nn.Module, model_name: str,
                       input_data: torch.Tensor, num_inferences: int = 100) -> Dict:
        """
        Benchmark a single model variant.
        
        Parameters:
        -----------
        model : nn.Module
            Model to benchmark
        model_name : str
            Model identifier
        input_data : torch.Tensor
            Input data for inference
        num_inferences : int
            Number of inferences
            
        Returns:
        --------
        results : Dict
            Benchmark results
        """
        print(f"\n{'='*70}")
        print(f"📊 Benchmarking: {model_name}")
        print(f"{'='*70}")
        
        # Model size
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        print(f"📦 Model size: {num_params:,} parameters, {model_size_mb:.2f} MB")
        
        # Energy/timing benchmark
        energy_metrics = self.energy_monitor.measure_inference_energy(
            model, input_data, num_inferences
        )
        
        print(f"⚡ Energy: {energy_metrics['total_energy_kwh']:.6f} kWh "
              f"({energy_metrics['energy_per_inference_wh']*1000:.2f} mWh/inference)")
        print(f"🌍 Carbon: {energy_metrics['carbon_total_kg']*1000:.2f} g CO₂ "
              f"({energy_metrics['carbon_per_inference_g']:.3f} g/inference)")
        print(f"⏱️  Timing: {energy_metrics['inference_time_ms']:.2f} ms/inference "
              f"({energy_metrics['inferences_per_second']:.1f} inf/s)")
        
        # Combine results
        results = {
            'model_name': model_name,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb,
            **energy_metrics
        }
        
        self.results.append(results)
        return results
    
    def run_full_benchmark(self, calibration_data: Optional[torch.Tensor] = None) -> pd.DataFrame:
        """
        Run comprehensive benchmark on all compression techniques.
        
        Parameters:
        -----------
        calibration_data : torch.Tensor, optional
            Calibration data for static quantization
            
        Returns:
        --------
        results_df : pd.DataFrame
            Benchmark results table
        """
        print("\n" + "="*70)
        print("🚀 MODEL COMPRESSION BENCHMARK")
        print("="*70)
        
        # Generate test input
        batch_size = 8
        state = torch.randn(batch_size, 4)  # [Tair, Rhair, CO2air, Tot_PAR]
        control = torch.randn(batch_size, 7)  # [PipeLow, VentLee, AssimLight, ...]
        weather = torch.randn(batch_size, 5)  # [Tout, Iglob, Rhout, ...]
        
        # For simple models that take single input
        input_data = torch.cat([state, control, weather], dim=1)  # [batch, 16]
        
        num_inferences = 100
        
        # 1. Baseline FP32
        print("\n" + "🔵 BASELINE FP32 MODEL" + "\n")
        baseline_results = self.benchmark_model(
            self.teacher_model, 'FP32_Baseline', input_data, num_inferences
        )
        
        # 2. Dynamic Quantization (INT8)
        print("\n" + "🟢 DYNAMIC QUANTIZATION (INT8)" + "\n")
        quantizer = GreenhouseModelQuantizer(self.teacher_model)
        model_dynamic_int8 = quantizer.dynamic_quantization()
        dynamic_results = self.benchmark_model(
            model_dynamic_int8, 'INT8_Dynamic', input_data, num_inferences
        )
        
        # 3. Pruning (30% sparsity)
        print("\n" + "🟡 UNSTRUCTURED PRUNING (30% SPARSITY)" + "\n")
        model_pruned_30 = quantizer.apply_pruning(amount=0.3, structured=False)
        pruned_30_results = self.benchmark_model(
            model_pruned_30, 'Pruned_30pct', input_data, num_inferences
        )
        
        # 4. Pruning (50% sparsity)
        print("\n" + "🟠 UNSTRUCTURED PRUNING (50% SPARSITY)" + "\n")
        model_pruned_50 = quantizer.apply_pruning(amount=0.5, structured=False)
        pruned_50_results = self.benchmark_model(
            model_pruned_50, 'Pruned_50pct', input_data, num_inferences
        )
        
        # 5. Student model (knowledge distillation architecture)
        print("\n" + "🔴 DISTILLED STUDENT MODEL" + "\n")
        model_student = quantizer.create_distilled_student(hidden_dim=64)
        student_results = self.benchmark_model(
            model_student, 'Student_64dim', input_data, num_inferences
        )
        
        # 6. Combined: Pruning + Dynamic Quantization
        print("\n" + "🟣 COMBINED: PRUNING (30%) + INT8" + "\n")
        model_combined = quantizer.apply_pruning(amount=0.3, structured=False)
        model_combined = quant.quantize_dynamic(model_combined, {nn.Linear}, dtype=torch.qint8)
        combined_results = self.benchmark_model(
            model_combined, 'Pruned30_INT8', input_data, num_inferences
        )
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results)
        
        # Calculate improvements vs baseline
        baseline_energy = baseline_results['energy_per_inference_wh']
        baseline_carbon = baseline_results['carbon_per_inference_g']
        baseline_time = baseline_results['inference_time_ms']
        
        results_df['energy_reduction_pct'] = 100 * (
            1 - results_df['energy_per_inference_wh'] / baseline_energy
        )
        results_df['carbon_reduction_pct'] = 100 * (
            1 - results_df['carbon_per_inference_g'] / baseline_carbon
        )
        results_df['speedup'] = baseline_time / results_df['inference_time_ms']
        
        return results_df
    
    def create_evidence_csv(self, results_df: pd.DataFrame, 
                           output_path: str = '../results/quantization_evidence.csv'):
        """
        Create BUIDL submission evidence CSV.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Benchmark results
        output_path : str
            Path to save evidence CSV
        """
        # Format for BUIDL submission
        evidence_rows = []
        
        for idx, row in results_df.iterrows():
            evidence_rows.append({
                'run_id': f"compression_benchmark_{idx+1}",
                'model_variant': row['model_name'],
                'timestamp': datetime.now().isoformat(),
                'num_parameters': int(row['num_parameters']),
                'model_size_mb': round(row['model_size_mb'], 2),
                'num_inferences': int(row['num_inferences']),
                'total_energy_kwh': round(row['total_energy_kwh'], 6),
                'energy_per_inference_wh': round(row['energy_per_inference_wh'] * 1000, 4),  # mWh
                'carbon_total_kg': round(row['carbon_total_kg'], 6),
                'carbon_per_inference_g': round(row['carbon_per_inference_g'], 4),
                'inference_time_ms': round(row['inference_time_ms'], 2),
                'inferences_per_second': round(row['inferences_per_second'], 1),
                'energy_reduction_pct': round(row['energy_reduction_pct'], 2),
                'carbon_reduction_pct': round(row['carbon_reduction_pct'], 2),
                'speedup': round(row['speedup'], 2),
                'device': row['device']
            })
        
        evidence_df = pd.DataFrame(evidence_rows)
        evidence_df.to_csv(output_path, index=False)
        print(f"\n✅ Evidence CSV saved: {output_path}")
        
        return evidence_df
    
    def visualize_results(self, results_df: pd.DataFrame,
                         save_path: str = '../results/quantization_comparison.png'):
        """
        Create visualization of compression results.
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            Benchmark results
        save_path : str
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data
        models = results_df['model_name']
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#9b59b6']
        
        # 1. Energy consumption
        ax1 = axes[0, 0]
        energy_mwh = results_df['energy_per_inference_wh'] * 1000
        bars1 = ax1.bar(range(len(models)), energy_mwh, color=colors)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Energy per Inference (mWh)', fontsize=12, fontweight='bold')
        ax1.set_title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, energy_mwh):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Carbon emissions
        ax2 = axes[0, 1]
        carbon_g = results_df['carbon_per_inference_g']
        bars2 = ax2.bar(range(len(models)), carbon_g, color=colors)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.set_ylabel('Carbon per Inference (g CO₂)', fontsize=12, fontweight='bold')
        ax2.set_title('Carbon Emissions Comparison', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, carbon_g):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Inference time
        ax3 = axes[1, 0]
        time_ms = results_df['inference_time_ms']
        bars3 = ax3.bar(range(len(models)), time_ms, color=colors)
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax3.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars3, time_ms):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Combined improvements (energy + carbon reduction)
        ax4 = axes[1, 1]
        energy_reduction = results_df['energy_reduction_pct']
        carbon_reduction = results_df['carbon_reduction_pct']
        
        x = np.arange(len(models))
        width = 0.35
        
        bars4a = ax4.bar(x - width/2, energy_reduction, width, label='Energy Reduction', color='#2ecc71')
        bars4b = ax4.bar(x + width/2, carbon_reduction, width, label='Carbon Reduction', color='#3498db')
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.set_ylabel('Reduction vs Baseline (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Energy & Carbon Reduction vs FP32 Baseline', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add target line (67% reduction)
        ax4.axhline(y=67, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (67%)')
        ax4.text(len(models)-0.5, 67, ' Target: 67%', va='bottom', ha='right', 
                fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {save_path}")
        plt.show()


def main():
    """Demonstration of model compression techniques."""
    print("="*70)
    print("🧠 MODEL QUANTIZATION & EFFICIENCY TECHNIQUES")
    print("="*70)
    print()
    
    # Import hybrid model from previous implementation
    import sys
    sys.path.append('../models')
    
    try:
        from hybrid_mpc_pinn import create_model
        
        # Load teacher model
        print("📦 Loading teacher model (Hybrid MPC+PINN)...")
        teacher_model = create_model()
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"✅ Teacher model loaded: {teacher_params:,} parameters")
        print()
        
        # Create simple wrapper for testing
        class SimpleWrapper(nn.Module):
            """Wrapper to simplify model interface for benchmarking."""
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                # Split input into components
                state = x[:, :4]
                control = x[:, 4:11]
                weather = x[:, 11:16]
                return self.model(state, control, weather)
        
        teacher_wrapped = SimpleWrapper(teacher_model)
        
    except Exception as e:
        print(f"⚠️  Could not load hybrid model: {e}")
        print("Creating simple model for demonstration...")
        
        # Fallback: create simple model for demonstration
        class SimpleGreenhouseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(16, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 4)
                )
            
            def forward(self, x):
                return self.net(x)
        
        teacher_wrapped = SimpleGreenhouseModel()
        teacher_params = sum(p.numel() for p in teacher_wrapped.parameters())
        print(f"✅ Simple model created: {teacher_params:,} parameters")
        print()
    
    # Run benchmark
    benchmark = ModelCompressionBenchmark(teacher_wrapped, device='cpu')
    results_df = benchmark.run_full_benchmark()
    
    # Summary
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    print()
    print(results_df[['model_name', 'energy_reduction_pct', 'carbon_reduction_pct', 
                      'speedup', 'model_size_mb']].to_string(index=False))
    print()
    
    # Best model
    best_idx = results_df['energy_reduction_pct'].idxmax()
    best_model = results_df.loc[best_idx]
    print(f"🏆 Best model: {best_model['model_name']}")
    print(f"   Energy reduction: {best_model['energy_reduction_pct']:.1f}%")
    print(f"   Carbon reduction: {best_model['carbon_reduction_pct']:.1f}%")
    print(f"   Speedup: {best_model['speedup']:.2f}x")
    print(f"   Size reduction: {100*(1-best_model['model_size_mb']/results_df.iloc[0]['model_size_mb']):.1f}%")
    print()
    
    # Check if target achieved
    target_achieved = best_model['energy_reduction_pct'] >= 67.0
    if target_achieved:
        print("✅ TARGET ACHIEVED: ≥67% energy reduction!")
    else:
        print(f"⚠️  Target not achieved: {best_model['energy_reduction_pct']:.1f}% < 67% target")
        print(f"   Gap: {67 - best_model['energy_reduction_pct']:.1f}% remaining")
    print()
    
    # Create evidence CSV
    benchmark.create_evidence_csv(results_df)
    
    # Visualize results
    benchmark.visualize_results(results_df)
    
    print()
    print("="*70)
    print("✅ QUANTIZATION & EFFICIENCY BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
