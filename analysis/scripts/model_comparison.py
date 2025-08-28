#!/usr/bin/env python3
"""
🏆 Model Comparison Module
Comprehensive statistical comparison of top performing models
"""

import os
import sys
import numpy as np
import subprocess
from pathlib import Path
import time
from typing import List, Dict, Tuple
import statistics

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Python executable path
PYTHON_PATH = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"

def get_model_environment_version(model_path: str) -> str:
    """Determine the environment version for a model based on filename."""
    filename = os.path.basename(model_path).lower()
    
    if 'stable' in filename or 'peak_performance' in filename:
        return 'stable'
    elif 'legacy' in filename or 'engineered' in filename:
        return 'legacy'
    else:
        return 'autonomous'

def parse_survival_rate_from_output(output: str) -> float:
    """Parse survival rate from evaluation output."""
    survival_rate = 0.0
    
    for line in output.split('\n'):
        line_lower = line.lower()
        
        # Look for various survival rate patterns
        if any(pattern in line_lower for pattern in ['survival rate:', 'survival:', 'survived:', 'survival =']):
            try:
                # Extract percentage from line
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        survival_rate = float(part.replace('%', ''))
                        break
                    # Also try numbers without % that might be survival rates
                    try:
                        num = float(part)
                        if 0 <= num <= 100:
                            survival_rate = num
                            break
                    except:
                        continue
                if survival_rate > 0:
                    break
            except:
                continue
        
        # Also look for episode completion patterns
        elif 'episodes completed' in line_lower or 'completed episodes' in line_lower:
            try:
                # Try to extract completion rate
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i < len(parts) - 1:
                        if parts[i + 1].isdigit():
                            completed = int(part)
                            total = int(parts[i + 1])
                            survival_rate = (completed / total) * 100
                            break
                if survival_rate > 0:
                    break
            except:
                continue
    
    # If no survival rate found, try to parse from final statistics
    if survival_rate == 0:
        lines = output.split('\n')
        for line in reversed(lines):  # Start from end
            if any(keyword in line.lower() for keyword in ['average', 'mean', 'final', 'overall']):
                try:
                    # Look for numbers that could be percentages
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            if 0 <= num <= 100:
                                survival_rate = num
                                break
                        except:
                            continue
                    if survival_rate > 0:
                        break
                except:
                    continue
    
    return survival_rate

def run_single_evaluation(model_path: str, episodes: int = 100) -> Dict:
    """Run a single evaluation session for a model."""
    try:
        print(f"    Running {episodes} episodes...")
        
        # Try direct import method first (avoids subprocess encoding issues)
        try:
            from train import evaluate_model_comprehensive
            
            # Temporarily redirect stdout to capture results
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                # Run evaluation directly
                try:
                    results = evaluate_model_comprehensive(model_path, num_episodes=episodes, render=False)
                except Exception as eval_error:
                    print(f"    Direct evaluation failed: {eval_error}")
                    raise ImportError("Fallback to subprocess")
            
            output = f.getvalue()
            
            # Extract survival rate from output
            survival_rate = parse_survival_rate_from_output(output)
            
            return {
                'model': os.path.basename(model_path),
                'episodes': episodes,
                'survival_rate': survival_rate,
                'success': True,
                'output': output
            }
            
        except (ImportError, Exception):
            # Fallback to subprocess method
            pass
        
        # Run the evaluation using train.py mode 5
        # Set environment variables to handle Unicode properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run([
            PYTHON_PATH, "train.py", "5", model_path, str(episodes)
        ], capture_output=True, text=True, timeout=300, env=env, encoding='utf-8', errors='ignore')  # 5 minute timeout
        
        if result.returncode == 0:
            # Parse the output to extract survival rate
            output = result.stdout
            survival_rate = parse_survival_rate_from_output(output)
            
            return {
                'model': os.path.basename(model_path),
                'episodes': episodes,
                'survival_rate': survival_rate,
                'success': True,
                'output': output
            }
        else:
            error_msg = result.stderr if result.stderr else "Unknown error"
            print(f"    ❌ Process failed with return code {result.returncode}")
            print(f"    Error: {error_msg[:200]}")  # Show first 200 chars of error
            return {
                'model': os.path.basename(model_path),
                'episodes': 0,
                'survival_rate': 0.0,
                'success': False,
                'error': error_msg
            }
        
    except subprocess.TimeoutExpired:
        return {
            'model': os.path.basename(model_path),
            'episodes': 0,
            'survival_rate': 0.0,
            'success': False,
            'error': "Evaluation timeout"
        }
    except Exception as e:
        return {
            'model': os.path.basename(model_path),
            'episodes': 0,
            'survival_rate': 0.0,
            'success': False,
            'error': str(e)
        }

def calculate_consistency_rating(std_dev: float, mean_val: float) -> str:
    """Calculate consistency rating based on coefficient of variation."""
    if mean_val == 0:
        return "N/A"
    
    cv = (std_dev / mean_val) * 100  # Coefficient of variation
    
    if cv < 10:
        return "High"
    elif cv < 20:
        return "Medium"
    else:
        return "Variable"

def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    if len(values) < 2:
        return (0, 0)
    
    mean_val = np.mean(values)
    std_err = np.std(values, ddof=1) / np.sqrt(len(values))
    
    # Use t-distribution for small samples (approximation for now)
    # For 95% CI with 10 samples, t-value ≈ 2.262
    t_val = 2.262 if len(values) == 10 else 2.0
    margin_error = t_val * std_err
    
    return (mean_val - margin_error, mean_val + margin_error)

def compare_models_comprehensive():
    """Main function to compare all models comprehensively."""
    print("🏆 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print("Running 10 evaluation sessions of 100 episodes each per model")
    print("Total episodes per model: 1000")
    print("=" * 80)
    
    # Find all models
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return
    
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        print("❌ No model files found!")
        return
    
    # Filter out temporary/checkpoint models and LEGACY models
    model_files = [f for f in model_files if not f.name.startswith('temp_') and not f.name.startswith('LEGACY_')]
    
    # Count filtered models
    all_files = list(models_dir.glob("*.zip"))
    legacy_files = [f for f in all_files if f.name.startswith('LEGACY_')]
    temp_files = [f for f in all_files if f.name.startswith('temp_')]
    
    # Sort models by name for consistent ordering
    model_files.sort(key=lambda x: x.name)
    
    print(f"Found {len(all_files)} total models:")
    print(f"  📦 {len(model_files)} compatible models to evaluate")
    if legacy_files:
        print(f"  📚 {len(legacy_files)} LEGACY models (skipped)")
    if temp_files:
        print(f"  🔄 {len(temp_files)} temporary models (skipped)")
    print("Models to evaluate:", [f.name for f in model_files])
    print()
    
    if not model_files:
        print("❌ No compatible models found to evaluate!")
        if legacy_files:
            print("💡 All models are marked as LEGACY (incompatible)")
            print("   Run option 11 (Check Model Compatibility) to identify issues")
        return
    
    # Store results for all models
    all_model_results = {}
    
    # Evaluate each model
    for i, model_path in enumerate(model_files, 1):
        model_name = model_path.name
        print(f"🔍 Evaluating Model {i}/{len(model_files)}: {model_name}")
        print("-" * 60)
        
        # Run 10 evaluation sessions of 100 episodes each
        model_survival_rates = []
        
        for run in range(1, 11):
            print(f"  📊 Run {run}/10...")
            
            # Run evaluation
            result = run_single_evaluation(str(model_path), episodes=100)
            
            if result['success']:
                survival_rate = result['survival_rate']
                model_survival_rates.append(survival_rate)
                print(f"    ✅ Survival rate: {survival_rate:.1f}%")
            else:
                print(f"    ❌ Evaluation failed: {result.get('error', 'Unknown error')}")
        
        # Store results for this model
        if model_survival_rates:
            all_model_results[model_name] = model_survival_rates
            mean_rate = np.mean(model_survival_rates)
            print(f"  ✅ Completed: Mean survival rate {mean_rate:.1f}% ({len(model_survival_rates)} successful runs)")
        else:
            print(f"  ❌ No valid results for {model_name}")
        
        print()
    
    # Generate comparison table
    print("\n" + "=" * 104)
    print("🏆 COMPREHENSIVE MODEL COMPARISON RESULTS")
    print("=" * 104)
    
    if not all_model_results:
        print("❌ No successful evaluations to compare!")
        return
    
    # Calculate statistics for each model
    model_stats = []
    
    for model_name, survival_rates in all_model_results.items():
        if len(survival_rates) > 0:
            mean_rate = np.mean(survival_rates)
            std_dev = np.std(survival_rates, ddof=1) if len(survival_rates) > 1 else 0
            min_rate = np.min(survival_rates)
            max_rate = np.max(survival_rates)
            
            # Calculate 95% confidence interval
            ci_lower, ci_upper = calculate_confidence_interval(survival_rates, 0.95)
            
            # Calculate consistency rating
            consistency = calculate_consistency_rating(std_dev, mean_rate)
            
            model_stats.append({
                'model': model_name,
                'mean': mean_rate,
                'std': std_dev,
                'min': min_rate,
                'max': max_rate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'consistency': consistency,
                'episodes': len(survival_rates) * 100,
                'runs': len(survival_rates)
            })
    
    # Sort by mean performance (descending)
    model_stats.sort(key=lambda x: x['mean'], reverse=True)
    
    # Print header
    print("Rank Model                               Mean%   ±Std   Min%   Max%   95% CI       Consist  Episodes")
    print("-" * 104)
    
    # Print results
    for rank, stats in enumerate(model_stats, 1):
        model_short = stats['model'][:35]  # Truncate long names
        
        print(f"{rank:<4} {model_short:<35} "
              f"{stats['mean']:>5.1f}   "
              f"±{stats['std']:>4.1f}  "
              f"{stats['min']:>5.1f}  "
              f"{stats['max']:>5.1f}  "
              f"{stats['ci_lower']:>5.1f}-{stats['ci_upper']:>4.1f}%   "
              f"{stats['consistency']:>8} "
              f"{stats['episodes']:>4}")
    
    print("-" * 104)
    print(f"Evaluated {len(model_stats)} models with {model_stats[0]['episodes'] if model_stats else 0} episodes each")
    
    # Additional analysis
    if len(model_stats) >= 2:
        print(f"\n📊 Analysis:")
        best_model = model_stats[0]
        print(f"   🥇 Best performer: {best_model['model'][:40]} ({best_model['mean']:.1f}% ±{best_model['std']:.1f})")
        
        if len(model_stats) >= 3:
            performance_gap = model_stats[0]['mean'] - model_stats[2]['mean']
            print(f"   📈 Performance gap (1st vs 3rd): {performance_gap:.1f} percentage points")
        
        # Find most consistent model
        most_consistent = min(model_stats, key=lambda x: x['std'] if x['mean'] > 0 else float('inf'))
        if most_consistent != best_model:
            print(f"   🎯 Most consistent: {most_consistent['model'][:40]} (±{most_consistent['std']:.1f} std)")
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"model_comparison_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("🏆 COMPREHENSIVE MODEL COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluation: 10 runs × 100 episodes per model\n\n")
        
        f.write("Rank Model                               Mean%   ±Std   Min%   Max%   95% CI       Consist  Episodes\n")
        f.write("-" * 104 + "\n")
        
        for rank, stats in enumerate(model_stats, 1):
            model_short = stats['model'][:35]
            f.write(f"{rank:<4} {model_short:<35} "
                   f"{stats['mean']:>5.1f}   "
                   f"±{stats['std']:>4.1f}  "
                   f"{stats['min']:>5.1f}  "
                   f"{stats['max']:>5.1f}  "
                   f"{stats['ci_lower']:>5.1f}-{stats['ci_upper']:>4.1f}%   "
                   f"{stats['consistency']:>8} "
                   f"{stats['episodes']:>4}\n")
        
        f.write("-" * 104 + "\n")
        f.write(f"Evaluated {len(model_stats)} models with {model_stats[0]['episodes'] if model_stats else 0} episodes each\n")
    
    print(f"\n📄 Results saved to: {results_file}")

if __name__ == "__main__":
    try:
        compare_models_comprehensive()
        
    except KeyboardInterrupt:
        print("\n❌ Comparison cancelled by user")
    except Exception as e:
        print(f"❌ Error during model comparison: {e}")
        import traceback
        traceback.print_exc()
