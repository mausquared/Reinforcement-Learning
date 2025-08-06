#!/usr/bin/env python3
"""
Top Performers Evaluation Script
Comprehensive statistical evaluation of peak performance models with 1000 total episodes
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for styling
from datetime import datetime
import json
import time
from pathlib import Path

# Define top performer models
TOP_PERFORMERS = [
    "peak_performance_4200k_survival_42.0%.zip",
    "peak_performance_3950k_survival_41.0%.zip", 
    "peak_performance_5050k_survival_41.0%.zip",
    "stable_autonomous_28_14000k_stable.zip",
    "peak_performance_4200k_survival_39.0%.zip",
    "peak_performance_1600k_survival_41.0%.zip",
    "peak_performance_300k_survival_35.0%.zip",
    "peak_performance_1400k_survival_42.0%.zip"
]

class TopPerformersEvaluator:
    def __init__(self):
        self.models_dir = "models"
        self.results_dir = "evaluation_results"
        self.python_path = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"
        self.target_models = TOP_PERFORMERS.copy()  # Default to all top performers
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Statistics storage
        self.all_results = {}
        self.summary_stats = {}
        
    def find_model_file(self, target_model):
        """Find the actual model file that matches the target name"""
        if not os.path.exists(self.models_dir):
            return None
            
        all_files = os.listdir(self.models_dir)
        
        # Try exact match first
        if target_model in all_files:
            return os.path.join(self.models_dir, target_model)
            
        # Try partial matches for flexibility
        target_base = target_model.replace(".zip", "").lower()
        for file in all_files:
            if file.endswith(".zip"):
                file_base = file.replace(".zip", "").lower()
                if target_base in file_base or file_base in target_base:
                    return os.path.join(self.models_dir, file)
                    
        return None
    
    def run_single_evaluation(self, model_path, episodes=100):
        """Run a single evaluation session of N episodes"""
        try:
            print(f"    Running {episodes} episodes...")
            
            # Set environment variables to handle Unicode properly
            import os
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
            
            # Run evaluation using train.py mode 5 (evaluation mode)
            result = subprocess.run([
                self.python_path, "train.py", "5", model_path, str(episodes)
            ], capture_output=True, text=True, timeout=300, env=env, encoding='utf-8', errors='replace')  # 5 minute timeout
            
            if result.returncode != 0:
                print(f"    ❌ Evaluation failed: {result.stderr}")
                return None
                
            # Parse output for statistics
            output_lines = result.stdout.split('\n')
            
            # Look for evaluation results
            survival_rate = None
            avg_reward = None
            avg_steps = None
            
            for line in output_lines:
                if "Survival Rate:" in line and "%" in line:
                    try:
                        # Extract from format like "💪 Survival Rate: 42.5% ± 7.2%"
                        parts = line.split("Survival Rate:")
                        if len(parts) > 1:
                            rate_part = parts[1].split("%")[0].strip()
                            # Handle cases with ± symbol
                            if "±" in rate_part:
                                rate_part = rate_part.split("±")[0].strip()
                            survival_rate = float(rate_part)
                    except:
                        pass
                elif "Average Reward:" in line:
                    try:
                        # Extract from format like "🏆 Average Reward: 245.67 ± 45.23"
                        parts = line.split("Average Reward:")
                        if len(parts) > 1:
                            reward_part = parts[1].split("±")[0].strip()
                            avg_reward = float(reward_part)
                    except:
                        pass
                elif "Average Length:" in line:
                    try:
                        # Extract from format like "⏱️  Average Length: 125.3 ± 23.4"
                        parts = line.split("Average Length:")
                        if len(parts) > 1:
                            steps_part = parts[1].split("±")[0].strip()
                            avg_steps = float(steps_part)
                    except:
                        pass
            
            if survival_rate is not None:
                return {
                    'survival_rate': survival_rate,
                    'avg_reward': avg_reward or 0,
                    'avg_steps': avg_steps or 0,
                    'episodes': episodes
                }
            else:
                print(f"    ⚠️ Could not parse results from output")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ Evaluation timed out after 5 minutes")
            return None
        except Exception as e:
            print(f"    ❌ Error during evaluation: {e}")
            return None
    
    def evaluate_model_comprehensive(self, model_name, runs=10, episodes_per_run=100):
        """Run comprehensive evaluation: multiple runs of N episodes each"""
        print(f"\n🎯 Evaluating {model_name}")
        print(f"📊 Running {runs} runs of {episodes_per_run} episodes each ({runs * episodes_per_run} total)")
        
        # Find the actual model file
        model_path = self.find_model_file(model_name)
        if not model_path:
            print(f"❌ Model file not found: {model_name}")
            return None
            
        print(f"📂 Found model: {model_path}")
        
        # Store results for each run
        run_results = []
        
        for run_num in range(1, runs + 1):
            print(f"  📈 Run {run_num}/{runs}")
            start_time = time.time()
            
            result = self.run_single_evaluation(model_path, episodes_per_run)
            if result:
                result['run_number'] = run_num
                result['timestamp'] = datetime.now().isoformat()
                run_results.append(result)
                
                duration = time.time() - start_time
                print(f"    ✅ Survival: {result['survival_rate']:.1f}% | "
                      f"Reward: {result['avg_reward']:.1f} | "
                      f"Steps: {result['avg_steps']:.1f} | "
                      f"Time: {duration:.1f}s")
            else:
                print(f"    ❌ Run {run_num} failed")
        
        if not run_results:
            print(f"❌ All evaluation runs failed for {model_name}")
            return None
            
        # Calculate comprehensive statistics
        survival_rates = [r['survival_rate'] for r in run_results]
        avg_rewards = [r['avg_reward'] for r in run_results]
        avg_steps = [r['avg_steps'] for r in run_results]
        
        stats = {
            'model_name': model_name,
            'model_path': model_path,
            'total_episodes': len(run_results) * episodes_per_run,
            'successful_runs': len(run_results),
            'failed_runs': runs - len(run_results),
            
            # Survival Rate Statistics
            'survival_mean': np.mean(survival_rates),
            'survival_std': np.std(survival_rates),
            'survival_min': np.min(survival_rates),
            'survival_max': np.max(survival_rates),
            'survival_median': np.median(survival_rates),
            'survival_q25': np.percentile(survival_rates, 25),
            'survival_q75': np.percentile(survival_rates, 75),
            
            # Reward Statistics
            'reward_mean': np.mean(avg_rewards),
            'reward_std': np.std(avg_rewards),
            'reward_min': np.min(avg_rewards),
            'reward_max': np.max(avg_rewards),
            
            # Steps Statistics
            'steps_mean': np.mean(avg_steps),
            'steps_std': np.std(avg_steps),
            'steps_min': np.min(avg_steps),
            'steps_max': np.max(avg_steps),
            
            # Confidence Intervals (95%)
            'survival_ci_lower': np.percentile(survival_rates, 2.5),
            'survival_ci_upper': np.percentile(survival_rates, 97.5),
            
            # Reliability Metrics
            'survival_coefficient_of_variation': np.std(survival_rates) / np.mean(survival_rates) if np.mean(survival_rates) > 0 else float('inf'),
            'consistent_performance': np.std(survival_rates) < 5.0,  # Less than 5% std dev is "consistent"
            
            # Raw data
            'run_results': run_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Store results
        self.all_results[model_name] = stats
        
        print(f"✅ {model_name} completed:")
        print(f"   📊 Survival: {stats['survival_mean']:.1f}% ± {stats['survival_std']:.1f}% "
              f"(95% CI: {stats['survival_ci_lower']:.1f}%-{stats['survival_ci_upper']:.1f}%)")
        print(f"   🎯 Consistency: {'High' if stats['consistent_performance'] else 'Variable'} "
              f"(CV: {stats['survival_coefficient_of_variation']:.3f})")
        
        return stats
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.all_results:
            print("❌ No results to compare")
            return
            
        print(f"\n🏆 COMPREHENSIVE PERFORMANCE COMPARISON")
        print(f"=" * 80)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, stats in self.all_results.items():
            summary_data.append({
                'Model': model_name.replace('.zip', '').replace('peak_performance_', '').replace('stable_autonomous_', 'stable_'),
                'Episodes': stats['total_episodes'],
                'Survival_Mean': stats['survival_mean'],
                'Survival_Std': stats['survival_std'],
                'Survival_Min': stats['survival_min'],
                'Survival_Max': stats['survival_max'],
                'CI_Lower': stats['survival_ci_lower'],
                'CI_Upper': stats['survival_ci_upper'],
                'Consistency': 'High' if stats['consistent_performance'] else 'Variable',
                'Reward_Mean': stats['reward_mean'],
                'Steps_Mean': stats['steps_mean']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Survival_Mean', ascending=False)
        
        # Print ranking table
        print(f"\n📊 PERFORMANCE RANKING (by mean survival rate):")
        print(f"-" * 120)
        print(f"{'Rank':<4} {'Model':<35} {'Mean%':<7} {'±Std':<6} {'Min%':<6} {'Max%':<6} {'95% CI':<12} {'Consist':<8} {'Episodes':<8}")
        print(f"-" * 120)
        
        for idx, row in df.iterrows():
            rank = df.index.get_loc(idx) + 1
            ci_range = f"{row['CI_Lower']:.1f}-{row['CI_Upper']:.1f}%"
            print(f"{rank:<4} {row['Model']:<35} {row['Survival_Mean']:<7.1f} "
                  f"±{row['Survival_Std']:<5.1f} {row['Survival_Min']:<6.1f} {row['Survival_Max']:<6.1f} "
                  f"{ci_range:<12} {row['Consistency']:<8} {row['Episodes']:<8}")
        
        # Statistical insights
        print(f"\n📈 STATISTICAL INSIGHTS:")
        print(f"-" * 40)
        
        best_model = df.iloc[0]
        worst_model = df.iloc[-1]
        
        print(f"🥇 Best Performer: {best_model['Model']}")
        print(f"   📊 Mean survival: {best_model['Survival_Mean']:.1f}% ± {best_model['Survival_Std']:.1f}%")
        print(f"   🎯 95% CI: {best_model['CI_Lower']:.1f}% - {best_model['CI_Upper']:.1f}%")
        print(f"   📈 Consistency: {best_model['Consistency']}")
        
        print(f"\n📊 Performance Spread:")
        print(f"   Range: {worst_model['Survival_Mean']:.1f}% - {best_model['Survival_Mean']:.1f}%")
        print(f"   Difference: {best_model['Survival_Mean'] - worst_model['Survival_Mean']:.1f} percentage points")
        
        # Consistency analysis
        consistent_models = df[df['Consistency'] == 'High']
        if len(consistent_models) > 0:
            print(f"\n🎯 Most Consistent Models:")
            for _, row in consistent_models.head(3).iterrows():
                print(f"   • {row['Model']}: {row['Survival_Mean']:.1f}% ± {row['Survival_Std']:.1f}%")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = os.path.join(self.results_dir, f"top_performers_comparison_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved to: {csv_path}")
        
        # Save JSON with all data
        json_path = os.path.join(self.results_dir, f"top_performers_detailed_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"💾 Complete data saved to: {json_path}")
        
        return df
    
    def generate_visualizations(self, df):
        """Generate visualization plots"""
        if df is None or len(df) == 0:
            return
            
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Survival Rate Comparison with Error Bars
        ax1 = axes[0, 0]
        models = [name[:20] + '...' if len(name) > 20 else name for name in df['Model']]
        y_pos = np.arange(len(models))
        
        ax1.barh(y_pos, df['Survival_Mean'], xerr=df['Survival_Std'], 
                capsize=5, alpha=0.7, color='skyblue')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(models)
        ax1.set_xlabel('Survival Rate (%)')
        ax1.set_title('Mean Survival Rate with Standard Deviation')
        ax1.grid(True, alpha=0.3)
        
        # 2. Box Plot of Survival Rates
        ax2 = axes[0, 1]
        survival_data = []
        model_labels = []
        
        for model_name, stats in self.all_results.items():
            survival_rates = [r['survival_rate'] for r in stats['run_results']]
            survival_data.append(survival_rates)
            short_name = model_name.replace('.zip', '').replace('peak_performance_', '')[:15]
            model_labels.append(short_name)
        
        bp = ax2.boxplot(survival_data, labels=model_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')
        ax2.set_ylabel('Survival Rate (%)')
        ax2.set_title('Survival Rate Distribution (Box Plot)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Confidence Intervals
        ax3 = axes[1, 0]
        ax3.errorbar(range(len(df)), df['Survival_Mean'], 
                    yerr=[df['Survival_Mean'] - df['CI_Lower'], 
                          df['CI_Upper'] - df['Survival_Mean']], 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels([name[:15] for name in df['Model']], rotation=45)
        ax3.set_ylabel('Survival Rate (%)')
        ax3.set_title('95% Confidence Intervals')
        ax3.grid(True, alpha=0.3)
        
        # 4. Consistency vs Performance
        ax4 = axes[1, 1]
        colors = ['green' if c == 'High' else 'orange' for c in df['Consistency']]
        scatter = ax4.scatter(df['Survival_Mean'], df['Survival_Std'], 
                            c=colors, alpha=0.7, s=100)
        ax4.set_xlabel('Mean Survival Rate (%)')
        ax4.set_ylabel('Standard Deviation (%)')
        ax4.set_title('Performance vs Consistency')
        ax4.grid(True, alpha=0.3)
        
        # Add model labels to scatter plot
        for i, model in enumerate(df['Model']):
            ax4.annotate(model[:10], (df.iloc[i]['Survival_Mean'], df.iloc[i]['Survival_Std']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.results_dir, f"top_performers_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualizations saved to: {plot_path}")
        
        plt.show()
    
    def run_full_evaluation(self):
        """Run the complete evaluation pipeline"""
        print(f"🚀 Starting Top Performers Comprehensive Evaluation")
        print(f"🎯 Target: {len(self.target_models)} models × 1000 episodes each")
        print(f"📊 Method: 10 runs of 100 episodes per model")
        print(f"⏱️ Estimated time: {len(self.target_models) * 10 * 2:.0f}-{len(self.target_models) * 10 * 4:.0f} minutes")
        
        start_time = time.time()
        
        # Evaluate each model
        successful_evaluations = 0
        for i, model_name in enumerate(self.target_models, 1):
            print(f"\n{'='*60}")
            print(f"🎯 Model {i}/{len(self.target_models)}: {model_name}")
            print(f"{'='*60}")
            
            result = self.evaluate_model_comprehensive(model_name, runs=10, episodes_per_run=100)
            if result:
                successful_evaluations += 1
            
            # Progress update
            elapsed = time.time() - start_time
            if i < len(self.target_models):
                estimated_remaining = (elapsed / i) * (len(self.target_models) - i)
                print(f"⏱️ Progress: {i}/{len(self.target_models)} | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"ETA: {estimated_remaining/60:.1f}min")
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"🏁 EVALUATION COMPLETE!")
        print(f"✅ Successfully evaluated: {successful_evaluations}/{len(self.target_models)} models")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📊 Total episodes run: {successful_evaluations * 1000}")
        print(f"{'='*80}")
        
        if successful_evaluations > 0:
            # Generate comparison report
            df = self.generate_comparison_report()
            
            # Generate visualizations
            self.generate_visualizations(df)
            
            print(f"\n🎉 Complete evaluation results available in: {self.results_dir}/")
        else:
            print(f"❌ No successful evaluations completed")

def main():
    """Main execution function"""
    print("🎯 TOP PERFORMERS COMPREHENSIVE EVALUATION")
    print("=" * 50)
    print("📊 This will run 1000 episodes for each top performer model")
    print("🎲 Using 10 runs of 100 episodes each for statistical reliability")
    print("📈 Generates confidence intervals and consistency metrics")
    
    evaluator = TopPerformersEvaluator()
    
    # Check if models exist
    missing_models = []
    found_models = []
    
    for model in TOP_PERFORMERS:
        if evaluator.find_model_file(model):
            found_models.append(model)
        else:
            missing_models.append(model)
    
    if found_models:
        print(f"\n✅ Found {len(found_models)} models:")
        for model in found_models:
            print(f"   • {model}")
    
    if missing_models:
        print(f"\n⚠️ Missing {len(missing_models)} models:")
        for model in missing_models:
            print(f"   • {model}")
        print(f"\n💡 Will evaluate only the found models")
    
    if not found_models:
        print(f"\n❌ No target models found in models/ directory")
        print(f"Please ensure your top performer models are available")
        return
    
    # Update the evaluator to only include found models
    evaluator.target_models = found_models
    
    print(f"\n🚀 Ready to evaluate {len(found_models)} models")
    print(f"📊 This will take approximately {len(found_models) * 10 * 2:.0f}-{len(found_models) * 10 * 4:.0f} minutes")
    
    confirm = input("\nProceed with comprehensive evaluation? (y/n): ").strip().lower()
    if confirm == 'y':
        evaluator.run_full_evaluation()
    else:
        print("Evaluation cancelled.")

if __name__ == "__main__":
    main()
