#!/usr/bin/env python3
"""
🐦 3D Hummingbird Environment - Quick Launcher
Simple launcher for the 3D matplotlib hummingbird environment
"""

import os
import sys
import subprocess

# Python executable path for this virtual environment
PYTHON_PATH = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"

def main():
    """Simple launcher with clear options."""
    print("🐦 3D HUMMINGBIRD REINFORCEMENT LEARNING")
    print("=" * 45)
    print("🌍 3D Environment with Matplotlib Visualization")
    print("🔋 Energy Management & Anti-Camping System")
    print("🌸 Multiple Flowers with Cooldowns")
    print("🎯 PPO Training with AUTONOMOUS LEARNING")
    print("🤖 Minimal reward engineering for strategy discovery")
    print("=" * 45)
    
    print("\nChoose an option:")
    print("1. 🎮 Test Environment (Watch 3D hummingbird)")
    print("2. 🎯 Train New Model (500K timesteps)")
    print("3. 🚀 Train New Model (1M timesteps)")
    print("4. 🎛️ Train New Model (Custom timesteps)")
    print("5. 🔄 Continue Training Existing Model")
    print("6. 🤖 Test Trained Model")
    print("7. 📊 Evaluate Model Performance")
    print("8. 📈 View Training Progress")
    print("9. 🔬 Analyze Environment Difficulty")
    print("10. ⚖️ Stable Training (Optimized hyperparameters)")
    print("11. ⏹️ Manual Training Control (Auto-save peaks)")
    print("12. 🔍 Find Peak Performance Models")
    print("13. 🔄⏹️ Continue Training with Manual Control")
    print("14. 📊 Extended Evaluation (500+ episodes for low variance)")
    print("15. 📚 Curriculum Learning Training (Progressive difficulty)")
    print("16. 🎓🚀 Continue Mastery Model with Enhanced Curriculum")
    print("17. 🎯🔥 Specialist HARD Mode Training (Transform graduate to expert)")
    print("18. 🏆 Evaluate Top Performers (1000 episodes each with statistics)")
    print("0. ❌ Exit")
    
    choice = input("\nEnter your choice (0-18): ").strip()
    
    if choice == "1":
        print("\n🎮 Testing 3D Environment...")
        print("Watch the hummingbird navigate 3D space!")
        print("Close the matplotlib window when done.")
        subprocess.run([PYTHON_PATH, "hummingbird_env.py"])
        
    elif choice == "2":
        print("\n🎯 Training New Model...")
        print("This will train for 500,000 timesteps (~30-60 minutes)")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([PYTHON_PATH, "train.py", "1"])  # Pass argument "1" for 500K timesteps
        else:
            print("Training cancelled.")
            
    elif choice == "3":
        print("\n🚀 Training New Model (Extended)...")
        print("This will train for 1,000,000 timesteps (~60-120 minutes)")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([PYTHON_PATH, "train.py", "2"])  # Pass argument "2" for 1M timesteps
        else:
            print("Training cancelled.")
            
    elif choice == "4":
        print("\n🎛️ Custom Training Session...")
        print("Define your own training duration!")
        
        while True:
            try:
                timesteps_input = input("\nEnter timesteps (e.g., 1500000 for 1.5M, 4000000 for 4M): ").strip()
                custom_timesteps = int(timesteps_input)
                
                if custom_timesteps < 10000:
                    print("⚠️  Warning: Very low timesteps may not produce good results.")
                    print("   Recommended minimum: 100,000 timesteps")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                # Convert to human-readable format
                if custom_timesteps >= 1000000:
                    readable = f"{custom_timesteps/1000000:.1f}M"
                elif custom_timesteps >= 1000:
                    readable = f"{custom_timesteps/1000:.0f}K"
                else:
                    readable = str(custom_timesteps)
                
                # Estimate training time (rough approximation)
                estimated_minutes = custom_timesteps / 10000  # Very rough estimate
                if estimated_minutes < 60:
                    time_estimate = f"~{estimated_minutes:.0f} minutes"
                else:
                    time_estimate = f"~{estimated_minutes/60:.1f} hours"
                
                print(f"\n📊 Training Configuration:")
                print(f"   Timesteps: {custom_timesteps:,} ({readable})")
                print(f"   Estimated time: {time_estimate}")
                print(f"   Training mode: AUTONOMOUS LEARNING")
                print(f"   Reward engineering: MINIMAL (strategy discovery)")
                print(f"   Environment: Enhanced 3D with physics")
                
                confirm = input("\nStart custom training? (y/n): ").strip().lower()
                if confirm == 'y':
                    subprocess.run([PYTHON_PATH, "train.py", "custom", str(custom_timesteps)])
                    break
                else:
                    print("Training cancelled.")
                    break
                    
            except ValueError:
                print("❌ Invalid input. Please enter a number (e.g., 1500000)")
                continue
            
    elif choice == "5":
        print("\n🔄 Continue Training Existing Model...")
        print("Extend training on a previously trained model")
        
        # List available models
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            model_files.sort(reverse=True)  # Most recent first
            
            if model_files:
                print(f"\nAvailable models ({len(model_files)} found):")
                for i, model in enumerate(model_files, 1):
                    print(f"  {i}. {model}")
                
                model_choice = input(f"\nChoose model to continue training (1-{len(model_files)}): ").strip()
                try:
                    choice_num = int(model_choice)
                    if 1 <= choice_num <= len(model_files):
                        selected_model = f"./models/{model_files[choice_num - 1]}"
                        
                        # Get additional timesteps
                        while True:
                            try:
                                additional_timesteps_input = input("\nAdditional timesteps (e.g., 2500000 for 2.5M): ").strip()
                                additional_timesteps = int(additional_timesteps_input)
                                
                                if additional_timesteps < 10000:
                                    print("⚠️  Warning: Very low timesteps may not provide meaningful improvement.")
                                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                                    if confirm != 'y':
                                        continue
                                
                                # Convert to human-readable format
                                if additional_timesteps >= 1000000:
                                    readable = f"{additional_timesteps/1000000:.1f}M"
                                elif additional_timesteps >= 1000:
                                    readable = f"{additional_timesteps/1000:.0f}K"
                                else:
                                    readable = str(additional_timesteps)
                                
                                # Estimate training time
                                estimated_minutes = additional_timesteps / 10000
                                if estimated_minutes < 60:
                                    time_estimate = f"~{estimated_minutes:.0f} minutes"
                                else:
                                    time_estimate = f"~{estimated_minutes/60:.1f} hours"
                                
                                # Get parameter configuration choice
                                print(f"\n🔧 Parameter Configuration:")
                                print(f"   1. Original parameters (preserves learned strategies)")
                                print(f"   2. Stable parameters (improved stability + survival rewards)")
                                
                                param_choice = input("\nChoose parameter configuration (1-2): ").strip()
                                use_stable = param_choice == "2"
                                
                                print(f"\n📊 Continue Training Configuration:")
                                print(f"   Base model: {model_files[choice_num - 1]}")
                                print(f"   Additional timesteps: {additional_timesteps:,} ({readable})")
                                print(f"   Estimated time: {time_estimate}")
                                print(f"   Training mode: CONTINUE EXISTING")
                                
                                if use_stable:
                                    print(f"   Parameters: STABLE (conservative hyperparameters)")
                                    print(f"   Environment: STABLE (with survival rewards)")
                                    print(f"   Benefits: Improved stability, higher survival rates")
                                    print(f"   Survival rewards: +0.1 per step + energy efficiency bonus")
                                else:
                                    print(f"   Parameters: ORIGINAL (preserves learned strategies)")
                                    print(f"   Environment: AUTONOMOUS LEARNING (minimal rewards)")
                                    print(f"   Benefits: Maintains existing training approach")
                                
                                confirm = input("\nContinue training? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    if use_stable:
                                        subprocess.run([PYTHON_PATH, "train.py", "continue", selected_model, str(additional_timesteps), "stable"])
                                    else:
                                        subprocess.run([PYTHON_PATH, "train.py", "continue", selected_model, str(additional_timesteps)])
                                break
                                
                            except ValueError:
                                print("❌ Invalid input. Please enter a number.")
                                
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                print("No trained models found in models/ directory.")
                print("Please train a model first (options 2, 3, or 4).")
        else:
            print("Models directory not found. Please train a model first.")
            
    elif choice == "6":
        print("\n🤖 Testing Trained Model...")
        print("🎮 Interactive test with 3D visualization")
        print("Looking for available models...")
        
        # List available models
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            model_files.sort()  # Sort files alphabetically
            if model_files:
                print(f"\nAvailable models ({len(model_files)} found):")
                for i, model in enumerate(model_files, 1):
                    print(f"  {i}. {model}")
                print(f"  {len(model_files) + 1}. Use default (best_model)")
                
                model_choice = input(f"\nChoose model (1-{len(model_files) + 1}): ").strip()
                try:
                    choice_num = int(model_choice)
                    if 1 <= choice_num <= len(model_files):
                        selected_model = f"./models/{model_files[choice_num - 1]}"
                        print(f"Testing model: {model_files[choice_num - 1]} (with 3D visualization)")
                        subprocess.run([PYTHON_PATH, "train.py", "4", selected_model])
                    elif choice_num == len(model_files) + 1:
                        print("Testing default model: best_model (with 3D visualization)")
                        subprocess.run([PYTHON_PATH, "train.py", "3"])
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input. Using default model (with 3D visualization).")
                    subprocess.run([PYTHON_PATH, "train.py", "3"])
            else:
                print("No trained models found in models/ directory.")
                print("Please train a model first (options 2 or 3).")
        else:
            print("Models directory not found. Please train a model first.")
            
    elif choice == "7":
        print("\n📊 Evaluating Model Performance...")
        print("Comprehensive evaluation with environment compatibility checking...")
        print("📈 No visualization - focused on statistical analysis")
        print("\n💡 NOTE: Models are evaluated based on their training environment:")
        print("   📊 LEGACY models: Trained with engineered rewards (may show poor results)")
        print("   🤖 AUTONOMOUS models: Trained for genuine strategy discovery")
        print("   ⚠️  Reward scores between different environment versions are NOT comparable!")
        
        # List available models (same as option 4)
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            model_files.sort()  # Sort files alphabetically
            if model_files:
                print(f"\nAvailable models ({len(model_files)} found):")
                for i, model in enumerate(model_files, 1):
                    print(f"  {i}. {model}")
                print(f"  {len(model_files) + 1}. Use default (best_model)")
                print(f"  {len(model_files) + 2}. Evaluate ALL models")
                
                model_choice = input(f"\nChoose model to evaluate (1-{len(model_files) + 2}): ").strip()
                try:
                    choice_num = int(model_choice)
                    if 1 <= choice_num <= len(model_files):
                        selected_model = f"./models/{model_files[choice_num - 1]}"
                        print(f"Evaluating model: {model_files[choice_num - 1]}")
                        print("Running 100 episodes for comprehensive evaluation (no visualization)...")
                        subprocess.run([PYTHON_PATH, "train.py", "5", selected_model])  # New evaluation mode
                    elif choice_num == len(model_files) + 1:
                        print("Evaluating default model: best_model")
                        print("Running 100 episodes for comprehensive evaluation (no visualization)...")
                        subprocess.run([PYTHON_PATH, "train.py", "5", "./models/best_model"])
                    elif choice_num == len(model_files) + 2:
                        print("Evaluating ALL models - this may take a while...")
                        print("(50 episodes per model for comprehensive comparison - no visualization)")
                        subprocess.run([PYTHON_PATH, "train.py", "6"])  # Evaluate all models
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input. Using default model.")
                    print("Running 100 episodes for comprehensive evaluation (no visualization)...")
                    subprocess.run([PYTHON_PATH, "train.py", "5", "./models/best_model"])
            else:
                print("No trained models found in models/ directory.")
                print("Please train a model first (options 2 or 3).")
        else:
            print("Models directory not found. Please train a model first.")
            
    elif choice == "8":
        print("\n📈 Viewing Training Progress...")
        subprocess.run([PYTHON_PATH, "train.py", "progress"])
            
    elif choice == "9":
        print("\n🔬 Analyzing Environment Difficulty...")
        print("Comprehensive analysis of environment parameters and difficulty")
        subprocess.run([PYTHON_PATH, "environment_analysis.py"])
            
    elif choice == "10":
        print("\n⚖️ Stable Training Configuration...")
        print("📊 Uses conservative hyperparameters for consistent learning")
        print("⚠️  NOTE: Uses SAME ENVIRONMENT as autonomous learning")
        print("🔧 Hyperparameter improvements:")
        print("   • Learning rate schedule: 3e-4 → 0 (linear decay)")
        print("   • Larger rollout buffer: 4096 steps")
        print("   • Observation normalization: ENABLED")
        print("   • Enhanced exploration: ent_coef = 0.01")
        print("   • Long-term thinking: gamma = 0.995")
        print("🤖 Same reward structure as autonomous learning, better training stability")
        
        while True:
            try:
                timesteps_input = input("\nEnter timesteps (recommended: 2000000+ for stability): ").strip()
                custom_timesteps = int(timesteps_input)
                
                if custom_timesteps < 100000:
                    print("⚠️  Warning: Very low timesteps may not show stability improvements.")
                    print("   Recommended minimum: 1,000,000 timesteps for stable training")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                # Convert to human-readable format
                if custom_timesteps >= 1000000:
                    readable = f"{custom_timesteps/1000000:.1f}M"
                elif custom_timesteps >= 1000:
                    readable = f"{custom_timesteps/1000:.0f}K"
                else:
                    readable = str(custom_timesteps)
                
                # Estimate training time
                estimated_minutes = custom_timesteps / 8000  # More conservative estimate for stable training
                if estimated_minutes < 60:
                    time_estimate = f"~{estimated_minutes:.0f} minutes"
                else:
                    time_estimate = f"~{estimated_minutes/60:.1f} hours"
                
                print(f"\n📋 STABLE TRAINING WITH IMPROVED HYPERPARAMETERS:")
                print(f"   🎯 Timesteps: {custom_timesteps:,} ({readable})")
                print(f"   📊 Learning rate schedule: 3e-4 → 0 (linear decay)")
                print(f"   📊 Larger rollout buffer: 4096 steps")
                print(f"   🎯 Observation normalization: ENABLED")
                print(f"   � Hyperparameter improvements:")
                print(f"      • Enhanced exploration: ent_coef = 0.01")
                print(f"      • Long-term thinking: gamma = 0.995")
                print(f"      • Conservative learning for stability")
                print(f"   🤖 Environment: Same as autonomous learning")
                print(f"   ⚠️  Reward structure: Unchanged from base environment")
                print(f"   ⏱️  Estimated time: {time_estimate}")
                
                confirm = input("\nStart stable training? (y/n): ").strip().lower()
                if confirm == 'y':
                    subprocess.run([PYTHON_PATH, "train.py", "stable", str(custom_timesteps)])
                    break
                else:
                    print("Training cancelled.")
                    break
                    
            except ValueError:
                print("❌ Invalid input. Please enter a number (e.g., 2000000)")
                continue
    
    elif choice == "11":
        print("\n⏹️ Manual Training Control...")
        print("🎯 Start training with auto-save capability")
        print("💡 You can monitor progress and models are auto-saved at peak performance")
        print("🏆 Auto-saves when survival rate hits new peaks ≥40%")
        print("🔄 Use Ctrl+C to stop training when you see desired performance")
        
        while True:
            try:
                timesteps_input = input("\nEnter maximum timesteps (e.g., 10000000 for 10M): ").strip()
                max_timesteps = int(timesteps_input)
                
                if max_timesteps < 100000:
                    print("⚠️  Warning: Very low timesteps may not reach peak performance.")
                    print("   Recommended minimum: 1,000,000 timesteps for meaningful results")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                
                # Convert to human-readable format
                if max_timesteps >= 1000000:
                    readable = f"{max_timesteps/1000000:.1f}M"
                elif max_timesteps >= 1000:
                    readable = f"{max_timesteps/1000:.0f}K"
                else:
                    readable = str(max_timesteps)
                
                # Estimate training time
                estimated_minutes = max_timesteps / 8000  # Conservative estimate
                if estimated_minutes < 60:
                    time_estimate = f"~{estimated_minutes:.0f} minutes"
                else:
                    time_estimate = f"~{estimated_minutes/60:.1f} hours"
                
                print(f"\n📋 MANUAL CONTROL TRAINING:")
                print(f"   🎯 Max timesteps: {max_timesteps:,} ({readable})")
                print(f"   ⏹️ Auto-save: Every peak performance ≥40% survival")
                print(f"   🏆 Peak threshold: 40%+ survival (updates best_model.zip)")
                print(f"   💾 Models saved as: peak_performance_XXXk_survival_XX.X%.zip")
                print(f"   📄 Summary files: peak_performance_XXXk_summary.txt")
                print(f"   🔄 You can Ctrl+C to stop when satisfied")
                print(f"   ⏱️ Estimated time: {time_estimate}")
                print(f"   🎛️ Training mode: STABLE (with survival rewards)")
                
                confirm = input("\nStart manual control training? (y/n): ").strip().lower()
                if confirm == 'y':
                    # Use stable training with auto-save
                    print("\n🚀 Starting manual control training with auto-save...")
                    print("👀 Watch for peak performance messages and auto-saves!")
                    print("🛑 Press Ctrl+C to stop training when you see desired performance")
                    subprocess.run([PYTHON_PATH, "train.py", "stable", str(max_timesteps)])
                    break
                else:
                    print("Training cancelled.")
                    break
                    
            except ValueError:
                print("❌ Invalid input. Please enter a number")
                continue
    
    elif choice == "12":
        print("\n🔍 Finding Peak Performance Models...")
        print("Looking for auto-saved peak performance models...")
        
        models_dir = "models"
        if os.path.exists(models_dir):
            all_files = os.listdir(models_dir)
            peak_models = [f for f in all_files if f.startswith('peak_performance_') and f.endswith('.zip')]
            summary_files = [f for f in all_files if f.startswith('peak_performance_') and f.endswith('_summary.txt')]
            
            if peak_models:
                print(f"\n🏆 Peak Performance Models Found ({len(peak_models)}):")
                print("-" * 80)
                
                # Sort by survival rate (extract from filename)
                def extract_survival_rate(filename):
                    try:
                        # Extract survival rate from filename like "peak_performance_2950k_survival_47.1%.zip"
                        parts = filename.split('_survival_')
                        if len(parts) == 2:
                            rate_part = parts[1].replace('%.zip', '')
                            return float(rate_part)
                    except:
                        pass
                    return 0.0
                
                sorted_peaks = sorted(peak_models, key=extract_survival_rate, reverse=True)
                
                for i, model in enumerate(sorted_peaks, 1):
                    model_path = os.path.join(models_dir, model)
                    file_size = os.path.getsize(model_path)
                    modified = os.path.getmtime(model_path)
                    import datetime
                    mod_time = datetime.datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
                    
                    # Extract info from filename
                    try:
                        parts = model.replace('peak_performance_', '').replace('.zip', '').split('_survival_')
                        if len(parts) == 2:
                            timestep_part = parts[0]
                            survival_part = parts[1].replace('%', '')
                            print(f"  {i}. {model}")
                            print(f"     🎯 Survival: {survival_part}% | 📊 Step: {timestep_part} | 📅 {mod_time}")
                            
                            # Check if summary file exists
                            summary_file = model.replace('.zip', '_summary.txt')
                            if summary_file in summary_files:
                                summary_path = os.path.join(models_dir, summary_file)
                                try:
                                    with open(summary_path, 'r') as f:
                                        content = f.read()
                                        if 'Average reward:' in content:
                                            reward_line = [line for line in content.split('\n') if 'Average reward:' in line]
                                            if reward_line:
                                                reward = reward_line[0].split(': ')[1]
                                                print(f"     💎 Reward: {reward} | 📄 Summary available")
                                except:
                                    pass
                        else:
                            print(f"  {i}. {model}")
                            print(f"     📅 Modified: {mod_time}")
                    except:
                        print(f"  {i}. {model}")
                        print(f"     📅 Modified: {mod_time}")
                    
                    print()
                
                print(f"💡 TIP: Use Option 6 to test any of these peak models!")
                print(f"🎯 These models were auto-saved when they achieved new performance peaks")
                
                if len(sorted_peaks) > 0:
                    best_model = sorted_peaks[0]
                    best_survival = extract_survival_rate(best_model)
                    print(f"\n🏆 BEST PERFORMER: {best_model}")
                    print(f"   🎯 Peak Survival Rate: {best_survival:.1f}%")
                    print(f"   💡 This is likely your missing 47% survival model!")
                
            else:
                print("\n❌ No peak performance models found.")
                print("💡 Peak models are created when using:")
                print("   • Option 11 (Manual Training Control)")
                print("   • Any training that achieves ≥40% survival rates")
                print("   • Models are auto-saved as peak_performance_XXXk_survival_XX.X%.zip")
        else:
            print("❌ Models directory not found.")
    
    elif choice == "13":
        print("\n🔄⏹️ Continue Training with Manual Control...")
        print("🎯 Continue an existing model with auto-save capability")
        print("🏆 Auto-saves when survival rate hits new peaks ≥40%")
        print("🔄 Use Ctrl+C to stop training when you see desired performance")
        print("💡 Perfect for extending your breakthrough models!")
        
        # List available models
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            model_files.sort(reverse=True)  # Most recent first
            
            if model_files:
                print(f"\nAvailable models ({len(model_files)} found):")
                for i, model in enumerate(model_files, 1):
                    # Add indicators for special model types
                    if 'peak_performance_' in model:
                        indicator = "🏆"
                    elif 'stable' in model.lower():
                        indicator = "⚖️"
                    elif 'best_model' in model:
                        indicator = "🥇"
                    else:
                        indicator = "📋"
                    print(f"  {i}. {indicator} {model}")
                
                model_choice = input(f"\nChoose model to continue with manual control (1-{len(model_files)}): ").strip()
                try:
                    choice_num = int(model_choice)
                    if 1 <= choice_num <= len(model_files):
                        selected_model = f"./models/{model_files[choice_num - 1]}"
                        selected_model_name = model_files[choice_num - 1]
                        
                        # Get additional timesteps
                        while True:
                            try:
                                additional_timesteps_input = input("\nAdditional timesteps (e.g., 5000000 for 5M): ").strip()
                                additional_timesteps = int(additional_timesteps_input)
                                
                                if additional_timesteps < 100000:
                                    print("⚠️  Warning: Very low timesteps may not reach new peaks.")
                                    print("   Recommended minimum: 1,000,000 timesteps for meaningful improvement")
                                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                                    if confirm != 'y':
                                        continue
                                
                                # Convert to human-readable format
                                if additional_timesteps >= 1000000:
                                    readable = f"{additional_timesteps/1000000:.1f}M"
                                elif additional_timesteps >= 1000:
                                    readable = f"{additional_timesteps/1000:.0f}K"
                                else:
                                    readable = str(additional_timesteps)
                                
                                # Estimate training time
                                estimated_minutes = additional_timesteps / 8000  # Conservative estimate
                                if estimated_minutes < 60:
                                    time_estimate = f"~{estimated_minutes:.0f} minutes"
                                else:
                                    time_estimate = f"~{estimated_minutes/60:.1f} hours"
                                
                                # Get parameter configuration choice
                                print(f"\n🔧 Parameter Configuration:")
                                print(f"   1. Original parameters (preserves learned strategies)")
                                print(f"   2. Stable parameters (improved stability + survival rewards)")
                                print(f"   💡 Recommendation: Use stable parameters for better peak detection")
                                
                                param_choice = input("\nChoose parameter configuration (1-2): ").strip()
                                use_stable = param_choice == "2"
                                
                                print(f"\n📋 CONTINUE TRAINING WITH MANUAL CONTROL:")
                                print(f"   📂 Base model: {selected_model_name}")
                                print(f"   🎯 Additional timesteps: {additional_timesteps:,} ({readable})")
                                print(f"   ⏹️ Auto-save: Every peak performance ≥40% survival")
                                print(f"   🏆 Peak detection: New survival or reward peaks")
                                print(f"   💾 Models saved as: peak_performance_XXXk_survival_XX.X%.zip")
                                print(f"   📄 Summary files: peak_performance_XXXk_summary.txt")
                                print(f"   🔄 You can Ctrl+C to stop when satisfied")
                                print(f"   ⏱️ Estimated time: {time_estimate}")
                                
                                if use_stable:
                                    print(f"   🎛️ Parameters: STABLE with BALANCED INCENTIVES")
                                    print(f"   � Environment: Balanced reward system (no reward hacking)")
                                    print(f"   🎯 Benefits: Discovery bonus +5, inefficiency penalty -2")
                                    print(f"   📈 Perfect for: Breaking through performance plateaus")
                                else:
                                    print(f"   🎛️ Parameters: ORIGINAL (preserves learned strategies)")
                                    print(f"   🤖 Environment: AUTONOMOUS LEARNING (minimal rewards)")
                                    print(f"   🎯 Benefits: Maintains existing training approach")
                                
                                confirm = input("\nStart continue training with manual control? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    print(f"\n🚀 Starting continue training with manual control...")
                                    print(f"📂 Base model: {selected_model_name}")
                                    print(f"👀 Watch for peak performance messages and auto-saves!")
                                    print(f"🛑 Press Ctrl+C to stop training when you see desired performance")
                                    print(f"🎯 Goal: Push your model to new performance peaks!")
                                    
                                    if use_stable:
                                        subprocess.run([PYTHON_PATH, "train.py", "continue", selected_model, str(additional_timesteps), "stable"])
                                    else:
                                        subprocess.run([PYTHON_PATH, "train.py", "continue", selected_model, str(additional_timesteps)])
                                break
                                
                            except ValueError:
                                print("❌ Invalid input. Please enter a number.")
                                
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                print("No trained models found in models/ directory.")
                print("Please train a model first (options 2, 3, 4, 10, or 11).")
        else:
            print("Models directory not found. Please train a model first.")
            
    elif choice == "14":
        print("\n📊 Extended Evaluation (Low Variance)...")
        print("🎯 Run 500-1000 episodes for much more stable performance estimates")
        print("📈 Includes confidence intervals and variance analysis")
        print("⏱️  Takes longer but provides reliable performance assessment")
        print("\n💡 Perfect for:")
        print("   • Final model validation")
        print("   • Comparing models with high confidence")
        print("   • Understanding true performance vs lucky streaks")
        
        # List available models
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            model_files.sort()  # Sort files alphabetically
            if model_files:
                print(f"\nAvailable models ({len(model_files)} found):")
                for i, model in enumerate(model_files, 1):
                    # Add indicators for special model types
                    if 'peak_performance_' in model:
                        indicator = "🏆"
                    elif 'stable' in model.lower():
                        indicator = "⚖️"
                    elif 'best_model' in model:
                        indicator = "🥇"
                    else:
                        indicator = "📋"
                    print(f"  {i}. {indicator} {model}")
                print(f"  {len(model_files) + 1}. Use default (best_model)")
                
                model_choice = input(f"\nChoose model for extended evaluation (1-{len(model_files) + 1}): ").strip()
                try:
                    choice_num = int(model_choice)
                    if 1 <= choice_num <= len(model_files):
                        selected_model = f"./models/{model_files[choice_num - 1]}"
                        selected_model_name = model_files[choice_num - 1]
                        
                        # Get number of episodes
                        while True:
                            try:
                                episodes_input = input("\nNumber of episodes (recommended 500-1000): ").strip()
                                num_episodes = int(episodes_input)
                                
                                if num_episodes < 100:
                                    print("⚠️  Warning: Too few episodes may still show high variance.")
                                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                                    if confirm != 'y':
                                        continue
                                
                                # Estimate time
                                estimated_minutes = num_episodes * 0.5  # Rough estimate
                                if estimated_minutes < 60:
                                    time_estimate = f"~{estimated_minutes:.0f} minutes"
                                else:
                                    time_estimate = f"~{estimated_minutes/60:.1f} hours"
                                
                                print(f"\n📋 EXTENDED EVALUATION:")
                                print(f"   📂 Model: {selected_model_name}")
                                print(f"   🎯 Episodes: {num_episodes:,}")
                                print(f"   📊 Output: Confidence intervals + variance analysis")
                                print(f"   🎯 Goal: Stable, reliable performance assessment")
                                print(f"   ⏱️ Estimated time: {time_estimate}")
                                
                                confirm = input("\nStart extended evaluation? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    print(f"\n🚀 Starting extended evaluation...")
                                    print(f"📊 Running {num_episodes} episodes for low-variance results...")
                                    
                                    # Run extended evaluation
                                    subprocess.run([PYTHON_PATH, "train.py", "5", selected_model, str(num_episodes)])
                                break
                                
                            except ValueError:
                                print("❌ Invalid input. Please enter a number.")
                                
                    elif choice_num == len(model_files) + 1:
                        print("Extended evaluation of default model: best_model")
                        episodes_input = input("Number of episodes (recommended 500-1000): ").strip()
                        try:
                            num_episodes = int(episodes_input)
                            subprocess.run([PYTHON_PATH, "train.py", "5", "./models/best_model", str(num_episodes)])
                        except ValueError:
                            print("Using default 500 episodes...")
                            subprocess.run([PYTHON_PATH, "train.py", "5", "./models/best_model", "500"])
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                print("No trained models found in models/ directory.")
                print("Please train a model first (options 2, 3, 4, 10, or 11).")
        else:
            print("Models directory not found. Please train a model first.")
            
    elif choice == "15":
        print("\n📚 Curriculum Learning Training...")
        print("🎓 Progressive difficulty training - start easy, get harder!")
        print("📈 Automatic progression based on performance milestones")
        print("🔍 NOW WITH BALANCED INCENTIVE SYSTEM!")
        print("� PERFECT for breaking through skill plateaus!")
        print("\n📊 Difficulty Levels:")
        print("   1. 🟢 Beginner: 8 flowers, 180 energy, fast regen (SKILL DISCOVERY)")
        print("   2. 🟡 Easy: 6 flowers, 120 energy, moderate costs") 
        print("   3. 🟠 Medium: 5 flowers, 100 energy (your current plateau)")
        print("   4. 🔴 Hard: 5 flowers, 80 energy, high costs")
        print("\n🎯 Auto-progression thresholds:")
        print("   • Beginner → Easy: 60% survival over 50 episodes")
        print("   • Easy → Medium: 50% survival over 100 episodes")
        print("   • Medium → Hard: 40% survival over 150 episodes")
        print("\n💡 BREAKTHROUGH STRATEGY:")
        print("   🔬 Learn advanced skills (pathfinding, cooldown mgmt) in forgiving beginner mode")
        print("   📈 Apply learned skills to progressively harder environments")
        print("   🚀 Break through your current 40% survival plateau!")
        
        print("\n🎛️ Training Options:")
        print("1. 🎓 Full Curriculum (Auto-progression through all levels)")
        print("2. 🎯 Start at specific difficulty level")
        print("3. 🔧 Manual difficulty control (no auto-progression)")
        
        curriculum_choice = input("\nChoose curriculum mode (1-3): ").strip()
        
        if curriculum_choice == "1":
            # Full curriculum training
            while True:
                try:
                    timesteps_input = input("\nTotal training timesteps (e.g., 5000000 for 5M): ").strip()
                    total_timesteps = int(timesteps_input)
                    
                    if total_timesteps < 500000:
                        print("⚠️  Warning: Curriculum learning needs substantial time.")
                        print("   Recommended minimum: 2,000,000 timesteps")
                        confirm = input("Continue anyway? (y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                    
                    # Convert to human-readable format
                    if total_timesteps >= 1000000:
                        readable = f"{total_timesteps/1000000:.1f}M"
                    elif total_timesteps >= 1000:
                        readable = f"{total_timesteps/1000:.0f}K"
                    else:
                        readable = str(total_timesteps)
                    
                    print(f"\n📚 FULL CURRICULUM TRAINING:")
                    print(f"   🎯 Total timesteps: {total_timesteps:,} ({readable})")
                    print(f"   🎓 Starting difficulty: BEGINNER")
                    print(f"   📈 Auto-progression: ENABLED")
                    print(f"   🏆 Target: Progress through all 4 difficulty levels")
                    print(f"   💾 Auto-save: Peak performance at each level")
                    
                    confirm = input("\nStart full curriculum training? (y/n): ").strip().lower()
                    if confirm == 'y':
                        print(f"\n🎓 Starting curriculum learning...")
                        print(f"📚 Beginning with BEGINNER difficulty...")
                        subprocess.run([PYTHON_PATH, "train.py", "curriculum", "beginner", "auto", str(total_timesteps)])
                    break
                    
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
                    
        elif curriculum_choice == "2":
            # Start at specific difficulty
            print("\n🎯 Choose starting difficulty:")
            print("1. 🟢 Beginner (recommended for new agents)")
            print("2. 🟡 Easy") 
            print("3. 🟠 Medium")
            print("4. 🔴 Hard (for advanced agents only)")
            
            diff_choice = input("Starting difficulty (1-4): ").strip()
            difficulty_map = {"1": "beginner", "2": "easy", "3": "medium", "4": "hard"}
            
            if diff_choice in difficulty_map:
                difficulty = difficulty_map[diff_choice]
                
                while True:
                    try:
                        timesteps_input = input(f"\nTraining timesteps for {difficulty.upper()} level: ").strip()
                        timesteps = int(timesteps_input)
                        
                        # Convert to human-readable format
                        if timesteps >= 1000000:
                            readable = f"{timesteps/1000000:.1f}M"
                        elif timesteps >= 1000:
                            readable = f"{timesteps/1000:.0f}K"
                        else:
                            readable = str(timesteps)
                        
                        print(f"\n📚 TARGETED CURRICULUM TRAINING:")
                        print(f"   🎯 Timesteps: {timesteps:,} ({readable})")
                        print(f"   🎓 Starting difficulty: {difficulty.upper()}")
                        print(f"   � Auto-progression: ENABLED")
                        print(f"   🏆 Can progress to higher difficulties")
                        
                        confirm = input(f"\nStart training at {difficulty.upper()} level? (y/n): ").strip().lower()
                        if confirm == 'y':
                            print(f"\n🎓 Starting curriculum learning at {difficulty.upper()} level...")
                            subprocess.run([PYTHON_PATH, "train.py", "curriculum", difficulty, "auto", str(timesteps)])
                        break
                        
                    except ValueError:
                        print("❌ Invalid input. Please enter a number.")
            else:
                print("❌ Invalid difficulty choice.")
                
        elif curriculum_choice == "3":
            # Manual difficulty control
            print("\n🔧 Manual Difficulty Control (No Auto-progression)")
            print("1. 🟢 Beginner")
            print("2. 🟡 Easy") 
            print("3. 🟠 Medium")
            print("4. 🔴 Hard")
            
            diff_choice = input("Fixed difficulty level (1-4): ").strip()
            difficulty_map = {"1": "beginner", "2": "easy", "3": "medium", "4": "hard"}
            
            if diff_choice in difficulty_map:
                difficulty = difficulty_map[diff_choice]
                
                while True:
                    try:
                        timesteps_input = input(f"\nTraining timesteps at {difficulty.upper()} level: ").strip()
                        timesteps = int(timesteps_input)
                        
                        print(f"\n📚 FIXED DIFFICULTY TRAINING:")
                        print(f"   🎯 Timesteps: {timesteps:,}")
                        print(f"   🎓 Fixed difficulty: {difficulty.upper()}")
                        print(f"   📈 Auto-progression: DISABLED")
                        print(f"   🔒 Will stay at this difficulty level")
                        
                        confirm = input(f"\nStart fixed {difficulty.upper()} training? (y/n): ").strip().lower()
                        if confirm == 'y':
                            print(f"\n🎓 Starting fixed difficulty training at {difficulty.upper()}...")
                            subprocess.run([PYTHON_PATH, "train.py", "curriculum", difficulty, "manual", str(timesteps)])
                        break
                        
                    except ValueError:
                        print("❌ Invalid input. Please enter a number.")
            else:
                print("❌ Invalid difficulty choice.")
        else:
            print("❌ Invalid curriculum mode choice.")
            
    elif choice == "16":
        print("\n🎓🚀 Continue Mastery Model with Enhanced Curriculum")
        print("=" * 55)
        print("🎯 Perfect for models that achieved medium mastery!")
        print("🧠 Enhanced with Great Filter solution:")
        print("   • 🌉 Pre-hard bridge stage (smooth difficulty transition)")
        print("   • 👁️ Environment parameter awareness (immediate adaptation)")
        print("🚀 Break through the 50%+ survival barrier!")
        
        if os.path.exists("models"):
            # List available mastery models
            print("\n📁 Available Mastery Models:")
            model_files = []
            for file in os.listdir("models"):
                if file.endswith(".zip"):
                    model_files.append(file)
            
            if model_files:
                # Sort by performance indicators (prioritize 50%+ models)
                priority_models = []
                good_models = []
                other_models = []
                
                for model in model_files:
                    if "50.0%" in model:
                        priority_models.append(model)
                    elif any(perf in model for perf in ["48.0%", "46.0%", "44.0%", "42.0%"]):
                        good_models.append(model)
                    else:
                        other_models.append(model)
                
                # Display prioritized list
                all_models = priority_models + good_models + other_models
                for i, model in enumerate(all_models[:15], 1):  # Show top 15
                    if "50.0%" in model:
                        print(f"   {i}. 🎯 {model} (MASTERY LEVEL!)")
                    elif any(perf in model for perf in ["48.0%", "46.0%", "44.0%"]):
                        print(f"   {i}. 🌟 {model} (HIGH PERFORMANCE)")
                    elif any(perf in model for perf in ["42.0%", "40.0%"]):
                        print(f"   {i}. 📈 {model} 15(GOOD CANDIDATE)")
                    else:
                        print(f"   {i}. 📄 {model}")
                
                if len(all_models) > 15:
                    print(f"   ... and {len(all_models) - 15} more models")
                
                # Add best_model option
                print(f"   {min(15, len(all_models)) + 1}. 🥇 Use default (best_model)")
                
                print("\n🎯 RECOMMENDATION: Choose the 50.0% survival model for best results!")
                
                try:
                    model_choice = int(input(f"\nSelect model (1-{min(15, len(all_models)) + 1}): ").strip())
                    if 1 <= model_choice <= min(15, len(all_models)):
                        selected_model = all_models[model_choice - 1]
                        
                        # Training timesteps
                        while True:
                            try:
                                timesteps_input = input("\nTraining timesteps (e.g., 3000000 for enhanced curriculum): ").strip()
                                timesteps = int(timesteps_input)
                                
                                if timesteps >= 1000000:
                                    readable = f"{timesteps/1000000:.1f}M"
                                elif timesteps >= 1000:
                                    readable = f"{timesteps/1000:.0f}K"
                                else:
                                    readable = str(timesteps)
                                
                                print(f"\n🎓 ENHANCED CURRICULUM CONTINUATION:")
                                print(f"   📦 Base model: {selected_model}")
                                print(f"   🎯 Timesteps: {timesteps:,} ({readable})")
                                print(f"   🌉 Bridge stage: pre_hard (ENABLED)")
                                print(f"   👁️ Parameter awareness: ENABLED")
                                print(f"   📈 Auto-progression: From medium → pre_hard → hard")
                                print(f"   🏆 Target: 50%+ sustained survival in hard mode")
                                
                                confirm = input(f"\nContinue training with enhanced curriculum? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    print(f"\n🎓 Starting enhanced curriculum training...")
                                    print(f"🚀 Loading mastery model and applying Great Filter solution...")
                                    # Use the continue mastery curriculum script
                                    subprocess.run([PYTHON_PATH, "continue_mastery_curriculum.py", selected_model, str(timesteps), "medium"])
                                break
                                
                            except ValueError:
                                print("❌ Invalid input. Please enter a number.")
                    elif model_choice == min(15, len(all_models)) + 1:
                        # Handle best_model selection
                        selected_model = "best_model.zip"
                        
                        # Training timesteps
                        while True:
                            try:
                                timesteps_input = input("\nTraining timesteps (e.g., 3000000 for enhanced curriculum): ").strip()
                                timesteps = int(timesteps_input)
                                
                                if timesteps >= 1000000:
                                    readable = f"{timesteps/1000000:.1f}M"
                                elif timesteps >= 1000:
                                    readable = f"{timesteps/1000:.0f}K"
                                else:
                                    readable = str(timesteps)
                                
                                print(f"\n🎓 ENHANCED CURRICULUM CONTINUATION:")
                                print(f"   📦 Base model: best_model.zip (default)")
                                print(f"   🎯 Timesteps: {timesteps:,} ({readable})")
                                print(f"   🌉 Bridge stage: pre_hard (ENABLED)")
                                print(f"   👁️ Parameter awareness: ENABLED")
                                print(f"   📈 Auto-progression: From medium → pre_hard → hard")
                                print(f"   🏆 Target: 50%+ sustained survival in hard mode")
                                
                                confirm = input(f"\nContinue training with enhanced curriculum? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    print(f"\n🎓 Starting enhanced curriculum training...")
                                    print(f"🚀 Loading default model and applying Great Filter solution...")
                                    # Use the continue mastery curriculum script
                                    subprocess.run([PYTHON_PATH, "continue_mastery_curriculum.py", selected_model, str(timesteps), "medium"])
                                break
                                
                            except ValueError:
                                print("❌ Invalid input. Please enter a number.")
                    else:
                        print("❌ Invalid model choice.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
            else:
                print("❌ No model files found in models/ directory.")
                print("   Please train a model first or use curriculum learning (option 15).")
        else:
            print("❌ Models directory not found.")
            print("   Please train a model first.")
            
    elif choice == "17":
        print("\n🎯🔥 Specialist HARD Mode Training")
        print("=" * 55)
        print("🎓 Transform curriculum graduates into HARD mode experts!")
        print("🧪 Ultra-low learning rate precision fine-tuning")
        print("🎯 Goal: Achieve 50%+ survival in HARD mode exclusively")
        print("🔬 Perfect for models that completed curriculum training")
        
        if os.path.exists("models"):
            # List available models for specialist training
            print("\n📁 Available Models for Specialist Training:")
            model_files = []
            for file in os.listdir("models"):
                if file.endswith(".zip"):
                    model_files.append(file)
            
            if model_files:
                # Sort by relevance (prioritize curriculum graduates and high performers)
                curriculum_models = []
                peak_models = []
                other_models = []
                
                for model in model_files:
                    if "curriculum" in model.lower() or "mastery" in model.lower():
                        curriculum_models.append(model)
                    elif any(perf in model for perf in ["40.0%", "42.0%", "44.0%", "46.0%", "48.0%", "50.0%"]):
                        peak_models.append(model)
                    else:
                        other_models.append(model)
                
                # Display prioritized list
                all_models = curriculum_models + peak_models + other_models
                for i, model in enumerate(all_models[:15], 1):  # Show top 15
                    if "curriculum" in model.lower() or "mastery" in model.lower():
                        print(f"   {i}. 🎓 {model} (CURRICULUM GRADUATE - IDEAL!)")
                    elif any(perf in model for perf in ["50.0%", "48.0%", "46.0%"]):
                        print(f"   {i}. 🌟 {model} (HIGH PERFORMANCE)")
                    elif any(perf in model for perf in ["44.0%", "42.0%", "40.0%"]):
                        print(f"   {i}. 📈 {model} (GOOD CANDIDATE)")
                    else:
                        print(f"   {i}. 📄 {model}")
                
                if len(all_models) > 15:
                    print(f"   ... and {len(all_models) - 15} more models")
                
                # Add best_model option
                print(f"   {min(15, len(all_models)) + 1}. 🥇 Use default (best_model)")
                
                print("\n🎯 RECOMMENDATION: Choose a curriculum graduate for best results!")
                
                try:
                    model_choice = int(input(f"\nSelect model (1-{min(15, len(all_models)) + 1}): ").strip())
                    if 1 <= model_choice <= min(15, len(all_models)):
                        selected_model = f"models/{all_models[model_choice - 1]}"
                        
                        # Training timesteps for specialist training
                        while True:
                            try:
                                print("\n🎯 Specialist Training Timesteps:")
                                print("   • 5M timesteps: Quick specialist training (~2-3 hours)")
                                print("   • 10M timesteps: Standard specialist training (~4-6 hours)")
                                print("   • 15M+ timesteps: Deep specialist training (6+ hours)")
                                
                                timesteps_input = input("\nTraining timesteps (e.g., 10000000 for 10M): ").strip()
                                timesteps = int(timesteps_input)
                                
                                if timesteps < 1000000:
                                    print("⚠️ Warning: Specialist training needs at least 1M timesteps for effectiveness")
                                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                                    if confirm != 'y':
                                        continue
                                
                                if timesteps >= 1000000:
                                    readable = f"{timesteps/1000000:.1f}M"
                                elif timesteps >= 1000:
                                    readable = f"{timesteps/1000:.0f}K"
                                else:
                                    readable = str(timesteps)
                                
                                print(f"\n🎯 SPECIALIST HARD MODE TRAINING:")
                                print(f"   📦 Base model: {all_models[model_choice - 1]}")
                                print(f"   🎯 Timesteps: {timesteps:,} ({readable})")
                                print(f"   🔬 Learning rate: 5e-6 (ultra-low precision)")
                                print(f"   🎯 Environment: HARD-locked (no progression)")
                                print(f"   📈 Entropy: 0.001 (minimal exploration)")
                                print(f"   🏆 Target: 50%+ survival in HARD mode")
                                
                                confirm = input(f"\nStart specialist training? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    print(f"\n🎯 Starting specialist HARD mode training...")
                                    print(f"🧪 Transforming curriculum graduate into HARD mode expert...")
                                    subprocess.run([PYTHON_PATH, "train.py", "specialist", selected_model, str(timesteps)])
                                break
                                
                            except ValueError:
                                print("❌ Invalid input. Please enter a number.")
                    elif model_choice == min(15, len(all_models)) + 1:
                        # Handle best_model selection
                        selected_model = "models/best_model.zip"
                        
                        # Training timesteps for specialist training
                        while True:
                            try:
                                print("\n🎯 Specialist Training Timesteps:")
                                print("   • 5M timesteps: Quick specialist training (~2-3 hours)")
                                print("   • 10M timesteps: Standard specialist training (~4-6 hours)")
                                print("   • 15M+ timesteps: Deep specialist training (6+ hours)")
                                
                                timesteps_input = input("\nTraining timesteps (e.g., 10000000 for 10M): ").strip()
                                timesteps = int(timesteps_input)
                                
                                if timesteps < 1000000:
                                    print("⚠️ Warning: Specialist training needs at least 1M timesteps for effectiveness")
                                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                                    if confirm != 'y':
                                        continue
                                
                                if timesteps >= 1000000:
                                    readable = f"{timesteps/1000000:.1f}M"
                                elif timesteps >= 1000:
                                    readable = f"{timesteps/1000:.0f}K"
                                else:
                                    readable = str(timesteps)
                                
                                print(f"\n🎯 SPECIALIST HARD MODE TRAINING:")
                                print(f"   📦 Base model: best_model.zip (default)")
                                print(f"   🎯 Timesteps: {timesteps:,} ({readable})")
                                print(f"   🔬 Learning rate: 5e-6 (ultra-low precision)")
                                print(f"   🎯 Environment: HARD-locked (no progression)")
                                print(f"   📈 Entropy: 0.001 (minimal exploration)")
                                print(f"   🏆 Target: 50%+ survival in HARD mode")
                                
                                confirm = input(f"\nStart specialist training? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    print(f"\n🎯 Starting specialist HARD mode training...")
                                    print(f"🧪 Transforming default model into HARD mode expert...")
                                    subprocess.run([PYTHON_PATH, "train.py", "specialist", selected_model, str(timesteps)])
                                break
                                
                            except ValueError:
                                print("❌ Invalid input. Please enter a number.")
                    else:
                        print("❌ Invalid model choice.")
                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
            else:
                print("❌ No model files found in models/ directory.")
                print("   Please train a model first or use curriculum learning (option 15).")
        else:
            print("❌ Models directory not found.")
            print("   Please train a model first.")
            
    elif choice == "18":
        print("\n🏆 Evaluating Top Performers...")
        print("=" * 60)
        print("📊 Comprehensive statistical evaluation of your best models")
        print("🎯 1000 episodes each (10 runs × 100 episodes)")
        print("📈 Generates confidence intervals, variance analysis, and rankings")
        print("🔬 Perfect for identifying your most reliable high-performers")
        print("\n💡 Target Models:")
        print("   • peak_performance_4200k_survival_42.0%")
        print("   • peak_performance_3950k_survival_41.0%")
        print("   • peak_performance_5050k_survival_41.0%")
        print("   • stable_autonomous_28_14000k_stable")
        print("   • peak_performance_4200k_survival_39.0%")
        print("   • peak_performance_1600k_survival_41.0%")
        print("   • peak_performance_300k_survival_35.0%")
        print("   • peak_performance_1400k_survival_42.0%")
        
        print("\n⏱️ Estimated time: 30-60 minutes for all models")
        print("📊 Outputs: CSV data, JSON results, and visualizations")
        
        confirm = input("\nStart comprehensive top performers evaluation? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\n🚀 Starting comprehensive evaluation...")
            print("📊 This will run 8000 total episodes across your top models")
            print("🎯 Results will be saved in evaluation_results/ directory")
            subprocess.run([PYTHON_PATH, "evaluate_top_performers.py"])
        else:
            print("Evaluation cancelled.")
    
    elif choice == "0":
        print("\n�👋 Goodbye!")
        sys.exit(0)
        
    else:
        print("\n❌ Invalid choice. Please enter 0-18.")
    
    # Ask if user wants to continue
    print("\n" + "=" * 45)
    continue_choice = input("Return to menu? (y/n): ").strip().lower()
    if continue_choice == 'y':
        print("\n" * 2)  # Clear space
        main()  # Recursive call to show menu again
    else:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
