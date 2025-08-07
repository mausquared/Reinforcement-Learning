#!/usr/bin/env python3
"""
🐦 3D Hummingbird Environment - Quick Launcher
Simple launcher for the 3D matplotlib hummingbird environment
"""

import os
import sys
import subprocess
import re  # Import the regular expressions module

# Python executable path for this virtual environment
PYTHON_PATH = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"

def get_timesteps_from_filename(filename):
    """Extracts the number of thousands of timesteps from a model filename."""
    # Handles formats like 'name_5000k.zip'
    match = re.search(r'_(\d+)[kK]\.zip$', filename)
    if match:
        return int(match.group(1)) * 1000
    return 0

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
    print("10. 🏆 Compare Top Performing Models")
    print("11. 🔍 Check Model Compatibility")
    print("12. 🔬 Detailed Model Evaluation")
    print("0. ❌ Exit")
    
    choice = input("\nEnter your choice (0-12): ").strip()
    
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
                        selected_model_filename = model_files[choice_num - 1]
                        selected_model = f"./models/{selected_model_filename}"
                        
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
                                
                                # --- NEW: Calculate and display total timesteps ---
                                base_timesteps = get_timesteps_from_filename(selected_model_filename)
                                total_timesteps = base_timesteps + additional_timesteps
                                
                                # Convert to human-readable format
                                if additional_timesteps >= 1000000:
                                    readable_add = f"{additional_timesteps/1000000:.1f}M"
                                elif additional_timesteps >= 1000:
                                    readable_add = f"{additional_timesteps/1000:.0f}K"
                                else:
                                    readable_add = str(additional_timesteps)

                                if total_timesteps >= 1000000:
                                    readable_total = f"{total_timesteps/1000000:.1f}M"
                                else:
                                    readable_total = f"{total_timesteps/1000:.0f}K"
                                
                                # Estimate training time
                                estimated_minutes = additional_timesteps / 10000
                                if estimated_minutes < 60:
                                    time_estimate = f"~{estimated_minutes:.0f} minutes"
                                else:
                                    time_estimate = f"~{estimated_minutes/60:.1f} hours"
                                
                                print(f"\n📊 Continue Training Configuration:")
                                print(f"   Base model: {selected_model_filename}")
                                print(f"   Existing timesteps: {base_timesteps:,}")
                                print(f"   Additional timesteps: {additional_timesteps:,} ({readable_add})")
                                print(f"   New total timesteps:  {total_timesteps:,} ({readable_total})")
                                print(f"   Estimated time: {time_estimate}")
                                print(f"   Training mode: CONTINUE EXISTING")
                                
                                confirm = input("\nContinue training? (y/n): ").strip().lower()
                                if confirm == 'y':
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
        print("\n🏆 Compare Top Performing Models...")
        print("Statistical comparison of best models with 10 runs × 100 episodes each")
        print("This will evaluate models with 1000 episodes total per model")
        print("⏱️  This may take 30-60 minutes depending on number of models")
        
        confirm = input("\nProceed with comprehensive model comparison? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([PYTHON_PATH, "model_comparison.py"])
        else:
            print("Model comparison cancelled.")
            
    elif choice == "11":
        print("\n🔍 Check Model Compatibility...")
        print("Check all models for environment compatibility")
        print("Incompatible models will be renamed with LEGACY_ prefix")
        print("⚠️  This will automatically rename incompatible models")
        
        confirm = input("\nProceed with compatibility check and auto-rename? (y/n): ").strip().lower()
        if confirm == 'y':
            subprocess.run([PYTHON_PATH, "model_compatibility_checker.py"])
        else:
            print("Compatibility check cancelled.")
            
    elif choice == "12":
        print("\n🔬 Detailed Model Evaluation...")
        print("Run a rigorous statistical analysis on a single model.")
        
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            model_files.sort(reverse=True)
            
            if model_files:
                print(f"\nAvailable models ({len(model_files)} found):")
                for i, model in enumerate(model_files, 1):
                    print(f"  {i}. {model}")
                
                model_choice = input(f"\nChoose model for detailed evaluation (1-{len(model_files)}): ").strip()
                try:
                    choice_num = int(model_choice)
                    if 1 <= choice_num <= len(model_files):
                        selected_model = f"./models/{model_files[choice_num - 1]}"
                        print(f"\nStarting detailed evaluation for: {model_files[choice_num - 1]}")
                        subprocess.run([PYTHON_PATH, "detailed_evaluation.py", selected_model])
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
            else:
                print("No trained models found in models/ directory.")
        else:
            print("Models directory not found.")
            
    elif choice == "0":
        print("\n👋 Goodbye!")
        sys.exit(0)
        
    else:
        print("\n❌ Invalid choice. Please enter 0-12.")
    
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
