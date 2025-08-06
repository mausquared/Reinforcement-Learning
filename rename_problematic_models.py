#!/usr/bin/env python3
"""
Script to test peak_performance models and rename problematic ones
"""

import os
import sys
import shutil
from stable_baselines3 import PPO

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import create_environment_for_model

def test_model_loading(model_path):
    """Test if a model can be loaded and used without errors."""
    try:
        print(f"   Testing model loading...")
        # Try to load the model
        model = PPO.load(model_path)
        
        print(f"   Creating compatible environment...")
        # Try to create a compatible environment
        env = create_environment_for_model(model_path, render_mode=None)
        
        print(f"   Testing environment reset...")
        # Try to reset the environment
        obs, info = env.reset()
        
        print(f"   Testing model prediction...")
        # Try to get an action from the model
        action, _states = model.predict(obs, deterministic=True)
        
        print(f"   Testing environment step...")
        # Try to take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()
        print(f"   ✅ Model test PASSED")
        return True
        
    except Exception as e:
        print(f"   ❌ Model test FAILED: {str(e)}")
        return False

def rename_model_files(original_path, new_name):
    """Rename model file and associated files."""
    base_name = os.path.splitext(original_path)[0]  # Remove .zip extension
    new_base_name = os.path.splitext(new_name)[0]
    
    files_to_rename = []
    
    # Check for related files
    potential_files = [
        f"{base_name}.zip",
        f"{base_name}_summary.txt",
        f"{base_name}_training_stats.pkl",
        f"{base_name}_3d_matplotlib_training_analysis.png"
    ]
    
    for file_path in potential_files:
        if os.path.exists(file_path):
            files_to_rename.append(file_path)
    
    # Rename all found files
    renamed_files = []
    for file_path in files_to_rename:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        
        # Create new filename
        if file_ext == '.zip':
            new_file_name = f"{new_base_name}.zip"
        elif file_ext == '.txt':
            new_file_name = f"{new_base_name}_summary.txt"
        elif file_ext == '.pkl':
            new_file_name = f"{new_base_name}_training_stats.pkl"
        elif file_ext == '.png':
            new_file_name = f"{new_base_name}_3d_matplotlib_training_analysis.png"
        else:
            continue
            
        new_file_path = os.path.join(file_dir, new_file_name)
        
        try:
            shutil.move(file_path, new_file_path)
            renamed_files.append((file_path, new_file_path))
            print(f"      📁 {os.path.basename(file_path)} → {os.path.basename(new_file_path)}")
        except Exception as e:
            print(f"      ⚠️ Failed to rename {file_path}: {e}")
    
    return renamed_files

def main():
    print("🔍 Peak Performance Model Error Detection & Renaming Script")
    print("=" * 60)
    print("Purpose: Test all peak_performance models and rename problematic ones")
    print("Action: Add '_cumulative' suffix to models that cause errors")
    print("=" * 60)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Models directory not found!")
        return
    
    # Find all peak_performance models
    all_files = os.listdir(models_dir)
    peak_models = [f for f in all_files if f.startswith('peak_performance_') and f.endswith('.zip')]
    
    if not peak_models:
        print("❌ No peak_performance models found!")
        return
    
    print(f"📊 Found {len(peak_models)} peak_performance models to test")
    print()
    
    working_models = []
    problematic_models = []
    
    for i, model_file in enumerate(peak_models, 1):
        model_path = os.path.join(models_dir, model_file)
        print(f"🧪 Testing {i}/{len(peak_models)}: {model_file}")
        
        if test_model_loading(model_path):
            working_models.append(model_file)
            print(f"   ✅ Model works correctly - keeping original name")
        else:
            problematic_models.append(model_file)
            print(f"   ❌ Model has errors - will rename with '_cumulative' suffix")
        
        print()
    
    # Summary of results
    print("📊 TESTING RESULTS:")
    print("=" * 40)
    print(f"✅ Working models: {len(working_models)}")
    print(f"❌ Problematic models: {len(problematic_models)}")
    print()
    
    if problematic_models:
        print("🔧 RENAMING PROBLEMATIC MODELS:")
        print("-" * 40)
        
        for model_file in problematic_models:
            # Create new name with _cumulative suffix
            base_name = model_file.replace('.zip', '')
            new_name = f"{base_name}_cumulative.zip"
            
            print(f"📝 Renaming: {model_file}")
            print(f"   New name: {new_name}")
            
            model_path = os.path.join(models_dir, model_file)
            new_path = os.path.join(models_dir, new_name)
            
            # Rename the model and associated files
            renamed_files = rename_model_files(model_path, new_path)
            
            if renamed_files:
                print(f"   ✅ Successfully renamed {len(renamed_files)} files")
            else:
                print(f"   ⚠️ No files were renamed")
            print()
    
    # Final summary
    print("🎯 FINAL SUMMARY:")
    print("=" * 40)
    print(f"✅ Models that work correctly: {len(working_models)}")
    print(f"🔧 Models renamed with '_cumulative': {len(problematic_models)}")
    
    if working_models:
        print(f"\n📋 Working models (unchanged):")
        for model in working_models:
            print(f"   ✅ {model}")
    
    if problematic_models:
        print(f"\n📋 Renamed models (had errors):")
        for model in problematic_models:
            base_name = model.replace('.zip', '')
            new_name = f"{base_name}_cumulative.zip"
            print(f"   🔧 {model} → {new_name}")
    
    print(f"\n🎉 Script completed successfully!")
    print(f"💡 You can now use Option 7 in the launcher to evaluate the working models")

if __name__ == "__main__":
    main()
