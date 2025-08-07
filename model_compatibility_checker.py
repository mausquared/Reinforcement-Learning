#!/usr/bin/env python3
"""
🔍 Model Compatibility Checker
Checks all models for environment compatibility and renames incompatible ones with LEGACY_ prefix
"""

import os
import sys
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_model_compatibility(model_path: str) -> Dict:
    """Check if a model is compatible with the current environment."""
    try:
        # Import required modules
        from stable_baselines3 import PPO
        from train import create_environment_for_model
        
        model_name = os.path.basename(model_path)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return {
                'model': model_name,
                'compatible': False,
                'error': f"Model file not found: {model_path}",
                'model_obs_space': None,
                'env_obs_space': None
            }
        
        # Try to load the model
        try:
            print(f"  📦 Loading model: {model_name}")
            model = PPO.load(model_path)
            model_obs_space = model.observation_space
            print(f"    Model observation space: {model_obs_space}")
        except Exception as e:
            return {
                'model': model_name,
                'compatible': False,
                'error': f"Failed to load model: {e}",
                'model_obs_space': None,
                'env_obs_space': None
            }
        
        # Try to create environment for this model
        try:
            print(f"  🌍 Creating environment for model...")
            env = create_environment_for_model(model_path, render_mode=None)
            env_obs_space = env.observation_space
            print(f"    Environment observation space: {env_obs_space}")
            
            # Test if we can reset the environment and get an observation
            obs, info = env.reset()
            
            # Check observation space compatibility
            compatible = True
            error_details = []
            
            # Check if it's a dict space (current environment)
            if hasattr(env_obs_space, 'spaces'):
                # Dict observation space
                if hasattr(model_obs_space, 'spaces'):
                    # Both are dict spaces - check each component
                    for key in env_obs_space.spaces:
                        if key not in model_obs_space.spaces:
                            compatible = False
                            error_details.append(f"Missing key '{key}' in model observation space")
                        else:
                            env_shape = env_obs_space.spaces[key].shape
                            model_shape = model_obs_space.spaces[key].shape
                            if env_shape != model_shape:
                                compatible = False
                                error_details.append(f"Shape mismatch for '{key}': model expects {model_shape}, env provides {env_shape}")
                    
                    # Check for extra keys in model
                    for key in model_obs_space.spaces:
                        if key not in env_obs_space.spaces:
                            compatible = False
                            error_details.append(f"Extra key '{key}' in model observation space")
                else:
                    # Model expects simple space, env provides dict
                    compatible = False
                    error_details.append("Model expects simple observation space, environment provides dict space")
            else:
                # Simple observation space
                if hasattr(model_obs_space, 'spaces'):
                    # Model expects dict, env provides simple
                    compatible = False
                    error_details.append("Model expects dict observation space, environment provides simple space")
                else:
                    # Both are simple spaces
                    env_shape = env_obs_space.shape
                    model_shape = model_obs_space.shape
                    if env_shape != model_shape:
                        compatible = False
                        error_details.append(f"Shape mismatch: model expects {model_shape}, env provides {env_shape}")
            
            # Test model prediction with current observation
            if compatible:
                try:
                    action, _states = model.predict(obs, deterministic=True)
                    print(f"    ✅ Model prediction test successful")
                except Exception as pred_error:
                    compatible = False
                    error_details.append(f"Model prediction failed: {pred_error}")
            
            env.close()
            
            return {
                'model': model_name,
                'compatible': compatible,
                'error': "; ".join(error_details) if error_details else None,
                'model_obs_space': str(model_obs_space),
                'env_obs_space': str(env_obs_space)
            }
            
        except Exception as e:
            return {
                'model': model_name,
                'compatible': False,
                'error': f"Environment creation/test failed: {e}",
                'model_obs_space': str(model_obs_space) if 'model_obs_space' in locals() else None,
                'env_obs_space': None
            }
        
    except Exception as e:
        return {
            'model': os.path.basename(model_path),
            'compatible': False,
            'error': f"Compatibility check failed: {e}",
            'model_obs_space': None,
            'env_obs_space': None
        }

def rename_incompatible_model(model_path: str) -> bool:
    """Rename an incompatible model with LEGACY_ prefix."""
    try:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        
        # Check if already has LEGACY_ prefix
        if model_name.startswith('LEGACY_'):
            print(f"    ℹ️  Model already has LEGACY_ prefix: {model_name}")
            return True
        
        # Create new name with LEGACY_ prefix
        new_name = f"LEGACY_{model_name}"
        new_path = os.path.join(model_dir, new_name)
        
        # Check if target name already exists
        if os.path.exists(new_path):
            # Add timestamp to make unique
            timestamp = int(time.time())
            base_name = model_name.replace('.zip', '')
            new_name = f"LEGACY_{base_name}_{timestamp}.zip"
            new_path = os.path.join(model_dir, new_name)
        
        # Rename the file
        os.rename(model_path, new_path)
        print(f"    ✅ Renamed to: {new_name}")
        return True
        
    except Exception as e:
        print(f"    ❌ Failed to rename {os.path.basename(model_path)}: {e}")
        return False

def check_all_models():
    """Main function to check all models and rename incompatible ones."""
    print("🔍 MODEL COMPATIBILITY CHECKER")
    print("=" * 60)
    print("Checking all models for environment compatibility")
    print("Incompatible models will be renamed with LEGACY_ prefix")
    print("=" * 60)
    
    # Find all models
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return
    
    model_files = list(models_dir.glob("*.zip"))
    if not model_files:
        print("❌ No model files found!")
        return
    
    # Filter out models that already have LEGACY_ prefix for initial check
    legacy_models = [f for f in model_files if f.name.startswith('LEGACY_')]
    regular_models = [f for f in model_files if not f.name.startswith('LEGACY_')]
    
    print(f"Found {len(model_files)} total models:")
    print(f"  📦 {len(regular_models)} regular models to check")
    print(f"  📚 {len(legacy_models)} already marked as LEGACY")
    print()
    
    if legacy_models:
        print("📚 Already marked as LEGACY:")
        for model in legacy_models:
            print(f"  - {model.name}")
        print()
    
    if not regular_models:
        print("✅ No regular models to check - all are already marked as LEGACY")
        return
    
    # Check each regular model
    compatible_models = []
    incompatible_models = []
    failed_checks = []
    renamed_models = []
    
    for i, model_path in enumerate(regular_models, 1):
        model_name = model_path.name
        print(f"🔍 Checking Model {i}/{len(regular_models)}: {model_name}")
        print("-" * 50)
        
        # Check compatibility
        result = check_model_compatibility(str(model_path))
        
        if result['compatible']:
            print(f"  ✅ COMPATIBLE")
            compatible_models.append(model_name)
        else:
            print(f"  ❌ INCOMPATIBLE: {result['error']}")
            incompatible_models.append(model_name)
            
            # Ask for confirmation before renaming
            print(f"  🔄 Renaming to LEGACY_{model_name}...")
            if rename_incompatible_model(str(model_path)):
                renamed_models.append(model_name)
            else:
                failed_checks.append(model_name)
        
        print()
    
    # Summary
    print("=" * 60)
    print("🔍 COMPATIBILITY CHECK SUMMARY")
    print("=" * 60)
    
    print(f"✅ Compatible models ({len(compatible_models)}):")
    for model in compatible_models:
        print(f"   - {model}")
    
    if incompatible_models:
        print(f"\n❌ Incompatible models ({len(incompatible_models)}):")
        for model in incompatible_models:
            status = "✅ Renamed" if model in renamed_models else "❌ Rename failed"
            print(f"   - {model} ({status})")
    
    if failed_checks:
        print(f"\n⚠️  Failed to rename ({len(failed_checks)}):")
        for model in failed_checks:
            print(f"   - {model}")
    
    print(f"\n📊 Final Status:")
    print(f"   📦 Compatible models: {len(compatible_models)}")
    print(f"   📚 Successfully renamed to LEGACY: {len(renamed_models)}")
    print(f"   ⚠️  Failed renames: {len(failed_checks)}")
    print(f"   📚 Total LEGACY models: {len(legacy_models) + len(renamed_models)}")
    
    # Recommendations
    if incompatible_models:
        print(f"\n💡 Recommendations:")
        print(f"   - LEGACY models are incompatible with current environment")
        print(f"   - They were likely trained with older observation spaces")
        print(f"   - Use compatible models for evaluation and comparison")
        print(f"   - Consider retraining LEGACY models with current environment")

def quick_compatibility_check(model_path: str) -> bool:
    """Quick check if a single model is compatible (returns True/False)."""
    result = check_model_compatibility(model_path)
    return result['compatible']

if __name__ == "__main__":
    try:
        check_all_models()
        
    except KeyboardInterrupt:
        print("\n❌ Compatibility check cancelled by user")
    except Exception as e:
        print(f"❌ Error during compatibility check: {e}")
        import traceback
        traceback.print_exc()
