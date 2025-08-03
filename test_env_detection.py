#!/usr/bin/env python3
"""Test environment detection for models"""

from train import get_model_environment_version

def test_environment_detection():
    """Test environment detection for different model types."""
    print("🔍 Testing Environment Detection:")
    print("=" * 60)
    
    test_models = [
        'peak_performance_2950k_survival_46.0%.zip',
        'stable_autonomous_28_14000k.zip', 
        'autonomous_training_28_10000k.zip',
        'training_14_3500k.zip',
        'best_model.zip'
    ]
    
    for model in test_models:
        env_type = get_model_environment_version(f'models/{model}')
        print(f'{model:50} -> {env_type:>10}')
    
    print("\n💡 Environment Types:")
    print("   stable     = StableHummingbirdEnv (with survival rewards)")
    print("   autonomous = ComplexHummingbird3DMatplotlibEnv (minimal rewards)")
    print("   legacy     = Old environment (not used)")
    
    print("\n🚨 POTENTIAL ISSUE:")
    print("If peak_performance models show 'autonomous' instead of 'stable',")
    print("that explains the performance mismatch!")

if __name__ == "__main__":
    test_environment_detection()
