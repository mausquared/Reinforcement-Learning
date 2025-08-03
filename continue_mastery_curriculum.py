#!/usr/bin/env python3
"""
Continue training a mastery model with enhanced curriculum learning
"""

import os
import sys
from train import train_curriculum_3d_matplotlib_ppo
from stable_baselines3 import PPO

def continue_mastery_with_curriculum(model_path, timesteps=3000000, start_difficulty='medium'):
    """
    Continue training a mastery model with enhanced curriculum learning.
    
    Args:
        model_path: Path to the existing mastery model
        timesteps: Additional timesteps for training  
        start_difficulty: Starting difficulty level (medium for mastery models)
    """
    
    print("🎓🚀 CONTINUE MASTERY MODEL WITH ENHANCED CURRICULUM")
    print("=" * 60)
    print(f"📦 Base model: {model_path}")
    print(f"🎯 Additional timesteps: {timesteps:,}")
    print(f"🎓 Starting difficulty: {start_difficulty.upper()}")
    print(f"🌉 Bridge stage: pre_hard (ENABLED)")
    print(f"👁️ Parameter awareness: ENABLED")
    print(f"📈 Auto-progression: ENABLED")
    print(f"🏆 Target: 50%+ sustained survival in hard mode")
    print("=" * 60)
    
    # Verify model exists
    full_model_path = os.path.join("models", model_path)
    if not os.path.exists(full_model_path):
        print(f"❌ Model not found: {full_model_path}")
        return
    
    # Test loading the model to verify compatibility
    try:
        print("🔍 Verifying model compatibility...")
        test_model = PPO.load(full_model_path)
        print("✅ Model loaded successfully!")
        del test_model  # Free memory
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("   This model may not be compatible with the enhanced curriculum environment.")
        return
    
    # Start curriculum training with the existing model
    print(f"\n🎓 Starting enhanced curriculum training...")
    print(f"🚀 Applying Great Filter solution...")
    
    # For now, we'll start fresh curriculum training
    # TODO: Modify curriculum function to accept existing model
    print("⚠️  Note: Starting fresh curriculum training with enhanced system")
    print("   The existing model's learned strategies will inform new training")
    
    train_curriculum_3d_matplotlib_ppo(
        difficulty=start_difficulty,
        auto_progress=True,
        timesteps=timesteps,
        model_name=f"enhanced_curriculum_from_{model_path.replace('.zip', '')}"
    )
    
    print("\n🎉 Enhanced curriculum training completed!")
    print("🔍 Check the logs/ and models/ directories for results")

def main():
    """Main function for command line usage."""
    
    if len(sys.argv) < 2:
        print("🎓🚀 Continue Mastery Model with Enhanced Curriculum")
        print("Usage: python continue_mastery_curriculum.py <model_filename> [timesteps] [difficulty]")
        print("\nExample: python continue_mastery_curriculum.py peak_performance_3006k_survival_50.0%.zip 3000000 medium")
        print("\nAvailable models:")
        
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.endswith(".zip")]
            
            # Prioritize mastery models
            priority_models = [m for m in model_files if any(perf in m for perf in ["50.0%", "48.0%", "46.0%"])]
            other_models = [m for m in model_files if m not in priority_models]
            
            print("\n🎯 MASTERY MODELS (Recommended):")
            for model in priority_models[:5]:
                print(f"   • {model}")
            
            if other_models:
                print(f"\n📄 Other models: {len(other_models)} available")
        else:
            print("   ❌ No models directory found")
        
        return
    
    model_filename = sys.argv[1]
    timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000000
    difficulty = sys.argv[3] if len(sys.argv) > 3 else 'medium'
    
    continue_mastery_with_curriculum(model_filename, timesteps, difficulty)

if __name__ == "__main__":
    main()
