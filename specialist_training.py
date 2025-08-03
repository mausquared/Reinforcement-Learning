#!/usr/bin/env python3
"""
SPECIALIST TRAINING: Fine-tune a curriculum graduate to master HARD mode

This script takes your fully-educated curriculum model and transforms it
into a HARD environment specialist through precision fine-tuning.
"""

import os
import sys
from train import train_specialist_hard_mode
from stable_baselines3 import PPO
import time

def specialist_hard_training(model_path, timesteps=10000000):
    """
    Transform a curriculum graduate into a HARD mode specialist.
    
    Args:
        model_path: Path to the curriculum-trained model
        timesteps: Additional specialist training steps (10M recommended)
    """
    
    print("🏆 SPECIALIST TRAINING: HARD MODE MASTERY")
    print("=" * 60)
    print("🎓 Graduate Model → 🥇 HARD Mode Specialist")
    print("=" * 60)
    print(f"📦 Base model: {model_path}")
    print(f"🎯 Specialist timesteps: {timesteps:,}")
    print(f"🔴 Environment: HARD (locked)")
    print(f"🔬 Mode: Precision fine-tuning")
    print(f"🎯 Goal: 8% → 50%+ survival rate")
    print("=" * 60)
    
    # Verify model exists
    full_model_path = os.path.join("models", model_path)
    if not os.path.exists(full_model_path):
        print(f"❌ Model not found: {full_model_path}")
        return
    
    # Test loading the model
    try:
        print("🔍 Loading curriculum graduate...")
        test_model = PPO.load(full_model_path)
        print("✅ Curriculum model loaded successfully!")
        print(f"📊 Policy: {test_model.policy}")
        del test_model  # Free memory
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Specialist training configuration
    print(f"\n🔬 SPECIALIST FINE-TUNING PARAMETERS:")
    print(f"   🎯 Learning Rate: 5e-6 (ultra-low for precision)")
    print(f"   🔍 Entropy: 0.001 (minimal exploration)")
    print(f"   🎛️ Environment: HARD only (no progression)")
    print(f"   📊 Batch Size: 128 (increased stability)")
    print(f"   ⏱️ Estimated Time: ~20-30 hours")
    
    # Model naming
    base_name = model_path.replace('.zip', '')
    specialist_name = f"specialist_hard_{base_name}"
    
    print(f"\n💾 Output Model: {specialist_name}.zip")
    print(f"📄 Training Log: specialist_training_log.txt")
    
    confirm = input(f"\n🚀 Begin specialist training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Specialist training cancelled.")
        return
    
    print(f"\n🎓 Starting specialist transformation...")
    print(f"🔥 This is the final forge - patience required!")
    print(f"📈 Watch for slow, steady improvements in HARD survival")
    
    # Call the specialist training function
    train_specialist_hard_mode(
        base_model_path=full_model_path,
        timesteps=timesteps,
        model_name=specialist_name
    )
    
    print(f"\n🏆 SPECIALIST TRAINING COMPLETE!")
    print(f"🥇 Your HARD mode specialist is ready!")

def main():
    """Main function for command line usage."""
    
    if len(sys.argv) < 2:
        print("🏆 SPECIALIST TRAINING: Transform Curriculum Graduate")
        print("Usage: python specialist_training.py <curriculum_model> [timesteps]")
        print("\nExample: python specialist_training.py enhanced_curriculum_from_peak_performance_3006k_survival_50.0%.zip 10000000")
        print("\n📚 Available curriculum models:")
        
        if os.path.exists("models"):
            model_files = [f for f in os.listdir("models") if f.endswith(".zip")]
            
            # Look for curriculum models
            curriculum_models = [m for m in model_files if any(keyword in m.lower() for keyword in ["curriculum", "enhanced"])]
            recent_models = sorted(model_files, key=lambda x: os.path.getmtime(os.path.join("models", x)), reverse=True)[:5]
            
            if curriculum_models:
                print("\n🎓 CURRICULUM GRADUATES (Recommended):")
                for model in curriculum_models[:5]:
                    print(f"   • {model}")
            
            print(f"\n📄 Recent models (top 5):")
            for model in recent_models:
                print(f"   • {model}")
        else:
            print("   ❌ No models directory found")
        
        return
    
    model_filename = sys.argv[1]
    timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 10000000
    
    specialist_hard_training(model_filename, timesteps)

if __name__ == "__main__":
    main()
