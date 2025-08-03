#!/usr/bin/env python3
"""
Test the enhanced curriculum learning system
Focus: Breaking through the 40% survival plateau with skill discovery
"""

from train import train_curriculum_3d_matplotlib_ppo

def test_curriculum_breakthrough():
    """Test curriculum learning with beginner mode for skill discovery."""
    print("🎓 CURRICULUM LEARNING - SKILL BREAKTHROUGH TEST")
    print("=" * 60)
    print("🎯 GOAL: Learn advanced skills to break 40% survival plateau")
    print("🔬 STRATEGY: Start in beginner mode with high error tolerance")
    print("📈 PROGRESSION: Auto-advance through difficulties")
    print("🏆 TARGET: Apply learned skills to achieve 50%+ survival")
    print("=" * 60)
    
    # Use substantial timesteps for real skill development
    total_timesteps = 3000000  # 3M for comprehensive curriculum learning
    
    print(f"📊 Configuration:")
    print(f"   🎓 Starting difficulty: BEGINNER")
    print(f"   🌸 Flowers: 8 (more practice opportunities)")
    print(f"   ⚡ Energy: 180 (high error tolerance)")
    print(f"   🔄 Regeneration: Fast (cooldown practice)")
    print(f"   📈 Auto-progression: ENABLED")
    print(f"   🎯 Total timesteps: {total_timesteps:,}")
    print(f"   ⚖️ Incentives: BALANCED (+5 discovery, -2 inefficiency)")
    print()
    
    print("🔬 SKILL DEVELOPMENT FOCUS:")
    print("   1. Efficient pathfinding between multiple flowers")
    print("   2. Cooldown timing and management strategies")  
    print("   3. Strategic retreat and energy conservation")
    print("   4. Risk assessment and decision making")
    print()
    
    print("🚀 Starting curriculum training...")
    print("👀 Watch for difficulty progression messages!")
    
    try:
        train_curriculum_3d_matplotlib_ppo(
            difficulty='beginner',
            auto_progress=True,
            timesteps=total_timesteps
        )
        print("✅ Curriculum training completed successfully!")
        print("🎓 Check final difficulty level and survival rates")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_curriculum_breakthrough()
