#!/usr/bin/env python3
"""
Minimal curriculum training test
"""

try:
    print("🔧 Importing modules...")
    from train import train_curriculum_3d_matplotlib_ppo
    
    print("🔧 Starting curriculum training test...")
    print("📊 Using 1000 timesteps for quick test")
    
    # Call the function directly
    train_curriculum_3d_matplotlib_ppo(
        difficulty='beginner',
        auto_progress=False,
        timesteps=1000
    )
    
    print("✅ Test completed successfully!")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
