"""
Simple launcher script to train the HummingbirdEnv with different algorithms.
"""

import os
import sys


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    os.system("pip install pygame gymnasium numpy stable-baselines3[extra] matplotlib tensorboard")


def main():
    """Main launcher for training."""
    print("🐦 HummingbirdEnv Training Launcher")
    print("=" * 50)
    
    # Check if dependencies are installed
    try:
        import gymnasium
        import numpy
        import pygame
        print("✅ Basic dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        install_deps = input("Install dependencies? (y/n): ")
        if install_deps.lower() == 'y':
            install_dependencies()
        else:
            print("Please install dependencies manually: pip install -r requirements.txt")
            return
    
    print("\nChoose training algorithm:")
    print("1. 🤖 PPO (Proximal Policy Optimization) - Simple Environment")
    print("2. 🧠 Q-Learning (Classic RL) - Simple Environment")
    print("3. 🎮 Test simple environment only")
    print("4. 📊 Compare PPO vs Q-Learning (Simple)")
    print("5. 🌸 Complex Hummingbird (Energy + Multiple Flowers)")
    print("6. 🔥 Test complex environment")
    
    choice = input("\nEnter choice (1-6): ")
    
    if choice == "1":
        print("\n🤖 Starting PPO Training...")
        try:
            import stable_baselines3
            os.system("python train_ppo.py")
        except ImportError:
            print("❌ Stable-Baselines3 not found. Installing...")
            os.system("pip install stable-baselines3[extra]")
            os.system("python train_ppo.py")
    
    elif choice == "2":
        print("\n🧠 Starting Q-Learning Training...")
        os.system("python train_qlearning.py")
    
    elif choice == "3":
        print("\n🎮 Testing environment...")
        os.system("python main.py")
    
    elif choice == "4":
        print("\n📊 Running both algorithms for comparison...")
        print("This will take a while...")
        
        # Train Q-Learning first (faster)
        print("\n1/2: Training Q-Learning...")
        os.system("echo 5 | python train_qlearning.py")  # Auto-select option 5
        
        # Train PPO
        print("\n2/2: Training PPO...")
        try:
            os.system("echo 4 | python train_ppo.py")  # Auto-select option 4
        except:
            print("PPO training failed. Please install stable-baselines3.")
        
        print("\n✅ Training comparison completed!")
        print("Check the 'models' folder for saved models.")
    
    elif choice == "5":
        print("\n🌸 Starting Complex Hummingbird Environment Training...")
        print("This includes energy management and multiple flowers!")
        os.system("python train_complex_ppo.py")
    
    elif choice == "6":
        print("\n🔥 Testing Complex Environment...")
        os.system("python complex_hummingbird_env.py")
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
