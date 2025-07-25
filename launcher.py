#!/usr/bin/env python3
"""
🐦 3D Hummingbird Environment - Quick Launcher
Simple launcher for the 3D matplotlib hummingbird environment
"""

import os
import sys

# Python executable path for this virtual environment
PYTHON_PATH = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"

def main():
    """Simple launcher with clear options."""
    print("🐦 3D HUMMINGBIRD REINFORCEMENT LEARNING")
    print("=" * 45)
    print("🌍 3D Environment with Matplotlib Visualization")
    print("🔋 Energy Management & Anti-Camping System")
    print("🌸 Multiple Flowers with Cooldowns")
    print("🎯 PPO Training with Advanced Features")
    print("=" * 45)
    
    print("\nChoose an option:")
    print("1. 🎮 Test Environment (Watch 3D hummingbird)")
    print("2. 🎯 Train New Model (500K timesteps)")
    print("3. 🚀 Train New Model (1M timesteps)")
    print("4. 🤖 Test Trained Model")
    print("5. 📊 Evaluate Model Performance")
    print("6. 📈 View Training Progress")
    print("7. ❌ Exit")
    
    choice = input("\nEnter your choice (1-7): ").strip()
    
    if choice == "1":
        print("\n🎮 Testing 3D Environment...")
        print("Watch the hummingbird navigate 3D space!")
        print("Close the matplotlib window when done.")
        os.system(f'"{PYTHON_PATH}" hummingbird_env.py')
        
    elif choice == "2":
        print("\n🎯 Training New Model...")
        print("This will train for 500,000 timesteps (~30-60 minutes)")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            os.system(f'"{PYTHON_PATH}" train.py 1')  # Pass argument "1" for 500K timesteps
        else:
            print("Training cancelled.")
            
    elif choice == "3":
        print("\n🚀 Training New Model (Extended)...")
        print("This will train for 1,000,000 timesteps (~60-120 minutes)")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            os.system(f'"{PYTHON_PATH}" train.py 2')  # Pass argument "2" for 1M timesteps
        else:
            print("Training cancelled.")
            
    elif choice == "4":
        print("\n🤖 Testing Trained Model...")
        print("Looking for trained models...")
        os.system(f'"{PYTHON_PATH}" train.py 3')  # Pass argument "3" for testing best model
            
    elif choice == "5":
        print("\n📊 Evaluating Model Performance...")
        print("Running comprehensive evaluation...")
        # Use the main function approach which handles missing models better
        os.system(f'"{PYTHON_PATH}" -c "from train import main; import sys; sys.argv = [\'train.py\', \'3\']; main()"')
            
    elif choice == "6":
        print("\n📈 Viewing Training Progress...")
        # Use the main function approach for plotting
        os.system(f'"{PYTHON_PATH}" -c "from train import main; import sys; sys.argv = [\'train.py\', \'4\']; main()"')
            
    elif choice == "7":
        print("\n👋 Goodbye!")
        sys.exit(0)
        
    else:
        print("\n❌ Invalid choice. Please enter 1-7.")
    
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
