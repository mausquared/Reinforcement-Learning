#!/usr/bin/env python3
"""
Quick Test of Evaluation Parsing
Test if we can parse the output from train.py correctly
"""

import subprocess
import os

def test_evaluation():
    """Test a single evaluation to check parsing"""
    python_path = "C:/Users/mdnva/OneDrive/Desktop/Projects/Reinforcement-Learning/.venv/Scripts/python.exe"
    model_path = "./models/peak_performance_4200k_survival_42.0%.zip"
    
    # Set environment variables
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    
    print("Testing evaluation parsing...")
    print(f"Model: {model_path}")
    print("Running 10 episodes...")
    
    try:
        result = subprocess.run([
            python_path, "train.py", "5", model_path, "10"
        ], capture_output=True, text=True, timeout=120, env=env, encoding='utf-8', errors='replace')
        
        if result.returncode != 0:
            print(f"❌ Evaluation failed: {result.stderr}")
            return
        
        print("\n📊 RAW OUTPUT:")
        print("-" * 40)
        print(result.stdout)
        print("-" * 40)
        
        # Parse output
        output_lines = result.stdout.split('\n')
        survival_rate = None
        avg_reward = None
        avg_steps = None
        
        print("\n🔍 PARSING RESULTS:")
        for line in output_lines:
            if "Survival Rate:" in line and "%" in line:
                try:
                    parts = line.split("Survival Rate:")
                    if len(parts) > 1:
                        rate_part = parts[1].split("%")[0].strip()
                        if "±" in rate_part:
                            rate_part = rate_part.split("±")[0].strip()
                        survival_rate = float(rate_part)
                        print(f"✅ Found Survival Rate: {survival_rate}%")
                except Exception as e:
                    print(f"❌ Error parsing survival rate from: {line}")
                    print(f"   Error: {e}")
                    
            elif "Average Reward:" in line:
                try:
                    parts = line.split("Average Reward:")
                    if len(parts) > 1:
                        reward_part = parts[1].split("±")[0].strip()
                        avg_reward = float(reward_part)
                        print(f"✅ Found Average Reward: {avg_reward}")
                except Exception as e:
                    print(f"❌ Error parsing reward from: {line}")
                    print(f"   Error: {e}")
                    
            elif "Average Length:" in line:
                try:
                    parts = line.split("Average Length:")
                    if len(parts) > 1:
                        steps_part = parts[1].split("±")[0].strip()
                        avg_steps = float(steps_part)
                        print(f"✅ Found Average Length: {avg_steps}")
                except Exception as e:
                    print(f"❌ Error parsing steps from: {line}")
                    print(f"   Error: {e}")
        
        print(f"\n📈 FINAL PARSED RESULTS:")
        print(f"   Survival Rate: {survival_rate}%")
        print(f"   Average Reward: {avg_reward}")
        print(f"   Average Steps: {avg_steps}")
        
        if survival_rate is not None:
            print("\n✅ Parsing successful!")
        else:
            print("\n❌ Parsing failed - could not find survival rate")
            
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_evaluation()
