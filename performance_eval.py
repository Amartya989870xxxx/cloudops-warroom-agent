"""
CloudOpsWarRoomEnv Performance Evaluation Script.
Runs a benchmark comparing a Random Agent vs an Expert Heuristic Agent across 5 tasks.
Generates a data table and a comparative performance chart.
"""
import os
import re
import sys
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Task IDs to evaluate
TASKS = ["noisy_alert", "bad_deploy", "cascade_failure", "cost_vs_performance", "fog_of_war"]
EPISODES_PER_TASK = 3  # 3 random + 3 expert per task = 30 total episodes

def run_random_benchmark():
    """Runs the random agent via inference.py and parses scores."""
    print("🚀 Running Random Agent Benchmarks...")
    results = []
    
    for task in TASKS:
        print(f"  Evaluating task: {task} (Random)")
        for ep in range(EPISODES_PER_TASK):
            # Run inference.py in local random mode
            cmd = [sys.executable, "inference.py", "--local", "--random", "--task", task, "--max-steps", "15"]
            try:
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                # Parse [END] score=S status="resolved"
                match = re.search(r"\[END\] score=([\d.]+) status=\"(\w+)\"", output)
                if match:
                    score = float(match.group(1))
                    status = match.group(2)
                    results.append({"task_id": task, "agent": "Random", "score": score, "status": status})
                else:
                    print(f"    Warning: No [END] tag found for {task} Ep {ep+1}")
            except Exception as e:
                print(f"    Error running {task} Ep {ep+1}: {e}")
                
    return results

def run_heuristic_benchmark():
    """Runs the expert heuristic agent and parses scores."""
    print("🧠 Running Expert Heuristic Agent Benchmarks...")
    from cloudops_env.env import CloudOpsWarRoomEnvironment
    from heuristic_agent import HeuristicExpertAgent
    
    results = []
    
    for task_id in TASKS:
        print(f"  Evaluating task: {task_id} (Expert)")
        for ep in range(EPISODES_PER_TASK):
            env = CloudOpsWarRoomEnvironment()
            agent = HeuristicExpertAgent(task_id)
            
            env.reset(task_id=task_id)
            done = False
            step = 0
            
            while not done and step < 15:
                step += 1
                action = agent.get_action()
                result = env.step(action)
                done = result.done
                info = result.info
            
            score = info.get("normalized_score", 0.0)
            status = "resolved" if (info.get("incident_resolved") and info.get("diagnosed_correctly")) else "failed"
            results.append({"task_id": task_id, "agent": "Expert", "score": score, "status": status})
            
    return results

def generate_report(data):
    """Generates the data table and performance chart."""
    df = pd.DataFrame(data)
    
    # 1. Summary table
    summary = df.groupby(["task_id", "agent"])["score"].mean().unstack()
    overall_avg = df.groupby("agent")["score"].mean()
    
    print("\n" + "="*50)
    print("📊 PERFORMANCE EVALUATION SUMMARY")
    print("="*50)
    print(summary)
    print("\nOVERALL AVERAGES:")
    print(overall_avg)
    print("="*50 + "\n")
    
    # 2. Plotting
    tasks = summary.index.tolist()
    random_scores = summary["Random"].tolist()
    expert_scores = summary["Expert"].tolist()
    
    x = np.arange(len(tasks))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, random_scores, width, label='Random Agent', color='red', alpha=0.7)
    rects2 = ax.bar(x + width/2, expert_scores, width, label='Expert Agent (Heuristic)', color='green', alpha=0.7)
    
    ax.set_ylabel('Normalized Score (0.0 - 1.0)')
    ax.set_title('CloudOpsWarRoomEnv Performance Comparison: Random vs Expert')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig("evaluation_comparison.png")
    print(f"✅ Success: Performance chart saved to 'evaluation_comparison.png'")
    
    # 3. Save raw data
    df.to_csv("evaluation_raw_data.csv", index=False)
    print(f"✅ Success: Raw data saved to 'evaluation_raw_data.csv'")

if __name__ == "__main__":
    random_res = run_random_benchmark()
    expert_res = run_heuristic_benchmark()
    
    all_data = random_res + expert_res
    generate_report(all_data)
