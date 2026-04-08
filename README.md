---
title: CloudOps Warroom Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛡️ CloudOps WarRoom Agent - High-Efficiency Autonomous SRE System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Resolved-success)
![Score](https://img.shields.io/badge/Score-0.9037-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)


An optimized autonomous SRE (Site Reliability Engineering) agent designed to resolve production incidents with maximum efficiency within the **CloudOps WarRoom** environment.
Built for high-efficiency incident resolution in reinforcement learning environments.

## 🎯 Overview

The **CloudOps WarRoom Agent** is a specialized autonomous system built to navigate the complexities of modern microservices operations. Designed for the OpenEnv framework, the agent observes system telemetry, reasons through fault propagation, and executes precise recovery actions. While most agents aim for comprehensive investigation, this agent is engineered for **efficiency-first resolution**, prioritizing the restoration of service health to maximize performance rewards.

## 🚀 Key Features

*   **Autonomous Incident Response**: Full lifecycle management from initial observation to final verification.
*   **Dual-Layer Decision Engine**: Combines a high-precision rule-based system with an LLM fallback for adaptive reasoning.
*   **Minimal Step Execution**: Specifically tuned to minimize environment penalties by identifying the shortest path to resolution.
*   **Hugging Face Ready**: Fully compatible with OpenEnv standards for seamless deployment on Hugging Face Spaces.

## 🧠 Core Strategy

In the CloudOps WarRoom environment, every step taken incurs a penalty. Through extensive testing and evaluation, we discovered that **completeness is the enemy of efficiency**. 

Our agent employs a **Minimum Viable Sequence** strategy:
1.  **Immediate Diagnosis**: Direct identification of the root cause service based on unhealthy telemetry.
2.  **Rapid Fix**: Execution of the `restart_service` action to restore service availability.

By bypassing non-essential investigation steps (like deep log queries or trace analysis) when the fault is clear, the agent achieves near-perfect efficiency scores.

## 📊 Performance

The agent consistently achieves top-tier results by focusing on execution speed.

| Metric | Result |
| :--- | :--- |
| **Final Score** | **~0.9037** |
| **Status** | **Resolved** |
| **Efficiency** | **Optimal (2 Steps)** |
| **Environment** | CloudOps WarRoom (v2) |

## 🏗️ Project Structure

```text
.
├── inference.py          # Primary entry point & execution loop
├── heuristic_agent.py    # Core rule-based decision logic
├── openenv.yaml          # OpenEnv manifest & task registry
├── requirements.txt      # Project dependencies
├── Dockerfile            # Container configuration
└── cloudops_env/         # Core environment simulation logic
```

## ⚙️ How It Works

1.  **Observation**: The agent receives a JSON state containing service health, metrics, and active alerts.
2.  **Rule-Based Logic**: Our `get_rule_based_action` system checks the current state against optimized SRE patterns.
3.  **LLM Fallback**: If the heuristics do not find a certain path, the agent utilizes a Large Language Model to reason through the system topology and propose a diagnosis.
4.  **Action Execution**: Actions are sent via REST API to the environment server, and the resulting reward/state is processed for the next step.

## 🏃 How to Run

Ensure you have Python 3.11+ installed.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Agent**:
    ```bash
    python3 inference.py --url https://amartyathedev-cloudops-warroom-env.hf.space
    ```

## 🌐 Hugging Face Space

👉 [CloudOps WarRoom Environment (Hugging Face Repo)](https://huggingface.co/spaces/AmartyaTheDev/cloudOps-warroom-env/tree/main)

This Space hosts the backend environment used by the agent for evaluation and interaction.

## 🛠️ Tech Stack

*   **Logic**: Python 3.11
*   **Interface**: FastAPI / REST
*   **Intelligence**: OpenAI-compatible LLM Client
*   **Infrastructure**: Docker, Hugging Face Spaces

## 👥 Team

*   **Amartya Majumder** — Team Leader & Core Logic
*   **Krishna Bhatia** — Environment Integration
*   **Ayush Arora** — Performance Tuning & Evaluation

---

## 🏁 Conclusion

The CloudOps WarRoom Agent demonstrates that in high-stakes operational environments, the most effective agents are those that can distinguish between "useful information" and "actionable intelligence". By focusing on the **diagnose → fix** path, we provide a robust baseline for autonomous SRE systems.

## 📄 License

This project is licensed under the MIT License.
