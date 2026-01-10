# master-AI-LLM-Agents-Courses

## Setup instructions for Linux

1. **Install Anaconda:**
  - Download the Linux installer from https://www.anaconda.com/download.
  - Open a terminal and navigate to the folder containing the downloaded .sh file.
  - Run the installer: bash Anaconda3*.sh and follow the prompts. Note: This requires about 5+ GB of disk space.
  
2. **Set up the environment:**
  - Open a terminal and navigate to the "project root directory" using: cd ~/Projects/llm_engineering (adjust the path as necessary).
  - Run ls to confirm the presence of subdirectories for each week of the course.
  - Create the environment: conda env create -f environment.yml
     This may take several minutes (even up to an hour for new Anaconda users). If it takes longer or errors occur, proceed to Part 2B.
  - Activate the environment: conda activate llms.
     You should see (llms) in your prompt, indicating successful activation.

# Agentic AI Orchestration Platform

A comprehensive platform for building, deploying, and managing agentic AI systems using LangGraph, LlamaIndex Agents, and Model Context Protocol (MCP).

## Features

### Core Orchestration
- **LangGraph-based workflow orchestration** with state management
- **Task decomposition** and parallel execution
- **Human-in-the-loop workflows** with approval gates
- **Conversation memory** and knowledge graph integration

### Agent Skills
- **Modular, reusable skills** for domain-specific tasks
- **SOP execution** with compliance tracking
- **Multi-step task automation** with validation
- **Tool invocation** with standardized interfaces

### MCP Integration
- **Standardized tool access** through Model Context Protocol
- **Multi-agent coordination** for complex tasks
- **Secure service integration** with audit logging
- **External data source connectivity**

### Model Optimization
- **Fine-tuning** with TRL (SFT, DPO, PPO)
- **GRPO-based reinforcement learning** for task alignment
- **Performance optimization** and evaluation
- **Model comparison** and versioning

### MLOps Infrastructure
- **CI/CD pipelines** for AI services
- **Docker containerization** with GPU support
- **MLflow model registry** and experiment tracking
- **Infrastructure-as-code** deployment (Terraform, Kubernetes)

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- NVIDIA GPU (optional, for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-ai-platform.git
cd agentic-ai-platform
