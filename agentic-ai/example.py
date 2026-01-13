from src.agents.orchestrator import AgenticOrchestrator

# Initialize orchestrator
orchestrator = AgenticOrchestrator(config={
    "skills_path": "config/skills.yaml",
    "memory_config": {
        "vector_store": "chroma",
        "persist_directory": "./memory"
    }
})

# Register skills
from src.skills.sop_executor import SOPExecutorSkill
sop_skill = SOPExecutorSkill(config={
    "name": "quality_check",
    "sop_reference": "sops/quality_check.yaml"
})
orchestrator.register_skill(sop_skill)

# Execute workflow
task_payload = {
    "task": "Perform quality check on product batch #123",
    "parameters": {
        "batch_id": "123",
        "checks": ["dimensions", "weight", "labeling"]
    }
}
result = orchestrator.execute(
    task=task_payload["task"],
    context=task_payload
)

print(f"Result: {result}")