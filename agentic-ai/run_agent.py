import asyncio
import os
import sys
import logging
from src.agents.orchestrator import create_orchestrator

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_agent_example():
    logger.info("Starting agent orchestration example...")

    # 1. Create an instance of the orchestrator, loading skills from config.yaml
    orchestrator = create_orchestrator(config_path="config.yaml")

    logger.info(f"Registered skills: {list(orchestrator.skills_registry.keys())}")
    # The PoemSkill tool should now be registered via the config
    logger.info(f"Registered tools: {list(orchestrator.tools_registry.keys())}")

    # 2. Define a task for the agent
    #    The task description should ideally guide the planner/router
    #    to select the appropriate skill.
    task_description = "Write a haiku about intelligent agents that learn."

    # 3. (Optional) Provide context for the task
    #    Context can contain additional parameters for skills or
    #    information for the agent's decision-making.
    task_context = {
        "output_format": "markdown",
        "topic": "intelligent agents that learn",
        "style": "haiku"
    }

    # 4. Execute the task asynchronously
    logger.info(f"Executing task: '{task_description}'")
    result = await orchestrator.execute_async(
        task=task_description,
        context=task_context
    )

    # 5. Print the result
    print("\n--- Task Execution Result ---")
    if result.get("output", {}).get("formatted"):
        print(result["output"]["formatted"])
    else:
        print(result)

    logger.info("Agent orchestration example finished.")

if __name__ == "__main__":
    # Ensure the project root and src directory are in the Python path
    # The project root is needed for poem_skill.py
    # The src directory is needed for src.agents.orchestrator
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir # In this case, current_dir is already the root
    src_path = os.path.join(project_root, 'src')

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    asyncio.run(start_agent_example())