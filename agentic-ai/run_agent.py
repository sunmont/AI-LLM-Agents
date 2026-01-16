import asyncio
import os
import sys
import logging
from src.agents.orchestrator import create_orchestrator
from src.mcp.mcp_client import create_mcp_client, MCPConnectionError

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def start_agent_example():
    logger.info("Starting agent orchestration example...")

    # 1. Create and connect the MCP Client
    mcp_client = None
    try:
        logger.info("Initializing MCP client...")
        # Using default config which points to a dummy server.
        # The skill is designed to fall back to a local implementation if this fails.
        mcp_client_instance = create_mcp_client()
        # The connect call will likely fail because no real MCP server is running.
        await mcp_client_instance.connect()
        mcp_client = mcp_client_instance
        logger.info("MCP client connected successfully.")
    except MCPConnectionError as e:
        logger.warning(f"Could not connect to MCP server: {e}. The PoemSkill will use its local fallback.")
    except Exception as e:
        # Catching other potential errors during client creation/connection
        logger.error(f"An unexpected error occurred with the MCP client: {e}")


    # 2. Create an instance of the orchestrator, passing the mcp_client
    # The orchestrator will pass the client to the skills it creates.
    orchestrator = create_orchestrator(config_path="config.yaml", mcp_client=mcp_client)

    logger.info(f"Registered skills: {list(orchestrator.skills_registry.keys())}")
    logger.info(f"Registered tools: {list(orchestrator.tools_registry.keys())}")

    # 3. Define a task for the agent
    task_description = "Write a haiku about intelligent agents that learn."

    # 4. (Optional) Provide context for the task
    task_context = {
        "output_format": "markdown",
        "topic": "intelligent agents that learn",
        "style": "haiku"
    }

    # 5. Execute the task asynchronously
    logger.info(f"Executing task: '{task_description}'")
    final_state = await orchestrator.execute_async(
        task=task_description,
        context=task_context
    )

    # 6. Print the result
    print("\n--- Task Execution Result ---")
    if final_state.get("output", {}).get("formatted"):
        print(final_state["output"]["formatted"])
    else:
        print(final_state)

    # 7. Check which method the PoemSkill used by inspecting the metadata
    try:
        # The result from the skill execution is in the 'results' list of the output
        skill_results = final_state.get("output", {}).get("results", [])
        if skill_results:
            poem_result_metadata = skill_results[0].get("metadata", {})
            exec_method = poem_result_metadata.get("execution_method", "unknown")
            logger.info(f"PoemSkill execution trace: The skill was executed using the '{exec_method}' method.")
        else:
            logger.warning("Could not find skill execution results in the final output to determine execution method.")
    except (IndexError, KeyError, AttributeError) as e:
        logger.error(f"Error while inspecting final state for execution method: {e}")


    logger.info("Agent orchestration example finished.")

if __name__ == "__main__":
    # Ensure the project root and src directory are in the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    src_path = os.path.join(project_root, 'src')

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    asyncio.run(start_agent_example())
