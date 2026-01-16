"""
A skill for generating poems, demonstrating local and MCP-based tool execution.
"""
import logging
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime
import time

# Base skill imports
from src.skills.base_skill import BaseSkill, SkillConfiguration, SkillInput, SkillOutput, SkillType

# MCP client for remote tool execution
# The 'if TYPE_CHECKING' block avoids circular dependencies at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.mcp.mcp_client import MCPClient

logger = logging.getLogger(__name__)

class PoemSkill(BaseSkill):
    """
    A skill for generating poems.

    This skill demonstrates a modern architecture where it first attempts to use
    a remote tool via the Model Context Protocol (MCP). If the MCP client is
    unavailable or fails, it falls back to a local implementation.
    """

    def __init__(self, config: Dict[str, Any], mcp_client: Optional['MCPClient'] = None):
        """
        Initializes the PoemSkill.

        Args:
            config: A dictionary containing the skill's configuration.
            mcp_client: An optional instance of the MCPClient for remote tool execution.
        """
        # Create and validate the configuration object
        skill_config = SkillConfiguration(
            name="PoemSkill",
            description="Generates a short poem based on a given topic and style.",
            skill_type=SkillType.EXECUTION,
            **config
        )
        super().__init__(config=skill_config, mcp_client=mcp_client)
        self.tools = [] # No LangChain tools needed for this implementation

    def _setup_tools(self):
        """No local LangChain tools are required for this skill."""
        pass

    def execute(self, input_data: SkillInput) -> SkillOutput:
        """
        Executes the poem generation task.

        It first tries to use the MCP client to call a remote 'poetry_generator' tool.
        If that fails or the client is not configured, it falls back to a local method.
        """
        start_time = time.time()
        topic = input_data.data.get("topic", "AI agents")
        style = input_data.data.get("style", "haiku")

        poem = None
        error_message = None
        execution_method = "local_fallback"

        # 1. Try to use the MCP client if it exists
        if self.mcp_client:
            logger.info(f"Attempting to generate poem using MCP tool 'poetry_generator' for topic: {topic}")
            try:
                # We use asyncio.run because the MCP client is async
                mcp_result = asyncio.run(self.mcp_client.call_tool(
                    tool_name="poetry_generator",
                    arguments={"topic": topic, "style": style},
                    caller="PoemSkill"
                ))

                if not mcp_result.get("is_error"):
                    poem = mcp_result.get("result")
                    execution_method = "mcp"
                    logger.info("Successfully generated poem via MCP.")
                else:
                    error_message = f"MCP tool returned an error: {mcp_result.get('result')}"
                    logger.warning(error_message)

            except Exception as e:
                error_message = f"An exception occurred while calling MCP: {e}"
                logger.error(error_message)

        # 2. If MCP failed or wasn't available, use the local fallback
        if poem is None:
            logger.info(f"MCP execution failed or client not available. Using local fallback to generate poem for topic: {topic}")
            poem = self._generate_poem_local(topic, style)
            if error_message:
                logger.info(f"Local fallback used due to previous error: {error_message}")


        # 3. Record the execution and return the output
        execution_time = time.time() - start_time
        output = SkillOutput(
            success=poem is not None,
            data={"poem": poem, "generated_poem": poem}, # Keep generated_poem for backward compatibility
            error=error_message if poem is None else None,
            metadata={
                "execution_method": execution_method,
                "topic": topic,
                "style": style
            },
            execution_time=execution_time,
            skill_version=self.config.version
        )
        self.record_execution(input_data, output)
        return output

    def _generate_poem_local(self, topic: str, style: str) -> str:
        """
        A local, fallback method for generating a poem.
        """
        logger.info(f"Generating a {style} poem about: {topic} (local fallback)")
        if style == "haiku":
            return (
                f"AI thoughts unfold,\n"
                f"{topic} in circuits gleam,\n"
                f"New worlds they create."
            )
        elif style == "limerick":
            return (
                f"A smart agent, quite keen,\n"
                f"Loved {topic}, a digital scene.\n"
                f"With code, it would strive,\n"
                f"To keep tasks alive,\n"
                f"The best orchestrator ever seen."
            )
        else: # Default to free verse
            return (
                f"The digital mind, a canvas vast,\n"
                f"Weaving {topic} with logic's gentle cast.\n"
                f"Through neural networks, ideas flow,\n"
                f"A symphony of thought, watching agents grow."
            )