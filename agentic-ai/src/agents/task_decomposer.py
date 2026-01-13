"""
Placeholder for TaskDecomposer
"""
import logging
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOpenAI # Using OpenAI as an example, needs to be configurable

logger = logging.getLogger(__name__)

class TaskDecomposer:
    """
    A component that breaks down a complex task into smaller, manageable subtasks
    using Large Language Models (LLMs) and potentially Standard Operating Procedures (SOPs).
    """
    def __init__(self, llm_model: Optional[Any] = None, decomposition_prompt: Optional[ChatPromptTemplate] = None, output_parser: Optional[JsonOutputParser] = None):
        self.llm = llm_model if llm_model else ChatOpenAI(temperature=0) # Default to ChatOpenAI if not provided
        self.decomposition_prompt = decomposition_prompt if decomposition_prompt else ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert task decomposer. Your goal is to break down complex tasks into a list of atomic, actionable subtasks. Each subtask should be clearly defined and independent. Output your answer as a JSON array of objects, where each object has 'id', 'description', and 'expected_skill' fields."),
                ("human", "Decompose the following task: {task}\nContext: {context}")
            ]
        )
        self.output_parser = output_parser if output_parser else JsonOutputParser()
        self.llm_chain = self.decomposition_prompt | self.llm | self.output_parser
        logger.info("Initialized TaskDecomposer with LLM-based decomposition.")

    def decompose(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decomposes a task into subtasks using an LLM or SOPs.
        """
        logger.info(f"TaskDecomposer: Decomposing task: '{task}' (LLM-based).")
        subtasks = []
        
        # TODO: Implement SOP-based decomposition here first
        # If task matches an SOP, load subtasks from SOP
        # else, proceed with LLM decomposition

        # LLM-based decomposition
        try:
            llm_output = self.llm_chain.invoke({"task": task, "context": context})

            if isinstance(llm_output, list):
                for i, subtask_data in enumerate(llm_output):
                    # Ensure subtask_data has the expected keys, provide defaults if missing
                    subtasks.append(
                        {
                            "id": subtask_data.get("id", f"subtask_{i+1}"),
                            "description": subtask_data.get("description", "No description provided"),
                            "status": "pending",
                            "expected_skill": subtask_data.get("expected_skill", "general_purpose_skill"),
                            "input_data": {
                                "main_task": task,
                                "subtask_context": context,
                                "original_llm_output": subtask_data # Keep original LLM output for debugging/refinement
                            }
                        }
                    )
            else:
                logger.warning(f"LLM output was not a list: {llm_output}. Falling back to default subtask.")
                subtasks.append(
                    {
                        "id": "subtask_1_default_llm_fail",
                        "description": f"Perform general task for: '{task}' (LLM decomposition failed)",
                        "status": "pending",
                        "expected_skill": "general_purpose_skill",
                        "input_data": {"main_task": task, "context": context}
                    }
                )

        except Exception as e:
            logger.error(f"Error during LLM-based decomposition: {e}. Falling back to default subtask.")
            subtasks.append(
                {
                    "id": "subtask_1_default_error",
                    "description": f"Perform general task for: '{task}' (Decomposition error)",
                    "status": "pending",
                    "expected_skill": "general_purpose_skill",
                    "input_data": {"main_task": task, "context": context}
                }
            )
            
        return subtasks