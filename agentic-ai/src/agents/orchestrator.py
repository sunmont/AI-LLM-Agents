"""
LangGraph-based agent orchestration system with task decomposition and memory management
"""
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import Tool
import networkx as nx
import json


class AgenticOrchestrator:
    """Graph-based agent orchestration system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_graph = StateGraph(dict)
        self.checkpointer = MemorySaver()
        self.skills_registry = {}
        self._setup_workflow()

    def _setup_workflow(self):
        """Build the LangGraph workflow with task decomposition"""
        # Define nodes
        self.workflow_graph.add_node("receive_input", self._receive_input)
        self.workflow_graph.add_node("decompose_task", self._decompose_task)
        self.workflow_graph.add_node("assign_subtasks", self._assign_subtasks)
        self.workflow_graph.add_node("execute_skill", self._execute_skill)
        self.workflow_graph.add_node("validate_results", self._validate_results)
        self.workflow_graph.add_node("human_review", self._human_review)
        self.workflow_graph.add_node("compile_output", self._compile_output)

        # Define edges
        self.workflow_graph.add_edge("receive_input", "decompose_task")
        self.workflow_graph.add_conditional_edges(
            "decompose_task",
            self._route_subtasks,
            {"parallel": "assign_subtasks", "single": "execute_skill"}
        )
        self.workflow_graph.add_edge("assign_subtasks", "execute_skill")
        self.workflow_graph.add_edge("execute_skill", "validate_results")
        self.workflow_graph.add_conditional_edges(
            "validate_results",
            self._route_validation,
            {"approved": "compile_output", "needs_review": "human_review"}
        )
        self.workflow_graph.add_edge("human_review", "compile_output")
        self.workflow_graph.add_edge("compile_output", END)

        # Set entry point
        self.workflow_graph.set_entry_point("receive_input")

        # Compile graph
        self.app = self.workflow_graph.compile(checkpointer=self.checkpointer)

    def _receive_input(self, state: Dict) -> Dict:
        """Receive and preprocess input"""
        return {"input": state.get("input"), "context": state.get("context", {})}

    def _decompose_task(self, state: Dict) -> Dict:
        """Decompose complex tasks into subtasks"""
        from .task_decomposer import TaskDecomposer
        decomposer = TaskDecomposer()
        subtasks = decomposer.decompose(state["input"])
        return {"subtasks": subtasks, "requires_parallel": len(subtasks) > 1}

    def _route_subtasks(self, state: Dict) -> str:
        """Route to parallel or single execution"""
        return "parallel" if state["requires_parallel"] else "single"

    def _assign_subtasks(self, state: Dict) -> Dict:
        """Assign subtasks to appropriate skills"""
        from ..skills.skill_router import SkillRouter
        router = SkillRouter(self.skills_registry)
        assignments = router.assign(state["subtasks"])
        return {"assignments": assignments}

    def _execute_skill(self, state: Dict) -> Dict:
        """Execute agent skill with tool invocation"""
        from ..skills.base_skill import SkillExecutor
        executor = SkillExecutor()
        results = executor.execute(state.get("assignments", [state]))
        return {"results": results, "execution_metadata": executor.metadata}

    def _validate_results(self, state: Dict) -> Dict:
        """Validate skill execution results"""
        from .validator import ResultValidator
        validator = ResultValidator()
        validation = validator.validate(state["results"])
        return {"validation": validation, **state}

    def _route_validation(self, state: Dict) -> str:
        """Route based on validation results"""
        return state["validation"]["status"]

    def _human_review(self, state: Dict) -> Dict:
        """Human-in-the-loop review interface"""
        from ..workflows.human_in_loop import HumanReviewWorkflow
        review = HumanReviewWorkflow().get_review(state["results"])
        return {"human_feedback": review, **state}

    def _compile_output(self, state: Dict) -> Dict:
        """Compile final output from results"""
        output = {
            "final_result": state.get("results", {}),
            "metadata": {
                "execution_time": state.get("execution_metadata", {}),
                "validation": state.get("validation", {}),
                "human_feedback": state.get("human_feedback", {})
            }
        }
        return {"output": output}

    def execute(self, input_data: Dict, config: Optional[Dict] = None) -> Dict:
        """Execute the agentic workflow"""
        config = config or {}
        thread_id = config.get("thread_id", "default")

        # Initialize state
        initial_state = {
            "input": input_data,
            "context": config.get("context", {}),
            "memory": self._retrieve_memory(thread_id)
        }

        # Execute graph
        final_state = self.app.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})

        # Save memory
        self._update_memory(thread_id, final_state)

        return final_state["output"]

    def register_skill(self, skill: Any):
        """Register a modular agent skill"""
        self.skills_registry[skill.name] = skill

    def _retrieve_memory(self, thread_id: str) -> Dict:
        """Retrieve conversation and task memory"""
        from ..memory.conversation_memory import MemoryManager
        return MemoryManager().retrieve(thread_id)

    def _update_memory(self, thread_id: str, state: Dict):
        """Update memory with execution results"""
        from ..memory.conversation_memory import MemoryManager
        MemoryManager().update(thread_id, state)