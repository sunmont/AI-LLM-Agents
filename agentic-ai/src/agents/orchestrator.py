from typing import Dict, List, Any, Optional, TypedDict, Callable
from typing_extensions import Annotated # Added import
import operator # Added import
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.tools import Tool
from pydantic import BaseModel, Field
import networkx as nx
import json
from datetime import datetime
import asyncio
from dataclasses import dataclass
import logging

from ..skills.skill_router import SkillRouter # Moved from _planner_agent
from .task_decomposer import TaskDecomposer # Moved from _decomposer_agent
from ..skills.skill_executor import SkillExecutor # Moved from _executor_agent
from .validator import ResultValidator # Moved from _validator_agent
from ..workflows.human_in_loop import HumanReviewWorkflow # Moved from _human_review_node
from ..memory.memory_manager import MemoryManager # Moved from _retrieve_memory and _update_memory

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State definition for agent workflows"""
    messages: Annotated[List[BaseMessage], operator.add]
    task: str
    subtasks: List[Dict[str, Any]]
    assignments: Dict[str, Any]
    results: List[Dict[str, Any]]
    validation: Dict[str, Any]
    human_feedback: Optional[Dict[str, Any]]
    output: Dict[str, Any]
    context: Dict[str, Any]
    memory: Dict[str, Any]
    current_step: str
    error: Optional[str]

class TaskSpecification(BaseModel):
    """Specification for a task"""
    id: str
    description: str
    priority: int = 1
    dependencies: List[str] = Field(default_factory=list)
    estimated_duration: int = 300  # seconds
    required_skills: List[str] = Field(default_factory=list)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)

class AgenticOrchestrator:
    """Graph-based agent orchestration system"""

    def __init__(self, config: Dict[str, Any], mcp_client: Optional['MCPClient'] = None):
        self.config = config
        self.mcp_client = mcp_client
        self.workflow_graph = StateGraph(AgentState)
        self.checkpointer = InMemorySaver()
        self.skills_registry: Dict[str, Any] = {}
        self.tools_registry: Dict[str, Tool] = {}
        self.agent_nodes: Dict[str, Callable] = {}
        self._setup_logging()
        self._setup_workflow()


    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _setup_workflow(self):
        """Build the LangGraph workflow with task decomposition"""
        logger.info("Setting up agent workflow graph")

        # Define nodes for different agent roles
        self.workflow_graph.add_node("supervisor", self._supervisor_agent)
        self.workflow_graph.add_node("planner", self._planner_agent)
        self.workflow_graph.add_node("decomposer", self._decomposer_agent)
        self.workflow_graph.add_node("executor", self._executor_agent)
        self.workflow_graph.add_node("validator", self._validator_agent)
        self.workflow_graph.add_node("compiler", self._compiler_agent)
        self.workflow_graph.add_node("human_review", self._human_review_node)

        # Define conditional edges for dynamic routing
        self.workflow_graph.set_entry_point("supervisor")

        self.workflow_graph.add_edge("supervisor", "planner")
        self.workflow_graph.add_edge("planner", "decomposer")

        # Conditional routing based on task complexity
        self.workflow_graph.add_conditional_edges(
            "decomposer",
            self._route_based_on_complexity,
            {
                "simple": "executor",
                "complex": "validator",
                "needs_human": "human_review"
            }
        )

        self.workflow_graph.add_edge("executor", "compiler")
        self.workflow_graph.add_edge("validator", "compiler")
        self.workflow_graph.add_edge("human_review", "compiler")
        self.workflow_graph.add_edge("compiler", END)

        # Add fallback edges


        # Compile the graph
        self.app = self.workflow_graph.compile(
            checkpointer=self.checkpointer
            # interrupt_before=["human_review"],
            # interrupt_after=["validator"]
        )

        logger.info("Workflow graph compiled successfully")

    def _supervisor_agent(self, state: AgentState) -> AgentState:
        """Supervisor agent that oversees the entire workflow"""
        logger.info(f"Supervisor processing task: {state.get('task', 'Unknown')}")

        # Initialize workflow state
        task_spec = TaskSpecification(
            id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=state.get("task", ""),
            context=state.get("context", {})
        )

        return {
            **state,
            "task_spec": task_spec.dict(),
            "current_step": "supervisor",
            "messages": state.get("messages", []) + [
                SystemMessage(content="Task initialized by supervisor"),
                HumanMessage(content=f"Task: {state.get('task', '')}")
            ]
        }

    def _planner_agent(self, state: AgentState) -> AgentState:
        """Planning agent that creates execution plan"""
        from ..skills.skill_router import SkillRouter

        logger.info("Planner agent creating execution plan")

        task = state.get("task", "")
        context = state.get("context", {})

        # Create execution plan
        plan = {
            "objective": task,
            "strategy": "sequential_execution",
            "estimated_steps": 3,
            "required_resources": ["memory", "tools"],
            "constraints": context.get("constraints", {})
        }

        # Route to appropriate skills
        router = SkillRouter()
        skill_plan = router.create_plan(task, context)

        return {
            **state,
            "plan": plan,
            "skill_plan": skill_plan,
            "current_step": "planner",
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Execution plan created: {json.dumps(plan, indent=2)}")
            ]
        }

    def _decomposer_agent(self, state: AgentState) -> AgentState:
        """Task decomposition agent"""
        from .task_decomposer import TaskDecomposer

        logger.info("Decomposer agent breaking down task")

        decomposer = TaskDecomposer()
        subtasks = decomposer.decompose(
            state.get("task", ""),
            state.get("context", {}),
            state.get("plan", {})
        )

        # Calculate complexity score
        complexity = self._calculate_complexity(subtasks)

        return {
            **state,
            "subtasks": subtasks,
            "complexity": complexity,
            "current_step": "decomposer",
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Task decomposed into {len(subtasks)} subtasks")
            ]
        }

    def _executor_agent(self, state: AgentState) -> AgentState:
        """Execution agent that runs skills and tools"""
        from ..skills.skill_executor import SkillExecutor

        logger.info("Executor agent running skills")

        executor = SkillExecutor()
        results = []

        for subtask in state.get("subtasks", []):
            try:
                result = executor.execute(subtask, state.get("context", {}))
                results.append(result)
                logger.info(f"Executed subtask: {subtask.get('id', 'Unknown')}")
            except Exception as e:
                logger.error(f"Failed to execute subtask: {e}")
                results.append({
                    "subtask": subtask,
                    "status": "failed",
                    "error": str(e)
                })

        return {
            **state,
            "results": results,
            "current_step": "executor",
            "execution_metadata": executor.get_metadata(),
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Executed {len(results)} subtasks")
            ]
        }

    def _validator_agent(self, state: AgentState) -> AgentState:
        """Validation agent that checks results"""
        from .validator import ResultValidator

        logger.info("Validator agent checking results")

        validator = ResultValidator()
        validation_results = validator.validate(
            state.get("results", []),
            state.get("task_spec", {}),
            state.get("context", {})
        )

        return {
            **state,
            "validation": validation_results,
            "current_step": "validator",
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Validation completed: {validation_results.get('status', 'unknown')}")
            ]
        }

    def _compiler_agent(self, state: AgentState) -> AgentState:
        """Compiler agent that creates final output"""
        logger.info("Compiler agent creating final output")

        # Compile final result
        final_output = {
            "task": state.get("task", ""),
            "results": state.get("results", []),
            "validation": state.get("validation", {}),
            "human_feedback": state.get("human_feedback", {}),
            "metadata": {
                "execution_time": datetime.now().isoformat(),
                "steps_completed": state.get("current_step", ""),
                "subtasks_count": len(state.get("subtasks", [])),
                "success_rate": self._calculate_success_rate(state.get("results", []))
            }
        }

        # Format for different output types
        output_format = state.get("context", {}).get("output_format", "json")
        if output_format == "text":
            final_output["formatted"] = self._format_as_text(final_output)
        elif output_format == "markdown":
            final_output["formatted"] = self._format_as_markdown(final_output)

        return {
            **state,
            "output": final_output,
            "current_step": "compiler",
            "messages": state.get("messages", []) + [
                SystemMessage(content="Final output compiled successfully")
            ]
        }

    def _human_review_node(self, state: AgentState) -> AgentState:
        """Human-in-the-loop review node"""
        from ..workflows.human_in_loop import HumanReviewWorkflow

        logger.info("Requesting human review")

        review_workflow = HumanReviewWorkflow()
        review_data = {
            "task": state.get("task", ""),
            "results": state.get("results", []),
            "validation": state.get("validation", {}),
            "context": state.get("context", {})
        }

        # Get human review (blocking or async based on config)
        if self.config.get("async_human_review", False):
            review = asyncio.run(review_workflow.get_async_review(review_data))
        else:
            review = review_workflow.get_review(review_data)

        return {
            **state,
            "human_feedback": review,
            "current_step": "human_review",
            "messages": state.get("messages", []) + [
                SystemMessage(content=f"Human review received: {review.get('status', 'pending')}")
            ]
        }

    def _route_based_on_complexity(self, state: AgentState) -> str:
        """Route to appropriate node based on task complexity"""
        complexity = state.get("complexity", 0)

        if complexity < 0.3:
            return "simple"
        elif complexity < 0.7:
            return "complex"
        else:
            return "needs_human"

    def _calculate_complexity(self, subtasks: List[Dict]) -> float:
        """Calculate task complexity score (0-1)"""
        if not subtasks:
            return 0.0

        factors = [
            len(subtasks) / 10,  # Number of subtasks
            sum(len(str(task)) > 100 for task in subtasks) / len(subtasks),  # Complexity of descriptions
            any("parallel" in task.get("execution_mode", "") for task in subtasks)  # Parallel execution
        ]

        return min(1.0, sum(factors) / len(factors))

    def _calculate_success_rate(self, results: List[Dict]) -> float:
        """Calculate success rate of execution"""
        if not results:
            return 0.0

        successful = sum(1 for r in results if r.get("status") == "success")
        return successful / len(results)

    def _format_as_text(self, output: Dict) -> str:
        """Format output as text"""
        lines = [
            f"Task: {output.get('task', 'Unknown')}",
            f"Status: {'Completed' if output.get('results') else 'Failed'}",
            f"Subtasks: {len(output.get('results', []))}",
            f"Success Rate: {output.get('metadata', {}).get('success_rate', 0) * 100:.1f}%",
            "\nResults:"
        ]

        for i, result in enumerate(output.get("results", []), 1):
            lines.append(f"  {i}. {result.get('subtask', {}).get('description', 'Unknown')}: {result.get('status', 'unknown')}")

        return "\n".join(lines)

    def _format_as_markdown(self, output: Dict) -> str:
        """Format output as markdown"""
        return f"""# Task Report

## Overview
- **Task**: {output.get('task', 'Unknown')}
- **Status**: {'✅ Completed' if output.get('results') else '❌ Failed'}
- **Execution Time**: {output.get('metadata', {}).get('execution_time', 'Unknown')}
- **Success Rate**: {output.get('metadata', {}).get('success_rate', 0) * 100:.1f}%

## Results
{self._format_results_table(output.get('results', []))}

## Validation
{json.dumps(output.get('validation', {}), indent=2)}

## Human Feedback
{json.dumps(output.get('human_feedback', {}), indent=2)}
"""

    def _format_results_table(self, results: List[Dict]) -> str:
        """Format results as markdown table"""
        if not results:
            return "No results"

        table = "| # | Description | Status | Duration |\n|:-|:-|:-|:-|\n"
        for i, result in enumerate(results, 1):
            table += f"| {i} | {result.get('subtask', {}).get('description', 'N/A')[:50]}... | {result.get('status', 'N/A')} | {result.get('duration', 'N/A')} |\n"

        return table

    def execute(self, task: str, config: Optional[Dict] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a task through the agent workflow"""
        config = config or {}
        context = context or {}
        thread_id = config.get("thread_id", f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        logger.info(f"Executing task '{task[:50]}...' with thread_id: {thread_id}")

        # Initialize state
        initial_state = {
            "messages": [
                SystemMessage(content="You are an intelligent agent orchestrator."),
                HumanMessage(content=task)
            ],
            "task": task,
            "context": context,
            "memory": self._retrieve_memory(thread_id),
            "current_step": "init",
            "error": None
        }

        try:
            # Execute the workflow
            final_state = self.app.invoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        **config.get("graph_config", {})
                    }
                }
            )

            # Update memory
            self._update_memory(thread_id, final_state)

            # Log completion
            logger.info(f"Task completed successfully: {thread_id}")

            return final_state.get("output", {})

        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            return {
                "error": str(e),
                "task": task,
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }

    def register_skill(self, skill: Any):
        """Register a modular agent skill"""
        if hasattr(skill, 'config') and hasattr(skill.config, 'name'):
            self.skills_registry[skill.config.name] = skill
            logger.info(f"Registered skill: {skill.config.name}")

            # Register skill tools
            if hasattr(skill, 'tools'):
                for tool in skill.tools:
                    self.tools_registry[tool.name] = tool

    def register_tool(self, tool: Tool):
        """Register a tool for agent use"""
        self.tools_registry[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def visualize_workflow(self, output_path: str = "workflow.png"):
        """Visualize the workflow graph"""
        try:
            import matplotlib.pyplot as plt

            # Create networkx graph from LangGraph
            G = nx.DiGraph()

            # Add nodes
            for node_name in self.workflow_graph.nodes:
                G.add_node(node_name, label=node_name)

            # Add edges (simplified - real implementation would parse compiled graph)
            edges = [
                ("supervisor", "planner"),
                ("planner", "decomposer"),
                ("decomposer", "executor"),
                ("executor", "compiler"),
                ("decomposer", "validator"),
                ("validator", "compiler"),
                ("decomposer", "human_review"),
                ("human_review", "compiler"),
                ("supervisor", "human_review"),
            ]

            for u, v in edges:
                G.add_edge(u, v)

            # Draw graph
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, seed=42)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)

            # Draw edges
            nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')

            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

            # Draw edge labels for conditions
            edge_labels = {
                ("decomposer", "executor"): "simple",
                ("decomposer", "validator"): "complex",
                ("decomposer", "human_review"): "needs_human",
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

            plt.title("Agent Orchestration Workflow", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Workflow visualization saved to: {output_path}")

        except ImportError:
            logger.warning("Matplotlib not installed. Skipping visualization.")
        except Exception as e:
            logger.error(f"Failed to visualize workflow: {e}")

    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about the workflow"""
        return {
            "nodes_count": len(self.workflow_graph.nodes),
            "skills_registered": len(self.skills_registry),
            "tools_registered": len(self.tools_registry),
            "checkpoints": len(self.checkpointer.list()) if hasattr(self.checkpointer, 'list') else 0,
            "graph_compiled": self.app is not None
        }

    async def execute_async(self, task: str, config: Optional[Dict] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute task asynchronously"""
        # Run in thread pool to avoid blocking
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.execute, task, config, context)
            return future.result()

    def _retrieve_memory(self, thread_id: str) -> Dict:
        """Retrieve conversation and task memory"""
        from ..memory.memory_manager import MemoryManager
        try:
            return MemoryManager().retrieve(thread_id)
        except Exception as e:
            logger.warning(f"Failed to retrieve memory for {thread_id}: {e}")
            return {}

    def _update_memory(self, thread_id: str, state: Dict):
        """Update memory with execution results, ensuring messages are JSON serializable"""
        from ..memory.memory_manager import MemoryManager
        try:
            # Convert BaseMessage objects to dictionary for JSON serialization
            serializable_state = state.copy()
            if "messages" in serializable_state:
                serializable_state["messages"] = [
                    msg.dict() if isinstance(msg, BaseMessage) else msg
                    for msg in serializable_state["messages"]
                ]
            MemoryManager().update(thread_id, serializable_state)
        except Exception as e:
            logger.warning(f"Failed to update memory for {thread_id}: {e}")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mcp.mcp_client import MCPClient

# Factory function for creating orchestrator
def create_orchestrator(config_path: Optional[str] = None, mcp_client: Optional['MCPClient'] = None) -> 'AgenticOrchestrator':
    """Factory function to create orchestrator with configuration"""
    config = {}

    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    orchestrator = AgenticOrchestrator(config, mcp_client=mcp_client)

    # Auto-register skills from config
    if "skills" in config:
        for skill_config in config["skills"]:
            try:
                # Dynamically import and instantiate skill
                module_name, class_name = skill_config["class"].rsplit(".", 1)
                module = __import__(module_name, fromlist=[class_name])
                skill_class = getattr(module, class_name)
                
                # Pass mcp_client to the skill's constructor
                skill_instance = skill_class(
                    config=skill_config.get("config", {}),
                    mcp_client=mcp_client
                )
                orchestrator.register_skill(skill_instance)
            except Exception as e:
                logger.error(f"Failed to register skill {skill_config.get('class')}: {e}")

    return orchestrator