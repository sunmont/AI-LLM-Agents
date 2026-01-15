import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.agents.orchestrator import AgenticOrchestrator, AgentState, TaskSpecification, END
from src.skills.base_skill import BaseSkill, SkillConfiguration, SkillType, SkillInput, SkillOutput
from src.memory.memory_manager import MemoryManager
from src.workflows.human_in_loop import HumanReviewWorkflow
from src.mcp.mcp_client import MCPClient
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from typing import Dict, Any, List
import yaml
import os
from src.agents.task_decomposer import TaskDecomposer
from src.skills.skill_router import SkillRouter
from src.skills.skill_executor import SkillExecutor
from src.agents.validator import ResultValidator

# Fixture for a dummy config file for the orchestrator
@pytest.fixture
def orchestrator_config_file(tmp_path):
    config_content = {
        "log_level": "INFO",
        "async_human_review": False, # For simplicity in testing sync flow first
        "skills": [
            {"class": "tests.test_orchestrator.MockDataRetrievalSkill", "config": {"name": "data_retrieval_skill"}},
            {"class": "tests.test_orchestrator.MockAnalysisSkill", "config": {"name": "analysis_skill"}}
        ]
    }
    config_path = tmp_path / "orchestrator_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_content, f)
    return config_path


# Mock skills for registration (these are actual BaseSkill descendants)
class MockDataRetrievalSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any]):
        base_config = SkillConfiguration(name=config["name"], description="Mocks data retrieval", skill_type=SkillType.DATA_RETRIEVAL)
        super().__init__(base_config)

    def _setup_tools(self):
        pass

    def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
        return SkillOutput(success=True, data={"retrieved_data": "mock_data"}, execution_time=0.1, skill_version="1.0.0")

class MockAnalysisSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any]):
        base_config = SkillConfiguration(name=config["name"], description="Mocks data analysis", skill_type=SkillType.ANALYSIS)
        super().__init__(base_config)

    def _setup_tools(self):
        pass

    def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
        return SkillOutput(success=True, data={"analyzed_data": "mock_analysis"}, execution_time=0.2, skill_version="1.0.0")

@pytest.fixture
def orchestrator_instance(orchestrator_config_file):
    """
    Provides an AgenticOrchestrator instance with its internal node methods mocked.
    This allows testing the orchestrator's graph flow in isolation by controlling state transitions directly.
    """
    # Create mock instances for the external components so we can assert on their calls
    mocks = {}
    mocks["memory_manager"] = MagicMock(spec=MemoryManager)
    mocks["memory_manager"].retrieve.return_value = {}
    mocks["memory_manager"].update.return_value = None

    mocks["human_review_workflow"] = MagicMock(spec=HumanReviewWorkflow)
    mocks["human_review_workflow"].get_review.return_value = {"status": "approved", "feedback": "mock_approved"}
    mocks["human_review_workflow"].get_async_review.return_value = asyncio.Future()
    mocks["human_review_workflow"].get_async_review.return_value.set_result({"status": "approved", "feedback": "mock_approved_async"})

    mocks["task_decomposer"] = MagicMock(spec=TaskDecomposer) # Keep for assertion, but its methods won't be called in this setup
    mocks["skill_router"] = MagicMock(spec=SkillRouter) # Keep for assertion
    mocks["skill_executor"] = MagicMock(spec=SkillExecutor) # Keep for assertion
    mocks["result_validator"] = MagicMock(spec=ResultValidator) # Keep for assertion

    # Patch the internal node methods of AgenticOrchestrator directly
    with patch('src.agents.orchestrator.AgenticOrchestrator._supervisor_agent') as mock_supervisor, \
         patch('src.agents.orchestrator.AgenticOrchestrator._planner_agent') as mock_planner, \
         patch('src.agents.orchestrator.AgenticOrchestrator._decomposer_agent') as mock_decomposer_node, \
         patch('src.agents.orchestrator.AgenticOrchestrator._executor_agent') as mock_executor_node, \
         patch('src.agents.orchestrator.AgenticOrchestrator._validator_agent') as mock_validator_node, \
         patch('src.agents.orchestrator.AgenticOrchestrator._compiler_agent') as mock_compiler, \
         patch('src.agents.orchestrator.AgenticOrchestrator._human_review_node') as mock_human_review_node, \
         patch('src.mlops.pipeline.MLOpsPipeline._setup_stages'), \
         patch('src.mlops.pipeline.MLOpsPipeline._setup_mlflow'), \
         patch('src.mlops.pipeline.MLOpsPipeline._setup_logging'), \
         patch('src.mlops.pipeline.MLOpsPipeline._get_git_repo', return_value=MagicMock()):

        # Configure mock return values for the node functions
        # These return values should simulate the state transformations of each node
        # They need to return a dictionary that matches AgentState structure, not just a value.
        mock_supervisor.side_effect = lambda state: {
            **state, 
            "task_spec": {"id": "test_id", "description": state["task"]}, 
            "current_step": "supervisor",
            "messages": state.get("messages", []) + [SystemMessage(content="Supervisor done")]
        }
        mock_planner.side_effect = lambda state: {
            **state, 
            "plan": {"objective": state["task"], "strategy": "test_plan"}, 
            "current_step": "planner",
            "messages": state.get("messages", []) + [SystemMessage(content="Planner done")]
        }
        mock_decomposer_node.side_effect = lambda state: {
            **state, 
            "subtasks": [{"id": "subtask_1", "description": "mock subtask", "skill": "data_retrieval_skill"}], 
            "complexity": 0.2, 
            "current_step": "decomposer",
            "messages": state.get("messages", []) + [SystemMessage(content="Decomposer done")]
        }
        mock_executor_node.side_effect = lambda state: {
            **state, 
            "results": [{"subtask": {"id": "subtask_1"}, "status": "success", "output": {"retrieved_data": "mock_data"}, "duration": 0.1}], 
            "current_step": "executor",
            "messages": state.get("messages", []) + [SystemMessage(content="Executor done")]
        }
        mock_validator_node.side_effect = lambda state: {
            **state, 
            "validation": {"status": "passed"}, 
            "current_step": "validator",
            "messages": state.get("messages", []) + [SystemMessage(content="Validator done")]
        }
        mock_compiler.side_effect = lambda state: {
            **state, 
            "output": {
                "task": state.get("task", ""), 
                "results": state.get("results", []), 
                "validation": state.get("validation", {}), 
                "metadata": {"steps_completed": "compiler", "success_rate": state.get("metadata", {}).get("success_rate", 0.0), "subtasks_count": len(state.get("subtasks",[]))}
            }, 
            "current_step": "compiler",
            "messages": state.get("messages", []) + [SystemMessage(content="Compiler done")]
        }
        mock_human_review_node.side_effect = lambda state: {
            **state, 
            "human_feedback": {"status": "approved"}, 
            "current_step": "human_review",
            "messages": state.get("messages", []) + [SystemMessage(content="Human Review done")]
        }

        # Instantiate the orchestrator with mocked node methods
        orchestrator = AgenticOrchestrator(config=yaml.safe_load(open(orchestrator_config_file)))
        orchestrator.git_repo = MagicMock()
        orchestrator.git_repo.head.commit.hexsha = "testsha"
        
        yield orchestrator, mocks

# --- Test cases for AgenticOrchestrator ---

@pytest.mark.asyncio
async def test_orchestrator_basic_flow_success(orchestrator_instance):
    orchestrator, mocks = orchestrator_instance
    task = "Perform a simple data retrieval and analysis."
    context = {"user": "test"}
    thread_id = "test_thread_1"

    # Now, run the actual workflow through the orchestrator
    final_output = orchestrator.execute(task, config={"thread_id": thread_id}, context=context)

    # Assertions for the final output and interactions with mocks
    expected_output_structure = {
        'task': 'Perform a simple data retrieval and analysis.',
        'results': [{
            'subtask': {'id': 'subtask_1'},
            'status': 'success',
            'output': {'retrieved_data': 'mock_data'},
            'duration': 0.1
        }],
        'validation': {
            'status': 'passed',
            'score': 1.0,
            'feedback': 'mock_validation_passed',
            'criteria_met': True
        },
        'human_feedback': {},
        'metadata': {
            'execution_time': Any, # Will be current timestamp, cannot assert exact value
            'steps_completed': 'compiler',
            'subtasks_count': 1, # Based on mock_decomposer_node.side_effect
            'success_rate': 1.0 # Based on mock_executor_node.side_effect
        },
        'formatted': Any # Content of formatted output will depend on internal formatting logic
    }
    # Deep compare, ignoring dynamic fields
    assert final_output["task"] == expected_output_structure["task"]
    assert final_output["results"] == expected_output_structure["results"]
    assert final_output["validation"] == expected_output_structure["validation"]
    assert final_output["human_feedback"] == expected_output_structure["human_feedback"]
    assert final_output["metadata"]["steps_completed"] == expected_output_structure["metadata"]["steps_completed"]
    assert final_output["metadata"]["subtasks_count"] == expected_output_structure["metadata"]["subtasks_count"]
    assert final_output["metadata"]["success_rate"] == expected_output_structure["metadata"]["success_rate"]
    
    mocks["memory_manager"].retrieve.assert_called_once_with(thread_id)
    mocks["memory_manager"].update.assert_called_once()
    mocks["task_decomposer"].decompose.assert_called_once() # This mock is still in the test and should be called
    mocks["skill_router"].create_plan.assert_called_once() # This mock is still in the test and should be called
    mocks["skill_executor"].execute.assert_called_once() # This mock is still in the test and should be called
    mocks["result_validator"].validate.called # This mock is still in the test and should be called
    mocks["human_review_workflow"].get_review.assert_not_called()
    mocks["human_review_workflow"].get_async_review.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_error_handling(orchestrator_instance):
    orchestrator, mocks = orchestrator_instance
    task = "Perform a failing task."
    context = {"user": "test"}
    thread_id = "test_thread_error"

    # Temporarily patch the _decomposer_agent method on the orchestrator instance
    with patch.object(orchestrator, '_decomposer_agent') as mock_decomposer_agent_for_test:
        mock_decomposer_agent_for_test.side_effect = lambda state: {
            **state,
            "error": "Simulated decomposition error",
            "current_step": "decomposer",
            "messages": state.get("messages", []) + [SystemMessage(content="Decomposer failed")]
        }

        final_output = orchestrator.execute(task, config={"thread_id": thread_id}, context=context)

        assert final_output["error"] == "Simulated decomposition error"
        assert final_output["task"] == task
        assert final_output["status"] == "failed" # This comes from the orchestrator.execute except block

        mocks["memory_manager"].retrieve.assert_called_once_with(thread_id)
        mocks["memory_manager"].update.assert_not_called()
        # The mock_decomposer_agent_for_test is called by the graph
        mock_decomposer_agent_for_test.assert_called_once()
        # Planner should be called before decomposer
        mocks["skill_router"].create_plan.assert_called_once() 
        # Other components should not have been called if the pipeline failed early
        mocks["skill_executor"].execute.assert_not_called()
        mocks["result_validator"].validate.assert_not_called()

@pytest.mark.asyncio
async def test_orchestrator_async_execution(orchestrator_instance):
    orchestrator, mocks = orchestrator_instance
    task = "Perform an async task."
    context = {"user": "async_user"}
    thread_id = "test_thread_async"

    # Now, run the actual workflow through the orchestrator
    final_output = await orchestrator.execute_async(task, config={"thread_id": thread_id}, context=context)

    expected_output_structure = {
        'task': 'Perform an async task.',
        'results': [{
            'subtask': {'id': 'subtask_1'},
            'status': 'success',
            'output': {'retrieved_data': 'mock_data'},
            'duration': 0.1
        }],
        'validation': {
            'status': 'passed',
            'score': 1.0,
            'feedback': 'mock_validation_passed', # Corrected from _async
            'criteria_met': True
        },
        'human_feedback': {},
        'metadata': {
            'execution_time': Any, # Cannot assert exact value
            'steps_completed': 'compiler',
            'subtasks_count': 1,
            'success_rate': 1.0
        },
        'formatted': Any # Content of formatted output will depend on internal formatting logic
    }
    # Deep compare, ignoring dynamic fields
    assert final_output["task"] == expected_output_structure["task"]
    assert final_output["results"] == expected_output_structure["results"]
    assert final_output["validation"] == expected_output_structure["validation"]
    assert final_output["human_feedback"] == expected_output_structure["human_feedback"]
    assert final_output["metadata"]["steps_completed"] == expected_output_structure["metadata"]["steps_completed"]
    assert final_output["metadata"]["subtasks_count"] == expected_output_structure["metadata"]["subtasks_count"]
    assert final_output["metadata"]["success_rate"] == expected_output_structure["metadata"]["success_rate"]

    mocks["memory_manager"].retrieve.assert_called_once_with(thread_id)
    mocks["memory_manager"].update.assert_called_once()
    mocks["task_decomposer"].decompose.assert_called_once()
    mocks["skill_router"].create_plan.assert_called_once()
    mocks["skill_executor"].execute.assert_called_once()
    mocks["result_validator"].validate.called
    mocks["human_review_workflow"].get_review.assert_not_called()
    mocks["human_review_workflow"].get_async_review.assert_not_called()

# You would add more detailed integration tests here to verify the flow between nodes
# by carefully controlling the return values of mocked sub-components (decomposer, executor, validator, etc.)
# and asserting on the calls made to them.
# For example, to test routing:
@pytest.mark.asyncio
async def test_orchestrator_conditional_routing_simple(orchestrator_instance):
    orchestrator, mocks = orchestrator_instance
    task = "Simple task, retrieve some basic data."
    context = {"user": "test"}
    thread_id = "test_thread_simple"

    # Now, run the actual workflow through the orchestrator
    final_output = orchestrator.execute(task, config={"thread_id": thread_id}, context=context)

    # Assertions for the final output and interactions with mocks
    expected_output_structure = {
        'task': task,
        'results': [{
            'subtask': {'id': 'subtask_1'},
            'status': 'success',
            'output': {'retrieved_data': 'mock_data'},
            'duration': 0.1
        }],
        'validation': {}, # Should be empty as validator is skipped
        'human_feedback': {},
        'metadata': {
            'execution_time': Any, # Will be current timestamp, cannot assert exact value
            'steps_completed': 'compiler',
            'subtasks_count': 1, # Based on mock_decomposer_node.side_effect
            'success_rate': 1.0 # Based on mock_executor_node.side_effect
        },
        'formatted': Any
    }
    
    assert final_output["task"] == expected_output_structure["task"]
    assert final_output["results"] == expected_output_structure["results"]
    assert final_output["validation"] == expected_output_structure["validation"] # Should be empty
    assert final_output["human_feedback"] == expected_output_structure["human_feedback"]
    assert final_output["metadata"]["steps_completed"] == expected_output_structure["metadata"]["steps_completed"]
    assert final_output["metadata"]["subtasks_count"] == expected_output_structure["metadata"]["subtasks_count"]
    assert final_output["metadata"]["success_rate"] == expected_output_structure["metadata"]["success_rate"]
    
    mocks["memory_manager"].retrieve.assert_called_once_with(thread_id)
    mocks["memory_manager"].update.assert_called_once()
    mocks["task_decomposer"].decompose.assert_called_once()
    mocks["skill_router"].create_plan.assert_called_once()
    mocks["skill_executor"].execute.assert_called_once()
    mocks["result_validator"].validate.assert_not_called() # Validator should not be called
    mocks["human_review_workflow"].get_review.assert_not_called()
    mocks["human_review_workflow"].get_async_review.assert_not_called()
