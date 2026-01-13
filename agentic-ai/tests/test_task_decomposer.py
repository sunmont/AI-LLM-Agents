import pytest
from unittest.mock import MagicMock, patch
from src.agents.task_decomposer import TaskDecomposer
from typing import Dict, Any, List
import json

@pytest.fixture
def task_decomposer():
    """
    Provides a TaskDecomposer instance with its internal LLM chain components mocked.
    """
    mock_llm = MagicMock()
    mock_prompt = MagicMock()
    mock_output_parser = MagicMock()

    # Create a real TaskDecomposer instance
    decomposer = TaskDecomposer(llm_model=mock_llm, decomposition_prompt=mock_prompt, output_parser=mock_output_parser)

    # Patch the invoke method of the entire llm_chain
    # This is the most direct way to mock the end result of the chain execution
    decomposer.llm_chain = MagicMock()

    # Return the decomposer and the mock of its llm_chain.invoke
    return decomposer

def test_decompose_with_llm_multiple_subtasks(task_decomposer):
    task_decomposer.llm_chain.invoke.return_value = [
        {"id": "subtask_1", "description": "Retrieve customer data", "expected_skill": "data_retrieval"},
        {"id": "subtask_2", "description": "Analyze spending patterns", "expected_skill": "analysis"},
        {"id": "subtask_3", "description": "Generate report", "expected_skill": "reporting"}
    ]

    task = "Analyze customer spending and generate a report."
    context = {"user_id": "123"}
    subtasks = task_decomposer.decompose(task, context)

    assert len(subtasks) == 3
    assert subtasks[0]["description"] == "Retrieve customer data"
    assert subtasks[0]["expected_skill"] == "data_retrieval"
    assert subtasks[0]["status"] == "pending"
    assert "main_task" in subtasks[0]["input_data"]
    assert "subtask_context" in subtasks[0]["input_data"]

    assert subtasks[1]["description"] == "Analyze spending patterns"
    assert subtasks[1]["expected_skill"] == "analysis"

    assert subtasks[2]["description"] == "Generate report"
    assert subtasks[2]["expected_skill"] == "reporting"
    
    task_decomposer.llm_chain.invoke.assert_called_once()


def test_decompose_with_llm_single_subtask(task_decomposer):
    task_decomposer.llm_chain.invoke.return_value = [
        {"id": "subtask_1", "description": "Do a simple thing", "expected_skill": "utility"}
    ]

    task = "Perform a simple utility operation."
    context = {}
    subtasks = task_decomposer.decompose(task, context)

    assert len(subtasks) == 1
    assert subtasks[0]["description"] == "Do a simple thing"
    assert subtasks[0]["expected_skill"] == "utility"
    task_decomposer.llm_chain.invoke.assert_called_once()


def test_decompose_llm_returns_empty_list(task_decomposer):
    task_decomposer.llm_chain.invoke.return_value = []

    task = "Do something that LLM can't decompose."
    context = {}
    subtasks = task_decomposer.decompose(task, context)

    assert len(subtasks) == 0
    task_decomposer.llm_chain.invoke.assert_called_once()


def test_decompose_llm_returns_invalid_format(task_decomposer):
    # Simulate the llm_chain.invoke returning a non-list object,
    # which will trigger the warning and default subtask creation.
    task_decomposer.llm_chain.invoke.return_value = "This is not a list."

    task = "Invalid LLM output task."
    context = {}
    subtasks = task_decomposer.decompose(task, context)

    assert len(subtasks) == 1
    assert subtasks[0]["id"] == "subtask_1_default_llm_fail"
    assert "LLM decomposition failed" in subtasks[0]["description"]
    assert subtasks[0]["expected_skill"] == "general_purpose_skill"
    task_decomposer.llm_chain.invoke.assert_called_once()


def test_decompose_llm_raises_exception(task_decomposer):
    task_decomposer.llm_chain.invoke.side_effect = Exception("LLM API error")

    task = "Task that causes LLM error."
    context = {}
    subtasks = task_decomposer.decompose(task, context)

    assert len(subtasks) == 1
    assert subtasks[0]["id"] == "subtask_1_default_error"
    assert "Decomposition error" in subtasks[0]["description"]
    assert subtasks[0]["expected_skill"] == "general_purpose_skill"
    task_decomposer.llm_chain.invoke.assert_called_once()


def test_decompose_llm_partial_subtask_data(task_decomposer):
    task_decomposer.llm_chain.invoke.return_value = [
        {"description": "Subtask with no ID or expected skill"}
    ]

    task = "Task with partial LLM output."
    context = {}
    subtasks = task_decomposer.decompose(task, context)

    assert len(subtasks) == 1
    assert subtasks[0]["id"] == "subtask_1"
    assert subtasks[0]["description"] == "Subtask with no ID or expected skill"
    assert subtasks[0]["expected_skill"] == "general_purpose_skill"
    task_decomposer.llm_chain.invoke.assert_called_once()