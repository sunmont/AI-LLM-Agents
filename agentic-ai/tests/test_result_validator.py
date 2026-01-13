import pytest
from unittest.mock import MagicMock, patch
from src.agents.validator import ResultValidator
from typing import Dict, Any, List
import json

@pytest.fixture
def result_validator():
    """
    Provides a ResultValidator instance with its internal LLM chain mocked.
    """
    mock_llm = MagicMock()
    mock_prompt = MagicMock()
    mock_output_parser = MagicMock()

    # Create a real ResultValidator instance
    validator = ResultValidator(llm_model=mock_llm, validation_prompt=mock_prompt, output_parser=mock_output_parser)

    # Patch the invoke method of the entire llm_chain
    validator.llm_chain = MagicMock()

    # Return the validator
    return validator

def test_validate_successful_outcome(result_validator):
    result_validator.llm_chain.invoke.return_value = {
        "status": "passed",
        "score": 1.0,
        "feedback": "All criteria met."
    }

    subtask_result = {"status": "success", "data": {"value": 10}}
    task_spec = {"success_criteria": "The value should be 10."}
    context = {}

    validation_output = result_validator.validate(subtask_result, task_spec, context)

    assert validation_output["status"] == "passed"
    assert validation_output["score"] == 1.0
    assert validation_output["feedback"] == "All criteria met."
    assert validation_output["criteria_met"] is True
    result_validator.llm_chain.invoke.assert_called_once()
    assert "result" in result_validator.llm_chain.invoke.call_args[0][0]
    assert "criteria" in result_validator.llm_chain.invoke.call_args[0][0]
    assert "context" in result_validator.llm_chain.invoke.call_args[0][0]
    assert json.loads(result_validator.llm_chain.invoke.call_args[0][0]["result"]) == subtask_result


def test_validate_failed_outcome(result_validator):
    result_validator.llm_chain.invoke.return_value = {
        "status": "failed",
        "score": 0.2,
        "feedback": "Value did not meet expectation."
    }

    subtask_result = {"status": "success", "data": {"value": 5}}
    task_spec = {"success_criteria": "The value should be 10."}
    context = {}

    validation_output = result_validator.validate(subtask_result, task_spec, context)

    assert validation_output["status"] == "failed"
    assert validation_output["score"] == 0.2
    assert validation_output["feedback"] == "Value did not meet expectation."
    assert validation_output["criteria_met"] is False
    result_validator.llm_chain.invoke.assert_called_once()


def test_validate_llm_returns_invalid_format(result_validator):
    # Simulate LLM returning something that's not a dict, which causes ValueError
    result_validator.llm_chain.invoke.return_value = "This is not a valid JSON dict."

    subtask_result = {"status": "success", "data": {"value": 10}}
    task_spec = {"success_criteria": "The value should be 10."}
    context = {}

    validation_output = result_validator.validate(subtask_result, task_spec, context)

    assert validation_output["status"] == "failed"
    assert validation_output["score"] == 0.0
    assert "Validation failed due to internal error" in validation_output["feedback"]
    assert validation_output["criteria_met"] is False
    result_validator.llm_chain.invoke.assert_called_once()


def test_validate_llm_raises_exception(result_validator):
    result_validator.llm_chain.invoke.side_effect = Exception("LLM API error during validation")

    subtask_result = {"status": "success", "data": {"value": 10}}
    task_spec = {"success_criteria": "The value should be 10."}
    context = {}

    validation_output = result_validator.validate(subtask_result, task_spec, context)

    assert validation_output["status"] == "failed"
    assert validation_output["score"] == 0.0
    assert "Validation failed due to internal error: LLM API error during validation" in validation_output["feedback"]
    assert validation_output["criteria_met"] is False
    result_validator.llm_chain.invoke.assert_called_once()

def test_validate_missing_success_criteria(result_validator):
    result_validator.llm_chain.invoke.return_value = {
        "status": "passed",
        "score": 1.0,
        "feedback": "Default criteria assumed met."
    }

    subtask_result = {"status": "success", "data": {"value": 10}}
    task_spec = {} # Missing success_criteria
    context = {}

    validation_output = result_validator.validate(subtask_result, task_spec, context)

    assert validation_output["status"] == "passed"
    assert validation_output["criteria_met"] is True
    # Check that default success criteria was passed to LLM
    assert "The subtask should complete successfully." in result_validator.llm_chain.invoke.call_args[0][0]["criteria"]
