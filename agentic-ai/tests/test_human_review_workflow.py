import pytest
import asyncio
from unittest.mock import patch, MagicMock
from src.workflows.human_in_loop import HumanReviewWorkflow
from typing import Dict, Any

@pytest.fixture
def review_data():
    return {"task": "Verify deployment", "payload": {"version": "1.2.3", "environment": "prod"}}

@pytest.fixture
def auto_approved_workflow():
    return HumanReviewWorkflow(auto_approve=True)

@pytest.fixture
def manual_review_workflow():
    return HumanReviewWorkflow(auto_approve=False)

@pytest.mark.asyncio
async def test_get_review_auto_approved(auto_approved_workflow, review_data):
    result = auto_approved_workflow.get_review(review_data)
    assert result["status"] == "approved"
    assert "auto-approved" in result["feedback"]
    assert result["reviewer"] == "system_auto"

@pytest.mark.asyncio
async def test_get_review_manual_review(manual_review_workflow, review_data, caplog):
    with patch('time.sleep', return_value=None): # Mock sleep to avoid actual delay
        with caplog.at_level('WARNING'):
            result = manual_review_workflow.get_review(review_data)
            assert result["status"] == "pending_human_review"
            assert "Human review required" in result["feedback"]
            assert result["reviewer"] == "awaiting_reviewer"
            assert "HUMAN REVIEW REQUIRED for task: Verify deployment. Data: {'task': 'Verify deployment', 'payload': {'version': '1.2.3', 'environment': 'prod'}}" in caplog.text

@pytest.mark.asyncio
async def test_get_async_review_auto_approved(auto_approved_workflow, review_data):
    result = await auto_approved_workflow.get_async_review(review_data)
    assert result["status"] == "approved"
    assert "auto-approved" in result["feedback"]
    assert result["reviewer"] == "system_auto"

@pytest.mark.asyncio
async def test_get_async_review_manual_review(manual_review_workflow, review_data, caplog):
    with patch('asyncio.sleep', return_value=None): # Mock async sleep
        with caplog.at_level('WARNING'):
            result = await manual_review_workflow.get_async_review(review_data)
            assert result["status"] == "pending_human_review"
            assert "Human review required" in result["feedback"]
            assert result["reviewer"] == "awaiting_reviewer"
            assert "HUMAN REVIEW REQUIRED for task: Verify deployment. Data: {'task': 'Verify deployment', 'payload': {'version': '1.2.3', 'environment': 'prod'}}" in caplog.text
