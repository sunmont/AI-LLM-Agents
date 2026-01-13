import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.mlops.pipeline import MLOpsPipeline, PipelineStage, PipelineResult
from src.mcp.mcp_client import MCPClient, create_mcp_client, MCPToolError # Import relevant MCP components
from datetime import datetime, timedelta
import yaml
import os

# Fixture to mock MLflow for all tests in this file
@pytest.fixture(autouse=True)
def mock_mlflow():
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow.log_metrics') as mock_log_metrics, \
         patch('mlflow.log_artifact') as mock_log_artifact, \
         patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.tracking.MlflowClient') as mock_mlflow_client_class:
        
        mock_active_run.return_value = MagicMock()
        mock_active_run.return_value.info.run_id = "test_run_id"
        mock_mlflow_client_class.return_value = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_active_run.return_value
        
        yield

# Fixture for a dummy config file
@pytest.fixture
def dummy_config_file(tmp_path):
    config_content = {
        "logging": {"level": "INFO"},
        "mlflow": {"tracking_uri": "http://test-mlflow", "experiment_name": "test-pipeline"},
        "mcp_tool_invocation": {
            "tool_name": "mock_mcp_tool",
            "arguments": {"param1": "value1"},
            "security_context": {"level": "medium"}
        }
    }
    config_path = tmp_path / "pipeline_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_content, f)
    return config_path

@pytest.fixture
def mock_mcp_client():
    """Provides a mock MCPClient instance."""
    client = MagicMock(spec=MCPClient)
    client.connect = MagicMock(return_value=None) # Mock the async connect method
    
    # Create a Future object to control the async return value of call_tool
    future_success = asyncio.Future()
    future_success.set_result({"status": "success", "output": "mcp_response"})
    client.call_tool.return_value = future_success # Set the default return value

    return client

@pytest.fixture
def mlops_pipeline_with_mcp(dummy_config_file, mock_mcp_client):
    """Provides an MLOpsPipeline instance with a mocked MCPClient and a mocked code_quality stage."""
    # Mock _run_code_quality_checks to always pass
    with patch('src.mlops.pipeline.MLOpsPipeline._run_code_quality_checks') as mock_code_quality:
        mock_code_quality.return_value = {
            "quality_score": 100.0,
            "total_issues": 0,
            "checks": {},
            "report_path": "mock_code_quality_report.json"
        }
        pipeline = MLOpsPipeline(config_path=dummy_config_file, mcp_client=mock_mcp_client)
        # Ensure _get_git_repo doesn't cause issues in tests
        pipeline.git_repo = MagicMock()
        pipeline.git_repo.head.commit.hexsha = "testsha"
        return pipeline

@pytest.mark.asyncio
async def test_mcp_tool_invocation_stage_success(mlops_pipeline_with_mcp, mock_mcp_client):
    pipeline = mlops_pipeline_with_mcp
    
    # Manually add the MCP stage to the pipeline's stages dictionary
    # since _setup_stages is called in __init__ and uses the config.
    pipeline.add_stage(PipelineStage(
        name="mcp_tool_invocation",
        description="Invoke a generic MCP tool as configured",
        execute=pipeline._invoke_mcp_tool,
        depends_on=[],
        timeout=600
    ))

    # Execute the pipeline up to the MCP stage
    result = await pipeline.execute_pipeline(trigger_event={"event": "test"})

    assert result["overall_success"] is True
    assert "mcp_tool_invocation" in result["stages"]
    assert result["stages"]["mcp_tool_invocation"]["success"] is True
    mock_mcp_client.call_tool.assert_called_once_with(
        tool_name="mock_mcp_tool",
        arguments={"param1": "value1"},
        caller="MLOpsPipeline",
        security_context={"level": "medium"}
    )
    # Verify connect was called if client was created by pipeline
    # For this test, client is injected, so connect won't be called directly by __init__
    # mock_mcp_client.connect.assert_called_once() # This will fail if client is injected

@pytest.mark.asyncio
async def test_mcp_tool_invocation_stage_failure(mlops_pipeline_with_mcp, mock_mcp_client):
    pipeline = mlops_pipeline_with_mcp
    
    # Configure mock_mcp_client.call_tool to raise an exception
    mock_mcp_client.call_tool.side_effect = MCPToolError("Tool failed during MCP call")

    pipeline.add_stage(PipelineStage(
        name="mcp_tool_invocation",
        description="Invoke a generic MCP tool as configured",
        execute=pipeline._invoke_mcp_tool,
        depends_on=[],
        timeout=600
    ))

    result = await pipeline.execute_pipeline(trigger_event={"event": "test"})

    assert result["overall_success"] is False
    assert "mcp_tool_invocation" in result["stages"]
    assert result["stages"]["mcp_tool_invocation"]["success"] is False
    assert "Tool failed during MCP call" in result["stages"]["mcp_tool_invocation"]["error"]
    mock_mcp_client.call_tool.assert_called_once()

@pytest.mark.asyncio
async def test_mcp_tool_invocation_stage_mcp_not_initialized(dummy_config_file):
    # Create pipeline without providing a client or mcp_config
    # This should result in self.mcp_client being None if not explicitly handled
    pipeline = MLOpsPipeline(config_path=dummy_config_file)
    pipeline.git_repo = MagicMock()
    pipeline.git_repo.head.commit.hexsha = "testsha"

    # Set mcp_client to None to simulate it not being initialized
    pipeline.mcp_client = None

    pipeline.add_stage(PipelineStage(
        name="mcp_tool_invocation",
        description="Invoke a generic MCP tool as configured",
        execute=pipeline._invoke_mcp_tool,
        depends_on=[],
        timeout=600
    ))

    result = await pipeline.execute_pipeline(trigger_event={"event": "test"})

    assert result["overall_success"] is False
    assert "mcp_tool_invocation" in result["stages"]
    assert result["stages"]["mcp_tool_invocation"]["success"] is False
    assert "MCPClient is not initialized." in result["stages"]["mcp_tool_invocation"]["error"]
