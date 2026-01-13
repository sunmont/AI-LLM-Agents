import pytest
from unittest.mock import MagicMock, patch
from src.skills.skill_executor import SkillExecutor
from src.skills.base_skill import SkillRegistry, BaseSkill, SkillConfiguration, SkillType, SkillInput, SkillOutput
from typing import Dict, Any, List
import time

# Clear the registry before each test to ensure isolation
@pytest.fixture(autouse=True)
def clear_skill_registry():
    SkillRegistry._skills = {}
    yield

# Mock a simple skill for testing
class MockSimpleSkill(BaseSkill):
    def __init__(self, name="simple_skill", description="A simple test skill", should_fail=False):
        config = SkillConfiguration(
            name=name,
            description=description,
            skill_type=SkillType.EXECUTION
        )
        super().__init__(config)
        self.should_fail = should_fail

    def _setup_tools(self):
        pass

    def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
        start_time = time.time()
        if self.should_fail:
            raise ValueError("MockSimpleSkill failed as requested.")
        
        output_data = {"processed_input": input_data.data, "status": "executed"}
        end_time = time.time()
        return SkillOutput(
            success=True,
            data=output_data,
            execution_time=end_time - start_time,
            skill_version=self.config.version
        )

@pytest.fixture
def skill_executor():
    # SkillExecutor now uses the singleton SkillRegistry directly, no need to pass it
    return SkillExecutor()

def test_execute_successful_skill(skill_executor):
    # Register the mock skill
    MockSimpleSkill()

    subtask = {
        "id": "subtask_1",
        "description": "Execute a simple task",
        "skill": "simple_skill",
        "input_data": {"value": 10}
    }
    context = {"user_id": "test_user"}

    result = skill_executor.execute(subtask, context)

    assert result["status"] == "success"
    assert result["output"]["processed_input"] == {"value": 10}
    assert result["output"]["status"] == "executed"
    assert result["subtask"]["id"] == "subtask_1"
    assert "duration" in result
    assert "timestamp" in result

def test_execute_failing_skill(skill_executor):
    # Register a failing mock skill
    MockSimpleSkill(should_fail=True)

    subtask = {
        "id": "subtask_2",
        "description": "Execute a failing task",
        "skill": "simple_skill",
        "input_data": {"value": 20}
    }
    context = {"user_id": "test_user"}

    result = skill_executor.execute(subtask, context)

    assert result["status"] == "failed"
    assert "MockSimpleSkill failed as requested." in result["error"]
    assert result["subtask"]["id"] == "subtask_2"
    assert "duration" in result
    assert "timestamp" in result

def test_execute_unregistered_skill(skill_executor):
    subtask = {
        "id": "subtask_3",
        "description": "Execute an unknown task",
        "skill": "unknown_skill",
        "input_data": {"value": 30}
    }
    context = {"user_id": "test_user"}

    result = skill_executor.execute(subtask, context)

    assert result["status"] == "failed"
    assert "Skill 'unknown_skill' not found." in result["error"]
    assert result["subtask"]["id"] == "subtask_3"
    assert "duration" in result
    assert "timestamp" in result
