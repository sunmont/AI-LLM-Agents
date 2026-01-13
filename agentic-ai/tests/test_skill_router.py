import pytest
from unittest.mock import MagicMock, patch
from src.skills.skill_router import SkillRouter
from src.skills.base_skill import SkillRegistry, BaseSkill, SkillConfiguration, SkillType, SkillInput, SkillOutput, SkillExecutionMode
from typing import Dict, Any, List

# Clear the registry before each test to ensure isolation
@pytest.fixture(autouse=True)
def clear_skill_registry():
    SkillRegistry._skills = {}
    yield

# Define some mock skills for testing
class MockDataRetrievalSkill(BaseSkill):
    def __init__(self, name="data_retrieval_skill", description="Retrieves data from various sources", domain_context=None):
        config = SkillConfiguration(
            name=name,
            description=description,
            skill_type=SkillType.DATA_RETRIEVAL,
            domain_context=domain_context
        )
        super().__init__(config)

    def _setup_tools(self):
        pass

    def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
        return SkillOutput(success=True, data={"result": "data"}, execution_time=0.1, skill_version="1.0.0")

class MockAnalysisSkill(BaseSkill):
    def __init__(self, name="analysis_skill", description="Analyzes provided data", domain_context=None):
        config = SkillConfiguration(
            name=name,
            description=description,
            skill_type=SkillType.ANALYSIS,
            domain_context=domain_context
        )
        super().__init__(config)

    def _setup_tools(self):
        pass

    def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
        return SkillOutput(success=True, data={"result": "analysis"}, execution_time=0.2, skill_version="1.0.0")

class MockExecutionSkill(BaseSkill):
    def __init__(self, name="execute_script", description="Executes a given script", domain_context="code_execution"):
        config = SkillConfiguration(
            name=name,
            description=description,
            skill_type=SkillType.EXECUTION,
            domain_context=domain_context
        )
        super().__init__(config)

    def _setup_tools(self):
        pass

    def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
        return SkillOutput(success=True, data={"result": "script_executed"}, execution_time=0.3, skill_version="1.0.0")


def test_skill_router_finds_best_matching_skill():
    # Register mock skills
    MockDataRetrievalSkill()
    MockAnalysisSkill()

    router = SkillRouter()

    # Test case 1: Task clearly matches data retrieval
    task1 = "Please retrieve some data for me."
    plan1 = router.create_plan(task1, {})
    assert len(plan1) == 1
    assert plan1[0]["skill_name"] == "data_retrieval_skill"

    # Test case 2: Task clearly matches analysis
    task2 = "Analyze the provided information."
    plan2 = router.create_plan(task2, {})
    assert len(plan2) == 1
    assert plan2[0]["skill_name"] == "analysis_skill"

def test_skill_router_no_matching_skill():
    # No skills registered
    router = SkillRouter()
    task = "Do something completely unknown."
    plan = router.create_plan(task, {})
    assert len(plan) == 0

def test_skill_router_with_domain_context():
    # Register mock skills with domain context
    MockExecutionSkill() # domain_context="code_execution"
    MockDataRetrievalSkill(name="generic_data_skill", description="Handles all kinds of data")

    router = SkillRouter()

    task1 = "Execute a Python script for code_execution."
    plan1 = router.create_plan(task1, {})
    assert len(plan1) == 1
    assert plan1[0]["skill_name"] == "execute_script"

    task2 = "Retrieve some data related to finance."
    plan2 = router.create_plan(task2, {})
    assert len(plan2) == 1
    assert plan2[0]["skill_name"] == "generic_data_skill" # Should match this one due to description keyword.


def test_skill_router_prioritizes_higher_score():
    class SpecificDataSkill(BaseSkill):
        def __init__(self, name="specific_data_skill", description="Retrieves highly specific data"):
            config = SkillConfiguration(
                name=name,
                description=description,
                skill_type=SkillType.DATA_RETRIEVAL
            )
            super().__init__(config)
        def _setup_tools(self): pass
        def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
            return SkillOutput(success=True, data={"result": "specific"}, execution_time=0.1, skill_version="1.0.0")

    class GeneralDataSkill(BaseSkill):
        def __init__(self, name="general_data_skill", description="Retrieves general data"):
            config = SkillConfiguration(
                name=name,
                description=description,
                skill_type=SkillType.DATA_RETRIEVAL
            )
            super().__init__(config)
        def _setup_tools(self): pass
        def execute(self, input_data: SkillInput, context: Dict[str, Any] = None) -> SkillOutput:
            return SkillOutput(success=True, data={"result": "general"}, execution_time=0.1, skill_version="1.0.0")

    GeneralDataSkill()
    SpecificDataSkill() # This one should get a higher score for "specific data"

    router = SkillRouter()
    task = "I need to retrieve highly specific data."
    plan = router.create_plan(task, {})
    assert len(plan) == 1
    assert plan[0]["skill_name"] == "specific_data_skill"

