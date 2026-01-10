"""
Modular, reusable Agent Skills for domain knowledge and SOP encoding
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
import json
import yaml


class SkillType(Enum):
    """Types of agent skills"""
    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMMUNICATION = "communication"


class SkillConfiguration(BaseModel):
    """Configuration for agent skills"""
    name: str
    description: str
    version: str = "1.0.0"
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    sop_reference: Optional[str] = None
    domain_context: Optional[str] = None


class BaseSkill(ABC):
    """Base class for all agent skills"""

    def __init__(self, config: SkillConfiguration):
        self.config = config
        self.tools = []
        self._setup_tools()

    @abstractmethod
    def _setup_tools(self):
        """Setup tools required for the skill"""
        pass

    @abstractmethod
    def execute(self, input_data: Dict, context: Optional[Dict] = None) -> Dict:
        """Execute the skill with given input and context"""
        pass

    def validate_input(self, input_data: Dict) -> bool:
        """Validate input against skill schema"""
        required_keys = set(self.config.inputs.keys())
        provided_keys = set(input_data.keys())
        return required_keys.issubset(provided_keys)

    def load_sop(self, sop_path: str) -> Dict:
        """Load Standard Operating Procedure"""
        with open(sop_path, 'r') as f:
            if sop_path.endswith('.yaml') or sop_path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def encode_domain_knowledge(self, knowledge_files: List[str]) -> str:
        """Encode domain knowledge from files"""
        knowledge = []
        for file in knowledge_files:
            with open(file, 'r') as f:
                content = f.read()
                knowledge.append(content)
        return "\n---\n".join(knowledge)


class SOPExecutorSkill(BaseSkill):
    """Skill for executing Standard Operating Procedures"""

    def __init__(self, config: SkillConfiguration):
        super().__init__(config)
        self.sop = self.load_sop(config.sop_reference) if config.sop_reference else {}

    def _setup_tools(self):
        """Setup SOP execution tools"""
        from langchain_community.tools import StructuredTool

        self.tools = [
            StructuredTool.from_function(
                func=self.execute_step,
                name="execute_sop_step",
                description="Execute a step from the SOP"
            ),
            StructuredTool.from_function(
                func=self.validate_compliance,
                name="validate_compliance",
                description="Validate compliance with SOP requirements"
            ),
            StructuredTool.from_function(
                func=self.record_deviation,
                name="record_deviation",
                description="Record any deviation from SOP"
            )
        ]

    def execute(self, input_data: Dict, context: Optional[Dict] = None) -> Dict:
        """Execute multi-step SOP"""
        steps = self.sop.get("steps", [])
        results = []
        compliance_checks = []

        for step in steps:
            step_result = self.execute_step(step, input_data)
            results.append(step_result)

            # Check compliance
            compliance = self.validate_compliance(step, step_result)
            compliance_checks.append(compliance)

            # Handle deviations
            if not compliance["is_compliant"]:
                deviation = self.record_deviation(step, compliance)

        return {
            "results": results,
            "compliance_summary": compliance_checks,
            "sop_version": self.sop.get("version"),
            "execution_timestamp": context.get("timestamp") if context else None
        }

    def execute_step(self, step: Dict, input_data: Dict) -> Dict:
        """Execute a single SOP step"""
        # Implement step execution logic
        return {"step_id": step["id"], "status": "completed", "output": {}}

    def validate_compliance(self, step: Dict, result: Dict) -> Dict:
        """Validate compliance with SOP step"""
        return {"step_id": step["id"], "is_compliant": True, "checks": []}

    def record_deviation(self, step: Dict, compliance: Dict) -> Dict:
        """Record deviation from SOP"""
        return {"step_id": step["id"], "deviation": compliance.get("issues", [])}


class DataRetrievalSkill(BaseSkill):
    """Skill for retrieving and processing data from various sources"""

    def _setup_tools(self):
        """Setup data retrieval tools"""
        from langchain_community.tools import tool

        @tool
        def query_vector_store(query: str, filters: Optional[Dict] = None) -> str:
            """Query vector database for relevant information"""
            from ..memory.vector_store import VectorStoreManager
            vsm = VectorStoreManager()
            return vsm.search(query, filters)

        @tool
        def fetch_external_data(source: str, query_params: Dict) -> str:
            """Fetch data from external sources"""
            # Implement API calls to external sources
            return f"Data from {source}"

        self.tools = [query_vector_store, fetch_external_data]

    def execute(self, input_data: Dict, context: Optional[Dict] = None) -> Dict:
        """Execute data retrieval operations"""
        query = input_data.get("query", "")
        sources = input_data.get("sources", ["vector_store"])

        results = {}
        for source in sources:
            if source == "vector_store":
                results[source] = self.tools[0].invoke({
                    "query": query,
                    "filters": input_data.get("filters", {})
                })
            elif source == "external":
                results[source] = self.tools[1].invoke({
                    "source": input_data.get("external_source"),
                    "query_params": input_data.get("query_params", {})
                })

        return {
            "retrieved_data": results,
            "query": query,
            "sources_queried": sources
        }