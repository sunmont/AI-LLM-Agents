"""
Modular, reusable Agent Skills for domain knowledge and SOP encoding
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
import json
import yaml
import asyncio
from datetime import datetime
import logging
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)

class SkillType(Enum):
    """Types of agent skills"""
    DATA_RETRIEVAL = "data_retrieval"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    TRANSFORMATION = "transformation"
    INTEGRATION = "integration"
    AUTOMATION = "automation"

class SkillExecutionMode(Enum):
    """Execution modes for skills"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    CONDITIONAL = "conditional"

class SkillConfiguration(BaseModel):
    """Configuration for agent skills"""
    name: str
    description: str
    version: str = "1.0.0"
    skill_type: SkillType
    execution_mode: SkillExecutionMode = SkillExecutionMode.SEQUENTIAL
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    sop_reference: Optional[str] = None
    domain_context: Optional[str] = None
    timeout: int = 300  # seconds
    retry_count: int = 3
    requires_approval: bool = False
    audit_logging: bool = True

    @validator('name')
    def validate_name(cls, v):
        if not v.isidentifier():
            raise ValueError('Skill name must be a valid identifier')
        return v

    @validator('timeout')
    def validate_timeout(cls, v):
        if v < 1 or v > 3600:
            raise ValueError('Timeout must be between 1 and 3600 seconds')
        return v

class SkillInput(BaseModel):
    """Input model for skills"""
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class SkillOutput(BaseModel):
    """Output model for skills"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    skill_version: str

@dataclass
class SkillResult:
    """Result from skill execution"""
    skill_name: str
    input: SkillInput
    output: SkillOutput
    timestamp: datetime
    execution_id: str

class SkillRegistry:
    """Registry for managing skills"""

    _instance = None
    _skills: Dict[str, 'BaseSkill'] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, skill: 'BaseSkill'):
        """Register a skill"""
        cls._skills[skill.config.name] = skill
        logger.info(f"Registered skill in registry: {skill.config.name}")

    @classmethod
    def get(cls, skill_name: str) -> Optional['BaseSkill']:
        """Get a skill by name"""
        return cls._skills.get(skill_name)

    @classmethod
    def list_skills(cls) -> List[str]:
        """List all registered skills"""
        return list(cls._skills.keys())

    @classmethod
    def get_by_type(cls, skill_type: SkillType) -> List['BaseSkill']:
        """Get skills by type"""
        return [skill for skill in cls._skills.values()
                if skill.config.skill_type == skill_type]

def skill_decorator(timeout: int = 300, retries: int = 3):
    """Decorator for skill execution with timeout and retries"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, input_data: SkillInput, *args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    # Set timeout
                    if asyncio.iscoroutinefunction(func):
                        # Async execution
                        result = asyncio.wait_for(
                            func(self, input_data, *args, **kwargs),
                            timeout=timeout
                        )
                    else:
                        # Sync execution with timeout
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(func, self, input_data, *args, **kwargs)
                            result = future.result(timeout=timeout)

                    logger.info(f"Skill {self.config.name} executed successfully on attempt {attempt + 1}")
                    return result

                except asyncio.TimeoutError:
                    last_error = TimeoutError(f"Skill {self.config.name} timed out after {timeout}s")
                    logger.warning(f"Skill {self.config.name} timed out (attempt {attempt + 1}/{retries})")
                except Exception as e:
                    last_error = e
                    logger.error(f"Skill {self.config.name} failed (attempt {attempt + 1}/{retries}): {e}")

            # All retries failed
            raise last_error
        return wrapper
    return decorator

class BaseSkill(ABC):
    """Base class for all agent skills"""

    def __init__(self, config: SkillConfiguration, mcp_client: Optional['MCPClient'] = None):
        self.config = config
        self.mcp_client = mcp_client
        self.tools: List[Any] = []
        self.cache: Dict[str, Any] = {}
        self.execution_history: List[SkillResult] = []
        self._setup_tools()
        self._validate_config()

        # Register in global registry
        SkillRegistry.register(self)

    def _validate_config(self):
        """Validate skill configuration"""
        if not self.config.name:
            raise ValueError("Skill must have a name")
        if not self.config.description:
            raise ValueError("Skill must have a description")

    @abstractmethod
    def _setup_tools(self):
        """Setup tools required for the skill"""
        pass

    @abstractmethod
    def execute(self, input_data: SkillInput, context: Optional[Dict] = None) -> SkillOutput:
        """Execute the skill with given input and context"""
        pass

    async def execute_async(self, input_data: SkillInput, context: Optional[Dict] = None) -> SkillOutput:
        """Execute the skill asynchronously"""
        # Default implementation runs sync in thread pool
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.execute, input_data, context)
            return future.result()

    @skill_decorator(timeout=300, retries=3)
    def execute_with_retry(self, input_data: SkillInput, context: Optional[Dict] = None) -> SkillOutput:
        """Execute skill with retry logic"""
        return self.execute(input_data, context)

    def validate_input(self, input_data: SkillInput) -> bool:
        """Validate input against skill schema"""
        try:
            # Check required fields based on skill configuration
            if self.config.inputs:
                for field, spec in self.config.inputs.items():
                    if spec.get("required", False) and field not in input_data.data:
                        return False

            # Additional validation can be added here
            return True

        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False

    def load_sop(self, sop_path: str) -> Dict[str, Any]:
        """Load Standard Operating Procedure"""
        try:
            with open(sop_path, 'r') as f:
                if sop_path.endswith('.yaml') or sop_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load SOP from {sop_path}: {e}")
            raise

    def encode_domain_knowledge(self, knowledge_files: List[str]) -> str:
        """Encode domain knowledge from files"""
        knowledge = []
        for file in knowledge_files:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    knowledge.append(f"=== {file} ===\n{content}")
            except Exception as e:
                logger.warning(f"Failed to read knowledge file {file}: {e}")

        return "\n\n---\n\n".join(knowledge)

    def cache_result(self, key: str, value: Any, ttl: int = 3600):
        """Cache a result with time-to-live"""
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "ttl": ttl
        }

    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            cache_entry = self.cache[key]
            age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
            if age < cache_entry["ttl"]:
                return cache_entry["value"]
            else:
                # Remove expired cache
                del self.cache[key]
        return None

    def record_execution(self, input_data: SkillInput, output: SkillOutput):
        """Record skill execution in history"""
        execution_id = hashlib.md5(
            f"{self.config.name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        result = SkillResult(
            skill_name=self.config.name,
            input=input_data,
            output=output,
            timestamp=datetime.now(),
            execution_id=execution_id
        )

        self.execution_history.append(result)

        # Limit history size
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

        # Log if configured
        if self.config.audit_logging:
            logger.info(f"Skill execution recorded: {self.config.name} - {execution_id}")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for this skill"""
        if not self.execution_history:
            return {"total_executions": 0, "success_rate": 0.0}

        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.output.success)

        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "last_execution": self.execution_history[-1].timestamp.isoformat() if self.execution_history else None,
            "average_execution_time": sum(r.output.execution_time for r in self.execution_history) / total if total > 0 else 0.0
        }

    def generate_documentation(self) -> Dict[str, Any]:
        """Generate documentation for this skill"""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "version": self.config.version,
            "type": self.config.skill_type.value,
            "execution_mode": self.config.execution_mode.value,
            "inputs": self.config.inputs,
            "outputs": self.config.outputs,
            "parameters": self.config.parameters,
            "tools": [tool.name for tool in self.tools] if hasattr(self, 'tools') else [],
            "statistics": self.get_execution_stats(),
            "sop_reference": self.config.sop_reference,
            "requires_approval": self.config.requires_approval
        }

    def __str__(self):
        return f"Skill(name={self.config.name}, type={self.config.skill_type.value}, version={self.config.version})"

    def __repr__(self):
        return self.__str__()

class CompositeSkill(BaseSkill):
    """Skill that composes multiple sub-skills"""

    def __init__(self, config: SkillConfiguration, sub_skills: List[BaseSkill], mcp_client: Optional['MCPClient'] = None):
        super().__init__(config, mcp_client=mcp_client)
        self.sub_skills = sub_skills

    def _setup_tools(self):
        """Combine tools from all sub-skills"""
        for skill in self.sub_skills:
            if hasattr(skill, 'tools'):
                self.tools.extend(skill.tools)

    def execute(self, input_data: SkillInput, context: Optional[Dict] = None) -> SkillOutput:
        """Execute all sub-skills based on execution mode"""
        start_time = datetime.now()
        results = []

        if self.config.execution_mode == SkillExecutionMode.SEQUENTIAL:
            for skill in self.sub_skills:
                result = skill.execute(input_data, context)
                results.append(result)
                if not result.success:
                    # Stop on first failure
                    break

        elif self.config.execution_mode == SkillExecutionMode.PARALLEL:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.sub_skills)) as executor:
                futures = {executor.submit(skill.execute, input_data, context): skill
                          for skill in self.sub_skills}
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

        elif self.config.execution_mode == SkillExecutionMode.PIPELINE:
            current_input = input_data
            for skill in self.sub_skills:
                result = skill.execute(current_input, context)
                results.append(result)
                if not result.success:
                    break
                # Pass output as input to next skill
                current_input = SkillInput(data=result.data, context=context)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Aggregate results
        all_successful = all(r.success for r in results)
        aggregated_data = {
            "sub_results": [r.data for r in results],
            "sub_success": [r.success for r in results]
        }

        output = SkillOutput(
            success=all_successful,
            data=aggregated_data,
            error=None if all_successful else "Some sub-skills failed",
            metadata={"sub_skill_count": len(self.sub_skills)},
            execution_time=execution_time,
            skill_version=self.config.version
        )

        self.record_execution(input_data, output)
        return output

# Factory function for creating skills
def create_skill(config: Union[Dict, SkillConfiguration], skill_class: Optional[type] = None) -> BaseSkill:
    """Factory function to create skills"""
    if isinstance(config, dict):
        config = SkillConfiguration(**config)

    if skill_class is None:
        # Determine skill class based on type
        if config.skill_type == SkillType.DATA_RETRIEVAL:
            from .data_retrieval import DataRetrievalSkill
            skill_class = DataRetrievalSkill
        elif config.skill_type == SkillType.EXECUTION:
            from .sop_executor import SOPExecutorSkill
            skill_class = SOPExecutorSkill
        else:
            raise ValueError(f"No default class for skill type: {config.skill_type}")

    return skill_class(config)