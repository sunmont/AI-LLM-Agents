"""
Model Context Protocol (MCP) integration for standardized tool access
"""
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool, Resource, Prompt
import json
import httpx
import ssl
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class MCPError(Exception):
    """Base exception for MCP errors"""
    pass

class MCPConnectionError(MCPError):
    """Raised when MCP connection fails"""
    pass

class MCPToolError(MCPError):
    """Raised when tool execution fails"""
    pass

class MCPSecurityLevel(Enum):
    """Security levels for MCP operations"""
    LOW = "low"       # Public data, no restrictions
    MEDIUM = "medium" # Internal data, some restrictions
    HIGH = "high"     # Sensitive data, strict restrictions
    CRITICAL = "critical"  # Highly sensitive, requires approval

@dataclass
class MCPToolCall:
    """Represents a tool call through MCP"""
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: datetime
    caller: str
    security_context: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class MCPClient:
    """MCP client for standardized tool access and multi-agent coordination"""

    def __init__(self, server_config: Dict[str, Any]):
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self.tools_registry: Dict[str, Tool] = {}
        self.resources: List[Resource] = []
        self.prompts: List[Prompt] = []
        self.agent_coordinators: Dict[str, Dict[str, Any]] = {}
        self.audit_log: List[MCPToolCall] = []
        self.connection_params: Optional[StdioServerParameters] = None
        self._setup_connection()

    def _setup_connection(self):
        """Setup MCP connection parameters"""
        # Configure SSL if needed
        ssl_context = None
        if self.server_config.get("use_ssl", False):
            ssl_context = ssl.create_default_context()
            if self.server_config.get("verify_ssl", True):
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            else:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

        # Create connection parameters
        if self.server_config.get("transport") == "stdio":
            self.connection_params = StdioServerParameters(
                command=self.server_config["command"],
                args=self.server_config.get("args", []),
                env=self.server_config.get("env", {}),
                cwd=self.server_config.get("cwd")
            )
        elif self.server_config.get("transport") == "http":
            # HTTP transport configuration
            pass
        else:
            raise ValueError(f"Unsupported transport: {self.server_config.get('transport')}")

    async def connect(self):
        """Connect to MCP server"""
        if not self.connection_params:
            raise MCPConnectionError("Connection parameters not configured")

        try:
            async with stdio_client(self.connection_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    self.session = session
                    await self._initialize_session()
                    logger.info("MCP connection established successfully")

                    # Keep connection alive if configured
                    if self.server_config.get("keep_alive", False):
                        await self._keep_alive()

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise MCPConnectionError(f"Connection failed: {e}")

    async def _initialize_session(self):
        """Initialize MCP session and list available tools"""
        if not self.session:
            raise RuntimeError("Session not established")

        try:
            # List available tools
            list_tools_result = await self.session.list_tools()
            for tool in list_tools_result.tools:
                self.tools_registry[tool.name] = tool
                logger.debug(f"Registered MCP tool: {tool.name}")

            # List available resources
            list_resources_result = await self.session.list_resources()
            self.resources = list_resources_result.resources

            # List available prompts
            list_prompts_result = await self.session.list_prompts()
            self.prompts = list_prompts_result.prompts

            # Initialize capabilities based on server
            await self.session.initialize()

            logger.info(f"Session initialized with {len(self.tools_registry)} tools, "
                       f"{len(self.resources)} resources, {len(self.prompts)} prompts")

        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            raise

    async def _keep_alive(self):
        """Keep connection alive with heartbeat"""
        try:
            while True:
                await asyncio.sleep(60)  # Heartbeat every 60 seconds
                if self.session:
                    # Send a ping or list tools to keep connection alive
                    await self.session.list_tools()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any],
                       caller: str = "unknown", security_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a tool through MCP with security and audit"""
        start_time = datetime.now()
        tool_call = MCPToolCall(
            tool_name=tool_name,
            arguments=arguments,
            timestamp=start_time,
            caller=caller,
            security_context=security_context
        )

        try:
            # Check if tool exists
            if tool_name not in self.tools_registry:
                raise MCPToolError(f"Tool {tool_name} not found in registry")

            # Validate security context
            if not self._check_security(tool_name, security_context):
                raise PermissionError(f"Access denied for tool {tool_name}")

            # Validate arguments against tool schema
            self._validate_tool_arguments(tool_name, arguments)

            # Audit logging
            self._audit_log_tool_call(tool_call, "started")

            # Execute tool
            call_tool_result = await self.session.call_tool(tool_name, arguments)

            # Record result
            tool_call.result = {
                "content": call_tool_result.content,
                "is_error": call_tool_result.isError
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            # Complete audit log
            self._audit_log_tool_call(tool_call, "completed", execution_time)

            return {
                "tool_name": tool_name,
                "result": call_tool_result.content,
                "is_error": call_tool_result.isError,
                "execution_time": execution_time,
                "timestamp": start_time.isoformat()
            }

        except Exception as e:
            tool_call.error = str(e)
            self._audit_log_tool_call(tool_call, "failed")
            logger.error(f"Tool {tool_name} execution failed: {e}")
            raise MCPToolError(f"Tool execution failed: {e}")
        finally:
            self.audit_log.append(tool_call)
            # Trim audit log if too large
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-10000:]

    async def read_resource(self, resource_uri: str,
                           mime_type: Optional[str] = None) -> Dict[str, Any]:
        """Read resource through MCP"""
        try:
            read_resource_result = await self.session.read_resource(resource_uri)

            return {
                "uri": resource_uri,
                "content": read_resource_result.contents,
                "mime_type": read_resource_result.mimeType,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to read resource {resource_uri}: {e}")
            raise MCPError(f"Resource read failed: {e}")

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get prompt through MCP"""
        try:
            get_prompt_result = await self.session.get_prompt(prompt_name, arguments or {})

            return {
                "prompt_name": prompt_name,
                "messages": get_prompt_result.messages,
                "description": get_prompt_result.description,
                "arguments": arguments or {}
            }

        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_name}: {e}")
            raise MCPError(f"Prompt retrieval failed: {e}")

    def _check_security(self, tool_name: str, security_context: Optional[Dict[str, Any]]) -> bool:
        """Check security permissions for tool access"""
        if not security_context:
            # Default to medium security if no context provided
            security_context = {"level": MCPSecurityLevel.MEDIUM.value}

        # Get tool security requirements
        tool = self.tools_registry.get(tool_name)
        if not tool:
            return False

        # Extract security level from tool description or metadata
        tool_security = self._extract_tool_security(tool)

        # Check if caller has sufficient clearance
        caller_level = security_context.get("security_level", MCPSecurityLevel.MEDIUM.value)

        # Convert to comparable values
        security_values = {level.value: idx for idx, level in enumerate(MCPSecurityLevel)}

        if security_values.get(caller_level, 1) >= security_values.get(tool_security, 1):
            return True

        # Check for specific permissions
        required_permissions = tool_security.get("permissions", [])
        caller_permissions = security_context.get("permissions", [])

        if all(perm in caller_permissions for perm in required_permissions):
            return True

        return False

    def _extract_tool_security(self, tool: Tool) -> str:
        """Extract security requirements from tool"""
        # Check description for security hints
        description = tool.description or ""

        if any(word in description.lower() for word in ["sensitive", "confidential", "restricted"]):
            return MCPSecurityLevel.HIGH.value
        elif any(word in description.lower() for word in ["internal", "private"]):
            return MCPSecurityLevel.MEDIUM.value
        else:
            return MCPSecurityLevel.LOW.value

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]):
        """Validate tool arguments against schema"""
        tool = self.tools_registry.get(tool_name)
        if not tool or not tool.inputSchema:
            return

        # Basic validation - in production, use JSON Schema validation
        schema = tool.inputSchema

        # Check required fields
        if "required" in schema:
            for field in schema["required"]:
                if field not in arguments:
                    raise ValueError(f"Missing required argument: {field}")

    def _audit_log_tool_call(self, tool_call: MCPToolCall, status: str,
                           execution_time: Optional[float] = None):
        """Log tool call for audit purposes"""
        audit_entry = {
            "tool": tool_call.tool_name,
            "caller": tool_call.caller,
            "status": status,
            "timestamp": tool_call.timestamp.isoformat(),
            "execution_time": execution_time,
            "security_context": tool_call.security_context
        }

        # Mask sensitive data in arguments for logging
        masked_args = self._mask_sensitive_data(tool_call.arguments)
        audit_entry["arguments"] = masked_args

        if tool_call.error:
            audit_entry["error"] = tool_call.error

        # Log to appropriate destination
        if self.server_config.get("audit_log_file"):
            with open(self.server_config["audit_log_file"], "a") as f:
                f.write(json.dumps(audit_entry) + "\n")

        logger.info(f"MCP Audit: {json.dumps(audit_entry)}")

    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data for logging"""
        sensitive_fields = ["password", "token", "secret", "key", "auth"]
        masked_data = data.copy()

        def mask_value(value):
            if isinstance(value, str) and len(value) > 4:
                return value[:2] + "***" + value[-2:]
            return "***"

        for key in list(masked_data.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                masked_data[key] = mask_value(masked_data[key])

        return masked_data

    def register_agent_coordinator(self, coordinator_id: str, coordinator_config: Dict[str, Any]):
        """Register agent coordinator for multi-agent workflows"""
        self.agent_coordinators[coordinator_id] = {
            "config": coordinator_config,
            "agents": [],
            "workflows": [],
            "tools": coordinator_config.get("tools", []),
            "security_policy": coordinator_config.get("security_policy", {})
        }
        logger.info(f"Registered agent coordinator: {coordinator_id}")

    async def coordinate_agents(self, coordinator_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents for complex tasks"""
        if coordinator_id not in self.agent_coordinators:
            raise ValueError(f"Coordinator {coordinator_id} not found")

        coordinator = self.agent_coordinators[coordinator_id]
        start_time = datetime.now()

        logger.info(f"Starting multi-agent coordination: {coordinator_id}")

        # Decompose task
        subtasks = await self._decompose_coordination_task(task, coordinator)

        # Distribute subtasks to agents/tools
        agent_results = []
        for subtask in subtasks:
            try:
                # Select appropriate agent/tool
                assignment = self._assign_to_agent(subtask, coordinator)

                # Execute with security context
                result = await self.call_tool(
                    assignment["tool"],
                    assignment["arguments"],
                    caller=coordinator_id,
                    security_context=coordinator["security_policy"]
                )

                agent_results.append({
                    "subtask": subtask,
                    "assignment": assignment,
                    "result": result
                })

            except Exception as e:
                logger.error(f"Agent coordination failed for subtask: {e}")
                agent_results.append({
                    "subtask": subtask,
                    "error": str(e),
                    "status": "failed"
                })

        # Aggregate and synthesize results
        aggregated = await self._aggregate_agent_results(agent_results, coordinator)

        execution_time = (datetime.now() - start_time).total_seconds()

        coordination_result = {
            "coordinator_id": coordinator_id,
            "task": task,
            "subtasks": subtasks,
            "agent_results": agent_results,
            "aggregated_result": aggregated,
            "execution_time": execution_time,
            "success_rate": len([r for r in agent_results if "error" not in r]) / len(agent_results) if agent_results else 0.0,
            "timestamp": start_time.isoformat()
        }

        logger.info(f"Multi-agent coordination completed: {coordinator_id}")

        return coordination_result

    async def _decompose_coordination_task(self, task: Dict[str, Any],
                                         coordinator: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose coordination task into subtasks"""
        decomposition_strategy = coordinator["config"].get("decomposition_strategy", "sequential")

        if decomposition_strategy == "sequential":
            # Simple linear decomposition
            return [{"id": f"subtask_{i}", "description": f"Step {i}", "data": task}
                   for i in range(coordinator["config"].get("max_subtasks", 3))]

        elif decomposition_strategy == "parallel":
            # Parallel decomposition based on task components
            components = task.get("components", [])
            return [{"id": f"parallel_{i}", "description": f"Component {comp}", "data": comp}
                   for i, comp in enumerate(components)]

        else:
            # Default to single task
            return [{"id": "single", "description": "Complete task", "data": task}]

    def _assign_to_agent(self, subtask: Dict[str, Any], coordinator: Dict[str, Any]) -> Dict[str, Any]:
        """Assign subtask to appropriate agent/tool"""
        # Simple assignment based on subtask type
        subtask_type = subtask.get("type", "general")

        # Find appropriate tool
        available_tools = coordinator.get("tools", []) or list(self.tools_registry.keys())

        # Match tool based on description or type
        for tool_name in available_tools:
            tool = self.tools_registry.get(tool_name)
            if tool and subtask_type in (tool.description or "").lower():
                return {
                    "tool": tool_name,
                    "arguments": subtask.get("data", {}),
                    "assignment_reason": f"Matched type: {subtask_type}"
                }

        # Default to first available tool
        if available_tools:
            return {
                "tool": available_tools[0],
                "arguments": subtask.get("data", {}),
                "assignment_reason": "Default assignment"
            }

        raise ValueError("No tools available for assignment")

    async def _aggregate_agent_results(self, results: List[Dict[str, Any]],
                                     coordinator: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        aggregation_strategy = coordinator["config"].get("aggregation_strategy", "combine")

        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]

        if aggregation_strategy == "combine":
            # Combine all results
            combined = {
                "successful": len(successful_results),
                "failed": len(failed_results),
                "results": [r.get("result", {}) for r in successful_results],
                "errors": [r.get("error") for r in failed_results]
            }
            return combined

        elif aggregation_strategy == "synthesize":
            # Synthesize into single coherent output
            # This could involve another LLM call to combine results
            synthesized = {
                "summary": f"Processed {len(results)} subtasks",
                "key_findings": [],
                "recommendations": [],
                "success_rate": len(successful_results) / len(results) if results else 0.0
            }
            return synthesized

        else:
            # Return raw results
            return {"raw_results": results}

    async def secure_tool_invocation(self, tool_name: str, arguments: Dict[str, Any],
                                   security_context: Dict[str, Any]) -> Dict[str, Any]:
        """Secure tool invocation with enhanced access control"""
        # Enhanced security checks
        security_level = security_context.get("security_level", MCPSecurityLevel.MEDIUM)

        # Check IP whitelist
        if not self._check_ip_whitelist(security_context.get("ip_address")):
            raise PermissionError("IP address not whitelisted")

        # Check rate limiting
        if not self._check_rate_limit(tool_name, security_context.get("user")):
            raise PermissionError("Rate limit exceeded")

        # Require MFA for high security operations
        if security_level in [MCPSecurityLevel.HIGH, MCPSecurityLevel.CRITICAL]:
            if not security_context.get("mfa_verified", False):
                raise PermissionError("MFA required for this operation")

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                self.call_tool(tool_name, arguments, security_context.get("user", "system"), security_context),
                timeout=security_context.get("timeout", 30)
            )
            return result

        except asyncio.TimeoutError:
            logger.error(f"Tool {tool_name} execution timeout")
            raise MCPToolError("Tool execution timeout")

    def _check_ip_whitelist(self, ip_address: Optional[str]) -> bool:
        """Check if IP is in whitelist"""
        whitelist = self.server_config.get("ip_whitelist", [])
        if not whitelist or not ip_address:
            return True  # No whitelist configured

        return ip_address in whitelist

    def _check_rate_limit(self, tool_name: str, user: Optional[str]) -> bool:
        """Check rate limiting for tool/user"""
        # Simplified rate limiting - in production use Redis or similar
        rate_limit = self.server_config.get("rate_limit", {})
        tool_limit = rate_limit.get(tool_name, rate_limit.get("default", 100))

        # Count calls in last minute (simplified)
        recent_calls = [call for call in self.audit_log
                       if call.tool_name == tool_name and
                       (datetime.now() - call.timestamp).total_seconds() < 60]

        return len(recent_calls) < tool_limit

    def get_audit_report(self, start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get audit report for a time period"""
        filtered_logs = self.audit_log

        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]

        return [
            {
                "tool": log.tool_name,
                "caller": log.caller,
                "timestamp": log.timestamp.isoformat(),
                "status": "success" if log.result and not log.result.get("is_error") else "failed",
                "error": log.error,
                "security_context": log.security_context
            }
            for log in filtered_logs
        ]

    def get_server_capabilities(self) -> Dict[str, Any]:
        """Get MCP server capabilities"""
        return {
            "tools": list(self.tools_registry.keys()),
            "resources": [r.uri for r in self.resources],
            "prompts": [p.name for p in self.prompts],
            "agent_coordinators": list(self.agent_coordinators.keys()),
            "connection_status": "connected" if self.session else "disconnected",
            "server_config": {
                "transport": self.server_config.get("transport"),
                "keep_alive": self.server_config.get("keep_alive", False)
            }
        }

# Factory function for creating MCP clients
def create_mcp_client(config_path: Optional[str] = None) -> MCPClient:
    """Create MCP client from configuration"""
    config = {}

    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server"],
            "keep_alive": True,
            "audit_log_file": "mcp_audit.log",
            "rate_limit": {"default": 100},
            "ip_whitelist": ["127.0.0.1", "localhost"]
        }

    return MCPClient(config)

# Async context manager for MCP client
class MCPClientContext:
    """Context manager for MCP client"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[MCPClient] = None

    async def __aenter__(self) -> MCPClient:
        self.client = MCPClient(self.config)
        await self.client.connect()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client and self.client.session:
            # Cleanup if needed
            pass