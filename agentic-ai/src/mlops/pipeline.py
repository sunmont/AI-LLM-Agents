"""
MLOps pipeline with CI/CD, Docker, and model versioning
"""
from typing import Dict, List, Any, Optional, Callable, Union
import yaml
import docker
from docker.models.containers import Container
import mlflow
from mlflow.tracking import MlflowClient
import git
from git import Repo
from datetime import datetime, timedelta
import subprocess
import json
import hashlib
from pathlib import Path
import shutil
import tempfile
import logging
from dataclasses import dataclass, field
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from src.mcp.mcp_client import MCPClient, create_mcp_client # Import MCPClient and create_mcp_client

logger = logging.getLogger(__name__)

@dataclass
class PipelineStage:
    """Represents a stage in the MLOps pipeline"""
    name: str
    description: str
    execute: Callable
    depends_on: List[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retry_count: int = 3
    enabled: bool = True

@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    stage: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class MLOpsPipeline:
    """End-to-end MLOps pipeline for AI services"""

    def __init__(self, config_path: str, mcp_client: Optional[MCPClient] = None, mcp_config: Optional[Dict[str, Any]] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.docker_client = docker.from_env()
        self.mlflow_client = MlflowClient()
        self.git_repo = self._get_git_repo()
        self.stages: Dict[str, PipelineStage] = {}
        self.results: List[PipelineResult] = []
        self.artifacts: Dict[str, Any] = {}

        self._setup_logging()
        self._setup_mlflow()
        self._setup_stages()
        self.mcp_client = mcp_client
        if not self.mcp_client and mcp_config:
            self.mcp_client = create_mcp_client() # Use default client for now, can be configured later
            asyncio.run(self.mcp_client.connect()) # Connect MCP client
            
        logger.info(f"Initialized MLOps pipeline with config from {config_path}")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('logging', {}).get('file', 'pipeline.log')),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow_config = self.config.get('mlflow', {})
        mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'http://localhost:5000'))
        mlflow.set_experiment(mlflow_config.get('experiment_name', 'agentic-ai-pipeline'))

        if not mlflow.active_run():
            mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        logger.info(f"MLflow initialized: {mlflow.active_run().info.run_id}")

    def _get_git_repo(self) -> Optional[Repo]:
        """Get git repository"""
        try:
            return git.Repo(search_parent_directories=True)
        except git.exc.InvalidGitRepositoryError:
            logger.warning("Not in a git repository")
            return None

    def _setup_stages(self):
        """Setup pipeline stages"""
        # Code Quality Stage
        self.add_stage(PipelineStage(
            name="code_quality",
            description="Run code quality and security checks",
            execute=self._run_code_quality_checks,
            timeout=600
        ))

        # Testing Stage
        self.add_stage(PipelineStage(
            name="testing",
            description="Run test suite",
            execute=self._run_tests,
            depends_on=["code_quality"],
            timeout=900
        ))

        # Build Stage
        self.add_stage(PipelineStage(
            name="build",
            description="Build Docker images",
            execute=self._build_docker_images,
            depends_on=["testing"],
            timeout=1200
        ))

        # Model Training Stage
        self.add_stage(PipelineStage(
            name="model_training",
            description="Train and evaluate models",
            execute=self._train_models,
            depends_on=["build"],
            timeout=7200  # 2 hours
        ))

        # Model Evaluation Stage
        self.add_stage(PipelineStage(
            name="model_evaluation",
            description="Evaluate model performance",
            execute=self._evaluate_models,
            depends_on=["model_training"],
            timeout=1800
        ))

        # Security Scan Stage
        self.add_stage(PipelineStage(
            name="security_scan",
            description="Scan for security vulnerabilities",
            execute=self._security_scan,
            depends_on=["build"],
            timeout=600
        ))

        # Deployment Stage
        self.add_stage(PipelineStage(
            name="deployment",
            description="Deploy to target environment",
            execute=self._deploy,
            depends_on=["model_evaluation", "security_scan"],
            timeout=1800
        ))

        # Monitoring Setup Stage
        self.add_stage(PipelineStage(
            name="monitoring",
            description="Setup monitoring and alerts",
            execute=self._setup_monitoring,
            depends_on=["deployment"],
            timeout=600
        ))
        
        # MCP Tool Invocation Stage
        self.add_stage(PipelineStage(
            name="mcp_tool_invocation",
            description="Invoke a generic MCP tool as configured",
            execute=self._invoke_mcp_tool,
            depends_on=[], # Dependencies can be configured in the YAML
            timeout=600
        ))


    def add_stage(self, stage: PipelineStage):
        """Add a stage to the pipeline"""
        self.stages[stage.name] = stage
        logger.info(f"Added pipeline stage: {stage.name}")

    def remove_stage(self, stage_name: str):
        """Remove a stage from the pipeline"""
        if stage_name in self.stages:
            del self.stages[stage_name]
            logger.info(f"Removed pipeline stage: {stage_name}")

    async def execute_pipeline(self, trigger_event: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the complete pipeline asynchronously"""
        trigger_event = trigger_event or {}

        logger.info(f"Starting pipeline execution triggered by: {trigger_event.get('event_type', 'manual')}")

        start_time = datetime.now()
        pipeline_result = {
            "trigger": trigger_event,
            "start_time": start_time.isoformat(),
            "stages": {},
            "overall_success": True,
            "error": None
        }

        try:
            # Execute stages in dependency order
            executed_stages = set()

            while len(executed_stages) < len(self.stages):
                # Find stages that can be executed (dependencies satisfied)
                executable_stages = []
                for stage_name, stage in self.stages.items():
                    if (stage_name not in executed_stages and
                        stage.enabled and
                        all(dep in executed_stages for dep in stage.depends_on)):
                        executable_stages.append(stage)

                if not executable_stages:
                    # Circular dependency or missing stage
                    raise RuntimeError("Cannot resolve stage dependencies")

                # Prepare tasks for parallel execution
                tasks = []
                for stage in executable_stages:
                    tasks.append(asyncio.create_task(self._execute_stage(stage, trigger_event)))
                
                # Run tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result_or_exception in enumerate(results):
                    stage = executable_stages[i]
                    if isinstance(result_or_exception, Exception):
                        logger.error(f"Stage {stage.name} execution failed: {result_or_exception}")
                        pipeline_result["overall_success"] = False
                        pipeline_result["error"] = str(result_or_exception)
                        break
                    else:
                        result = result_or_exception
                        self.results.append(result)

                        pipeline_result["stages"][stage.name] = {
                            "success": result.success,
                            "execution_time": result.execution_time,
                            "error": result.error
                        }

                        if not result.success:
                            pipeline_result["overall_success"] = False
                            pipeline_result["error"] = f"Stage {stage.name} failed"
                            logger.error(f"Pipeline failed at stage: {stage.name}")
                            break

                        executed_stages.add(stage.name)
                        logger.info(f"Stage completed: {stage.name}")

                if not pipeline_result["overall_success"]:
                    break

            # Finalize pipeline
            end_time = datetime.now()
            pipeline_result["end_time"] = end_time.isoformat()
            pipeline_result["duration"] = (end_time - start_time).total_seconds()

            # Log to MLflow
            mlflow.log_metrics({
                "pipeline_duration": pipeline_result["duration"],
                "successful_stages": sum(1 for r in self.results if r.success),
                "failed_stages": sum(1 for r in self.results if not r.success)
            })

            # Generate report
            report_path = self._generate_pipeline_report(pipeline_result)
            pipeline_result["report_path"] = report_path

            if pipeline_result["overall_success"]:
                logger.info(f"Pipeline completed successfully in {pipeline_result['duration']:.2f} seconds")
            else:
                logger.error(f"Pipeline failed: {pipeline_result['error']}")

            return pipeline_result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_result["overall_success"] = False
            pipeline_result["error"] = str(e)
            return pipeline_result

        finally:
            # Cleanup
            self._cleanup_temp_files()

    async def _execute_stage(self, stage: PipelineStage, trigger_event: Dict[str, Any]) -> PipelineResult:
        """Execute a single pipeline stage with retries"""
        start_time = datetime.now()
        last_error = None

        for attempt in range(stage.retry_count):
            try:
                logger.info(f"Executing stage {stage.name} (attempt {attempt + 1}/{stage.retry_count})")

                # Handle both sync and async execute functions
                if asyncio.iscoroutinefunction(stage.execute):
                    output = await stage.execute(trigger_event)
                else:
                    output = stage.execute(trigger_event)

                execution_time = (datetime.now() - start_time).total_seconds()

                result = PipelineResult(
                    stage=stage.name,
                    success=True,
                    output=output,
                    execution_time=execution_time
                )

                logger.info(f"Stage {stage.name} completed in {execution_time:.2f} seconds")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Stage {stage.name} attempt {attempt + 1} failed: {e}")

                if attempt < stage.retry_count - 1:
                    logger.info(f"Retrying stage {stage.name} in 5 seconds...")
                    import time
                    time.sleep(5)

        # All retries failed
        execution_time = (datetime.now() - start_time).total_seconds()

        result = PipelineResult(
            stage=stage.name,
            success=False,
            output=None,
            error=str(last_error),
            execution_time=execution_time
        )

        return result

    def _run_code_quality_checks(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Run code quality and security checks"""
        logger.info("Running code quality checks...")

        checks = [
            ("pylint", ["pylint", "src/", "--rcfile=.pylintrc", "--output-format=json"]),
            ("mypy", ["mypy", "src/", "--ignore-missing-imports", "--no-error-summary"]),
            ("bandit", ["bandit", "-r", "src/", "-f", "json", "-ll"]),
            ("black", ["black", "--check", "--diff", "src/"]),
            ("isort", ["isort", "--check-only", "--diff", "src/"]),
            ("flake8", ["flake8", "src/", "--count", "--exit-zero"]),
        ]

        results = {}
        total_issues = 0

        for check_name, command in checks:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)

                if check_name == "pylint":
                    issues = json.loads(result.stdout) if result.stdout else []
                    issue_count = len(issues)
                elif check_name == "bandit":
                    issues = json.loads(result.stdout) if result.stdout else {}
                    issue_count = len(issues.get("results", []))
                else:
                    issue_count = result.returncode if result.returncode != 0 else 0

                results[check_name] = {
                    "success": result.returncode == 0 or check_name in ["flake8"],  # flake8 exit-zero
                    "issue_count": issue_count,
                    "output": result.stdout[:1000] if result.stdout else "",
                    "error": result.stderr[:1000] if result.stderr else ""
                }

                total_issues += issue_count

            except subprocess.TimeoutExpired:
                results[check_name] = {"success": False, "error": "Timeout", "issue_count": 0}
            except Exception as e:
                results[check_name] = {"success": False, "error": str(e), "issue_count": 0}

        # Calculate quality score (0-100)
        max_issues_per_check = 100
        quality_score = max(0, 100 - (total_issues / (len(checks) * max_issues_per_check)) * 100)

        # Log to MLflow
        mlflow.log_metrics({
            "code_quality_score": quality_score,
            "total_issues": total_issues
        })

        for check_name, result in results.items():
            mlflow.log_metric(f"{check_name}_issues", result["issue_count"])

        # Save detailed report
        report_path = "reports/code_quality_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "quality_score": quality_score,
                "total_issues": total_issues,
                "checks": results
            }, f, indent=2)

        mlflow.log_artifact(report_path)

        # Fail pipeline if quality score below threshold
        quality_threshold = self.config.get('code_quality', {}).get('threshold', 80)
        if quality_score < quality_threshold:
            raise ValueError(f"Code quality score {quality_score:.1f} below threshold {quality_threshold}")

        return {
            "quality_score": quality_score,
            "total_issues": total_issues,
            "checks": results,
            "report_path": report_path
        }

    def _run_tests(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Run test suite"""
        logger.info("Running tests...")

        test_config = self.config.get('testing', {})

        test_cmds = [
            ["pytest", "tests/unit/", "-v",
             f"--cov={test_config.get('coverage_path', 'src')}",
             "--cov-report=xml",
             f"--cov-report=html:{test_config.get('coverage_html_dir', 'reports/coverage')}",
             f"--junitxml={test_config.get('junit_report', 'reports/junit.xml')}",
             f"--numprocesses={test_config.get('parallel_processes', 'auto')}"],

            ["pytest", "tests/integration/", "-v",
             "--timeout=300",
             f"--junitxml={test_config.get('junit_integration', 'reports/junit_integration.xml')}"],

            ["pytest", "tests/e2e/", "-v",
             "--timeout=600",
             f"--junitxml={test_config.get('junit_e2e', 'reports/junit_e2e.xml')}"],
        ]

        test_results = []
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        for cmd in test_cmds:
            test_type = "unit" if "unit" in cmd[2] else "integration" if "integration" in cmd[2] else "e2e"

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

                # Parse test results
                tests_run = 0
                tests_passed = 0
                tests_failed = 0

                # Extract from output
                for line in result.stdout.split('\n'):
                    if 'passed' in line and 'failed' in line and 'skipped' in line:
                        parts = line.split()
                        tests_run = int(parts[0])
                        tests_passed = int(parts[2])
                        tests_failed = int(parts[4])
                        break

                test_results.append({
                    "type": test_type,
                    "success": result.returncode == 0,
                    "tests_run": tests_run,
                    "tests_passed": tests_passed,
                    "tests_failed": tests_failed,
                    "output": result.stdout[:2000],
                    "error": result.stderr[:1000] if result.stderr else ""
                })

                total_tests += tests_run
                passed_tests += tests_passed
                failed_tests += tests_failed

            except subprocess.TimeoutExpired:
                test_results.append({
                    "type": test_type,
                    "success": False,
                    "error": "Timeout after 30 minutes",
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0
                })
            except Exception as e:
                test_results.append({
                    "type": test_type,
                    "success": False,
                    "error": str(e),
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0
                })

        # Calculate coverage if available
        coverage_report = None
        coverage_path = test_config.get('coverage_xml', 'coverage.xml')
        if os.path.exists(coverage_path):
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_path)
            root = tree.getroot()

            line_coverage = float(root.get('line-rate', 0)) * 100
            branch_coverage = float(root.get('branch-rate', 0)) * 100

            coverage_report = {
                "line_coverage": line_coverage,
                "branch_coverage": branch_coverage,
                "total_lines": int(root.get('lines-valid', 0)),
                "covered_lines": int(root.get('lines-covered', 0))
            }

        # Calculate pass rate
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Log to MLflow
        mlflow.log_metrics({
            "test_pass_rate": pass_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests
        })

        if coverage_report:
            mlflow.log_metrics({
                "line_coverage": coverage_report["line_coverage"],
                "branch_coverage": coverage_report["branch_coverage"]
            })

        # Save test report
        report_path = "reports/test_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "pass_rate": pass_rate,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "coverage": coverage_report,
                "test_results": test_results
            }, f, indent=2)

        mlflow.log_artifact(report_path)

        # Fail pipeline if pass rate below threshold
        pass_threshold = test_config.get('pass_threshold', 90)
        if pass_rate < pass_threshold:
            raise ValueError(f"Test pass rate {pass_rate:.1f}% below threshold {pass_threshold}%")

        # Fail if coverage below threshold
        coverage_threshold = test_config.get('coverage_threshold', 80)
        if coverage_report and coverage_report["line_coverage"] < coverage_threshold:
            raise ValueError(f"Code coverage {coverage_report['line_coverage']:.1f}% below threshold {coverage_threshold}%")

        return {
            "pass_rate": pass_rate,
            "total_tests": total_tests,
            "coverage": coverage_report,
            "test_results": test_results,
            "report_path": report_path
        }

    def _build_docker_images(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Build Docker images for AI service"""
        logger.info("Building Docker images...")

        docker_config = self.config.get('docker', {})
        build_results = {}

        # Build main application image
        image_tag = (f"{docker_config.get('registry', 'localhost:5000')}/"
                     f"{docker_config.get('image_name', 'agentic-ai')}:"
                     f"{self._get_git_sha()[:8]}")

        # Create Dockerfile dynamically
        dockerfile_content = self._generate_dockerfile()
        dockerfile_path = "Dockerfile.pipeline"

        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        try:
            # Build image
            logger.info(f"Building image: {image_tag}")

            image, build_logs = self.docker_client.images.build(
                path=".",
                dockerfile=dockerfile_path,
                tag=image_tag,
                buildargs=docker_config.get('build_args', {}),
                rm=True,
                forcerm=True,
                pull=True
            )

            # Run security scan on built image
            scan_result = self._scan_docker_image(image_tag)

            # Test image
            test_result = self._test_docker_image(image_tag)

            # Push to registry if configured
            push_result = None
            if docker_config.get('push_to_registry', False):
                push_result = self._push_docker_image(image_tag)

            build_results["main_image"] = {
                "image_tag": image_tag,
                "build_success": True,
                "security_scan": scan_result,
                "test_result": test_result,
                "push_result": push_result,
                "size_mb": self._get_image_size_mb(image_tag)
            }

            logger.info(f"Docker image built successfully: {image_tag}")

        except docker.errors.BuildError as e:
            logger.error(f"Docker build failed: {e}")
            build_results["main_image"] = {
                "image_tag": image_tag,
                "build_success": False,
                "error": str(e),
                "logs": str(e.build_log) if hasattr(e, 'build_log') else ""
            }
            raise

        # Build additional images if configured
        for additional in docker_config.get('additional_images', []):
            try:
                additional_tag = f"{image_tag}-{additional['name']}"

                additional_image, _ = self.docker_client.images.build(
                    path=additional.get('path', '.'),
                    dockerfile=additional.get('dockerfile', 'Dockerfile'),
                    tag=additional_tag,
                    buildargs=additional.get('build_args', {}),
                    rm=True
                )

                build_results[additional['name']] = {
                    "image_tag": additional_tag,
                    "build_success": True,
                    "size_mb": self._get_image_size_mb(additional_tag)
                }

            except Exception as e:
                logger.warning(f"Failed to build additional image {additional['name']}: {e}")
                build_results[additional['name']] = {
                    "build_success": False,
                    "error": str(e)
                }

        # Log to MLflow
        mlflow.log_metrics({
            "docker_build_success": 1 if build_results["main_image"]["build_success"] else 0,
            "docker_image_size_mb": build_results["main_image"].get("size_mb", 0)
        })

        # Save build report
        report_path = "reports/docker_build_report.json"
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "git_sha": self._get_git_sha(),
                "images": build_results
            }, f, indent=2)

        mlflow.log_artifact(report_path)

        return {
            "images": build_results,
            "main_image_tag": image_tag,
            "report_path": report_path
        }

    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for the AI service"""
        docker_config = self.config.get('docker', {})
        base_image = docker_config.get('base_image', 'python:3.9-slim')

        dockerfile = f"""FROM {base_image}
# Set working directory
WORKDIR /app

# Copy poetry.lock and pyproject.toml files to the container
COPY pyproject.toml poetry.lock* /app/

# Install poetry and dependencies
RUN pip install poetry
RUN poetry install --no-root --no-dev --no-interaction --no-ansi

# Copy the rest of the application code
COPY . /app

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI={docker_config.get('mlflow_tracking_uri', 'http://mlflow:5000')}
ENV MLFLOW_EXPERIMENT_NAME={docker_config.get('mlflow_experiment_name', 'agentic-ai-pipeline')}

# Expose port for FastAPI or other services
EXPOSE {docker_config.get('port', '8000')}

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "{docker_config.get('port', '8000')}"]
"""
        return dockerfile

    def _scan_docker_image(self, image_tag: str) -> Dict[str, Any]:
        """Scan Docker image for vulnerabilities using Trivy"""
        logger.info(f"Scanning Docker image: {image_tag} for vulnerabilities...")
        try:
            # Use Trivy to scan the image
            # Assumes Trivy is installed and accessible in the pipeline environment
            # Example: trivy image --format json --output results.json <image_tag>
            scan_command = ["trivy", "image", "--format", "json", image_tag]
            result = subprocess.run(scan_command, capture_output=True, text=True, timeout=900)
            
            scan_report = json.loads(result.stdout) if result.stdout else {}
            vulnerabilities_found = 0
            if scan_report and scan_report.get("Vulnerabilities"):
                vulnerabilities_found = len(scan_report["Vulnerabilities"])

            # Log to MLflow
            mlflow.log_metric("image_vulnerabilities_count", vulnerabilities_found)

            report_path = f"reports/trivy_scan_{image_tag.replace(':', '_').replace('/', '_')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(scan_report, f, indent=2)
            mlflow.log_artifact(report_path)

            if result.returncode != 0:
                logger.warning(f"Trivy scan found vulnerabilities or exited with non-zero: {result.stderr}")
                return {"success": False, "vulnerabilities_found": vulnerabilities_found, "report_path": report_path, "error": result.stderr[:1000]}

            logger.info(f"Docker image scan complete. Vulnerabilities found: {vulnerabilities_found}")
            return {"success": True, "vulnerabilities_found": vulnerabilities_found, "report_path": report_path}

        except subprocess.TimeoutExpired:
            logger.error(f"Trivy scan timed out for image {image_tag}")
            return {"success": False, "error": "Trivy scan timeout", "vulnerabilities_found": 0}
        except Exception as e:
            logger.error(f"Error during Trivy scan: {e}")
            return {"success": False, "error": str(e), "vulnerabilities_found": 0}

    def _test_docker_image(self, image_tag: str) -> Dict[str, Any]:
        """Run tests inside the built Docker image"""
        logger.info(f"Testing Docker image: {image_tag}")
        try:
            # Run the tests defined in the Docker image
            # This assumes the Dockerfile's CMD or ENTRYPOINT can be overridden for testing,
            # or that tests are accessible and runnable within the image.
            # Example: docker run --rm <image_tag> pytest tests/
            
            # For simplicity, let's assume a test command specified in config
            test_command_in_container = self.config.get('docker', {}).get('test_command_in_container', "pytest tests/")
            
            container: Container = self.docker_client.containers.run(
                image_tag,
                command=test_command_in_container,
                remove=True,
                detach=False,
                environment={"PYTHONUNBUFFERED": "1"}
            )
            
            # The logs from the container will contain test results
            logs = container.logs().decode('utf-8')
            
            # Basic pass/fail check
            success = "failed" not in logs.lower() and "error" not in logs.lower()
            
            mlflow.log_metric("docker_image_test_success", 1 if success else 0)
            mlflow.log_text(logs, "docker_image_test_logs.txt")

            if not success:
                logger.warning(f"Docker image tests failed for {image_tag}")
                return {"success": False, "output": logs[:2000], "error": "Tests failed in container"}

            logger.info(f"Docker image tests passed for {image_tag}")
            return {"success": True, "output": logs[:2000]}

        except docker.errors.ContainerError as e:
            logger.error(f"Docker container error during testing: {e}")
            return {"success": False, "error": str(e), "output": str(e.stderr)}
        except Exception as e:
            logger.error(f"Error testing Docker image: {e}")
            return {"success": False, "error": str(e)}

    def _push_docker_image(self, image_tag: str) -> Dict[str, Any]:
        """Push Docker image to registry"""
        logger.info(f"Pushing Docker image: {image_tag} to registry...")
        try:
            # Push the image
            push_result = self.docker_client.images.push(image_tag)
            logger.info(f"Docker image pushed successfully: {image_tag}")
            return {"success": True, "output": push_result}
        except Exception as e:
            logger.error(f"Error pushing Docker image: {e}")
            raise

    def _get_image_size_mb(self, image_tag: str) -> float:
        """Get Docker image size in MB"""
        try:
            image = self.docker_client.images.get(image_tag)
            return image.attrs["Size"] / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not get image size for {image_tag}: {e}")
            return 0.0

    def _train_models(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate models"""
        logger.info("Training models...")

        # Assume model training is handled by an external script or a dedicated skill
        # For a full implementation, this would trigger a training job
        # and wait for its completion.
        training_config = self.config.get('model_training', {})
        model_name = training_config.get('model_name', 'default_model')
        dataset_path = training_config.get('dataset_path', 'data/training_data.csv')

        with mlflow.start_run(run_name="model_training_run") as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("dataset_path", dataset_path)
            mlflow.log_params(training_config.get('hyperparameters', {}))

            # Simulate training
            logger.info(f"Simulating training for model: {model_name} with dataset: {dataset_path}")
            time.sleep(10) # Simulate training time

            # Simulate metrics
            loss = 0.05
            accuracy = 0.95
            mlflow.log_metrics({"loss": loss, "accuracy": accuracy})

            # Simulate saving model
            model_path = "models/trained_model.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'w') as f:
                f.write("dummy model content") # Placeholder
            mlflow.log_artifact(model_path)

            logger.info(f"Model {model_name} trained. Run ID: {run.info.run_id}")

            return {
                "model_name": model_name,
                "run_id": run.info.run_id,
                "metrics": {"loss": loss, "accuracy": accuracy},
                "model_path": model_path
            }

    def _evaluate_models(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance"""
        logger.info("Evaluating models...")

        evaluation_config = self.config.get('model_evaluation', {})
        model_uri = evaluation_config.get('model_uri', 'models:/default_model/Staging')
        test_dataset_path = evaluation_config.get('test_dataset_path', 'data/test_data.csv')

        with mlflow.start_run(nested=True, run_name="model_evaluation_run") as run:
            mlflow.log_param("model_uri", model_uri)
            mlflow.log_param("test_dataset_path", test_dataset_path)

            # Simulate loading model
            logger.info(f"Simulating loading model from: {model_uri}")
            time.sleep(5) # Simulate loading time

            # Simulate evaluation
            precision = 0.85
            recall = 0.90
            f1_score = 0.87
            mlflow.log_metrics({"precision": precision, "recall": recall, "f1_score": f1_score})

            logger.info(f"Model evaluated. Run ID: {run.info.run_id}")

            # Decide whether to transition model to production based on metrics
            promote_threshold = evaluation_config.get('promote_threshold', 0.8)
            if f1_score > promote_threshold:
                logger.info(f"Model performance (F1: {f1_score}) exceeds threshold ({promote_threshold}). Promoting to production.")
                # Simulate MLflow model stage transition
                # self.mlflow_client.transition_model_version_stage(
                #     name="default_model",
                #     version=1, # This would need to be dynamically determined
                #     stage="Production"
                # )
                promoted = True
            else:
                logger.info(f"Model performance (F1: {f1_score}) below threshold ({promote_threshold}). Not promoting.")
                promoted = False

            return {
                "model_uri": model_uri,
                "metrics": {"precision": precision, "recall": recall, "f1_score": f1_score},
                "promoted_to_production": promoted
            }

    def _security_scan(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Scan for security vulnerabilities"""
        logger.info("Running general security scan...")

        scan_config = self.config.get('security_scan', {})
        # This could be integrating with various security tools
        # For now, simulate a check.
        
        simulated_vulnerabilities = scan_config.get('simulated_vulnerabilities', 5)
        critical_vulnerabilities = scan_config.get('critical_vulnerabilities', 1)

        logger.info(f"Simulating security scan: {simulated_vulnerabilities} vulnerabilities found.")
        time.sleep(5)

        mlflow.log_metrics({
            "security_total_vulnerabilities": simulated_vulnerabilities,
            "security_critical_vulnerabilities": critical_vulnerabilities
        })

        if critical_vulnerabilities > scan_config.get('max_critical_vulnerabilities', 0):
            raise ValueError(f"Critical vulnerabilities found ({critical_vulnerabilities}) exceeds threshold.")

        return {
            "total_vulnerabilities": simulated_vulnerabilities,
            "critical_vulnerabilities": critical_vulnerabilities,
            "success": True
        }

    def _deploy(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy to target environment"""
        logger.info("Deploying models and services...")

        deployment_config = self.config.get('deployment', {})
        target_env = deployment_config.get('target_environment', 'staging')
        model_to_deploy = deployment_config.get('model_uri', 'models:/default_model/Production')
        service_image = self.artifacts.get("build_main_image_tag", "agentic-ai:latest") # Get built image tag

        logger.info(f"Simulating deployment of {model_to_deploy} and service image {service_image} to {target_env}...")
        time.sleep(10)

        # Simulate health checks
        health_check_passed = True
        if deployment_config.get('simulate_health_check_fail', False):
            health_check_passed = False

        mlflow.log_metrics({
            "deployment_success": 1 if health_check_passed else 0,
            "target_environment": target_env
        })

        if not health_check_passed:
            raise ValueError("Deployment health checks failed.")

        return {
            "environment": target_env,
            "model_deployed": model_to_deploy,
            "service_image": service_image,
            "success": True,
            "url": f"http://{target_env}.example.com"
        }

    def _setup_monitoring(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Setup monitoring and alerts"""
        logger.info("Setting up monitoring and alerts...")

        monitoring_config = self.config.get('monitoring', {})
        
        # Simulate integration with monitoring tools (Prometheus, Grafana, PagerDuty, etc.)
        alert_rules_configured = True
        dashboards_created = True

        logger.info("Simulating monitoring setup complete.")
        time.sleep(5)

        mlflow.log_metrics({
            "monitoring_setup_success": 1 if (alert_rules_configured and dashboards_created) else 0,
            "alert_rules_configured": 1 if alert_rules_configured else 0,
            "dashboards_created": 1 if dashboards_created else 0
        })

        return {
            "alert_rules_configured": alert_rules_configured,
            "dashboards_created": dashboards_created,
            "success": True
        }

    def _get_git_sha(self) -> str:
        """Get current git commit SHA"""
        if self.git_repo:
            return self.git_repo.head.commit.hexsha
        return "unknown"

    def _generate_pipeline_report(self, pipeline_result: Dict[str, Any]) -> str:
        """Generate a comprehensive pipeline report"""
        logger.info("Generating pipeline report...")
        report_data = {
            "report_timestamp": datetime.now().isoformat(),
            "git_sha": self._get_git_sha(),
            "pipeline_summary": {
                "overall_success": pipeline_result["overall_success"],
                "duration": pipeline_result["duration"],
                "start_time": pipeline_result["start_time"],
                "end_time": pipeline_result["end_time"],
                "error": pipeline_result["error"]
            },
            "stage_details": pipeline_result["stages"],
            "artifacts_logged": list(mlflow.active_run().list_artifacts()) if mlflow.active_run() else []
        }

        report_path = "reports/pipeline_summary_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        mlflow.log_artifact(report_path)
        logger.info(f"Pipeline report generated at {report_path}")
        return report_path
    
    def _cleanup_temp_files(self):
        """Cleanup temporary files created during pipeline execution."""
        logger.info("Cleaning up temporary files...")
        temp_dockerfile = Path("Dockerfile.pipeline")
        if temp_dockerfile.exists():
            temp_dockerfile.unlink()
            logger.debug(f"Removed temporary Dockerfile: {temp_dockerfile}")
        # Add more cleanup for other temporary files/directories if needed
        logger.info("Temporary file cleanup complete.")

    async def _invoke_mcp_tool(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes an MCP tool as specified in the pipeline configuration.
        """
        logger.info("Invoking MCP tool...")

        if not self.mcp_client:
            raise RuntimeError("MCPClient is not initialized.")

        mcp_tool_config = self.config.get('mcp_tool_invocation', {})
        tool_name = mcp_tool_config.get('tool_name')
        arguments = mcp_tool_config.get('arguments', {})
        security_context = mcp_tool_config.get('security_context', {})

        if not tool_name:
            raise ValueError("MCP tool_name not specified in pipeline config for 'mcp_tool_invocation' stage.")

        try:
            mcp_result = await self.mcp_client.call_tool(
                tool_name=tool_name,
                arguments=arguments,
                caller="MLOpsPipeline",
                security_context=security_context
            )
            logger.info(f"MCP tool '{tool_name}' invoked successfully. Result: {mcp_result}")
            return mcp_result
        except Exception as e:
            logger.error(f"Failed to invoke MCP tool '{tool_name}': {e}", exc_info=True)
            raise
