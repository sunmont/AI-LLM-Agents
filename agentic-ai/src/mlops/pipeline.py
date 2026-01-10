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

    def __init__(self, config_path: str):
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

    def add_stage(self, stage: PipelineStage):
        """Add a stage to the pipeline"""
        self.stages[stage.name] = stage
        logger.info(f"Added pipeline stage: {stage.name}")

    def remove_stage(self, stage_name: str):
        """Remove a stage from the pipeline"""
        if stage_name in self.stages:
            del self.stages[stage_name]
            logger.info(f"Removed pipeline stage: {stage_name}")

    def execute_pipeline(self, trigger_event: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the complete pipeline"""
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

                # Execute stages in parallel where possible
                with ThreadPoolExecutor(max_workers=len(executable_stages)) as executor:
                    future_to_stage = {
                        executor.submit(self._execute_stage, stage, trigger_event): stage
                        for stage in executable_stages
                    }

                    for future in as_completed(future_to_stage):
                        stage = future_to_stage[future]
                        try:
                            result = future.result(timeout=stage.timeout)
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

                        except Exception as e:
                            logger.error(f"Stage {stage.name} execution failed: {e}")
                            pipeline_result["overall_success"] = False
                            pipeline_result["error"] = str(e)
                            break

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

    def _execute_stage(self, stage: PipelineStage, trigger_event: Dict[str, Any]) -> PipelineResult:
        """Execute a single pipeline stage with retries"""
        start_time = datetime.now()
        last_error = None

        for attempt in range(stage.retry_count):
            try:
                logger.info(f"Executing stage {stage.name} (attempt {attempt + 1}/{stage.retry_count})")

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
        image_tag = f"{docker_config.get('registry', 'localhost:5000')}/" \
                   f"{docker_config.get('image_name', 'agentic-ai')}:" \
                   f"{self._get_git_sha()[:8]}"

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