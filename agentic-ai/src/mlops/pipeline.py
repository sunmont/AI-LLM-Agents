"""
MLOps pipeline with CI/CD, Docker, and model versioning
"""
from typing import Dict, List, Any, Optional
import yaml
import docker
import mlflow
import git
from datetime import datetime
import subprocess
import json
import hashlib
from pathlib import Path


class MLOpsPipeline:
    """End-to-end MLOps pipeline for AI services"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.docker_client = docker.from_env()
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.git_repo = git.Repo(search_parent_directories=True)

        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def ci_cd_pipeline(self, trigger_event: Dict[str, Any]):
        """CI/CD pipeline triggered by events"""
        print(f"CI/CD Pipeline triggered by {trigger_event.get('event_type')}")

        # 1. Code quality checks
        self._run_code_quality_checks()

        # 2. Run tests
        test_results = self._run_tests()

        # 3. Build and test Docker image
        if test_results['passed']:
            docker_image = self._build_docker_image()
            self._test_docker_image(docker_image)

            # 4. Model training (if triggered)
            if trigger_event.get('train_model', False):
                model_info = self._train_model(trigger_event)

                # 5. Model evaluation
                evaluation_results = self._evaluate_model(model_info)

                # 6. Model deployment (if meets criteria)
                if evaluation_results['deploy']:
                    self._deploy_model(model_info, docker_image)

        # 7. Update monitoring
        self._update_monitoring()

    def _run_code_quality_checks(self):
        """Run code quality and security checks"""
        print("Running code quality checks...")

        checks = [
            ("pylint", ["pylint", "src/", "--rcfile=.pylintrc"]),
            ("mypy", ["mypy", "src/", "--ignore-missing-imports"]),
            ("bandit", ["bandit", "-r", "src/", "-f", "json"]),
            ("black", ["black", "--check", "src/"]),
            ("isort", ["isort", "--check-only", "src/"])
        ]

        results = {}
        for check_name, command in checks:
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                results[check_name] = {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
            except Exception as e:
                results[check_name] = {"success": False, "error": str(e)}

        # Log results
        with open("code_quality_report.json", "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _run_tests(self) -> Dict[str, Any]:
        """Run test suite"""
        print("Running tests...")

        test_cmds = [
            ["pytest", "tests/unit/", "-v", "--cov=src", "--cov-report=xml"],
            ["pytest", "tests/integration/", "-v"],
            ["pytest", "tests/e2e/", "-v", "--timeout=300"]
        ]

        test_results = []
        for cmd in test_cmds:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                test_results.append({
                    "command": cmd[0],
                    "success": result.returncode == 0,
                    "coverage": self._parse_coverage(result.stdout)
                })
            except Exception as e:
                test_results.append({"command": cmd[0], "success": False, "error": str(e)})

        overall_success = all(r["success"] for r in test_results)

        return {
            "passed": overall_success,
            "results": test_results,
            "timestamp": datetime.now().isoformat()
        }

    def _build_docker_image(self) -> str:
        """Build Docker image for AI service"""
        print("Building Docker image...")

        # Create Dockerfile dynamically if needed
        dockerfile_content = self._generate_dockerfile()
        with open("Dockerfile.ci", "w") as f:
            f.write(dockerfile_content)

        # Build image
        image_tag = f"{self.config['docker']['registry']}/{self.config['docker']['image_name']}:{self._get_git_sha()[:8]}"

        try:
            image, build_logs = self.docker_client.images.build(
                path=".",
                dockerfile="Dockerfile.ci",
                tag=image_tag,
                buildargs=self.config['docker'].get('build_args', {}),
                rm=True
            )

            # Push to registry if configured
            if self.config['docker'].get('push_to_registry', False):
                self._push_docker_image(image_tag)

            print(f"Docker image built: {image_tag}")
            return image_tag

        except docker.errors.BuildError as e:
            print(f"Docker build failed: {e}")
            raise

    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for the AI service"""
        base_image = self.config['docker']['base_image']

        dockerfile = f"""
FROM {base_image}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .
COPY setup.py .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 agentic
USER agentic

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

EXPOSE 8080

CMD ["python", "-m", "src.api.server"]
"""
        return dockerfile

    def _train_model(self, trigger_event: Dict[str, Any]) -> Dict[str, Any]:
        """Train model with experiment tracking"""
        print("Training model...")

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(trigger_event.get('training_params', {}))
            mlflow.log_param("git_commit", self._get_git_sha())
            mlflow.log_param("docker_image", trigger_event.get('docker_image'))

            # Train model
            from src.models.finetuning.trainer import ModelFineTuner, FineTuningConfig

            config = FineTuningConfig(
                base_model=trigger_event['model_config']['base_model'],
                dataset_path=trigger_event['model_config']['dataset'],
                output_dir=f"models/{run.info.run_id}",
                training_method=trigger_event['model_config'].get('method', 'sft')
            )

            trainer = ModelFineTuner(config)
            trainer.prepare_model_and_tokenizer()
            dataset = trainer.load_dataset()

            # Train based on method
            if config.training_method == 'sft':
                trainer.train_sft(dataset)
            elif config.training_method == 'dpo':
                trainer.train_dpo(dataset)
            elif config.training_method == 'grpo':
                trainer.train_grpo(dataset)

            # Save model
            trainer.save_model()

            # Log artifacts
            mlflow.log_artifacts(config.output_dir, "model")

            # Log metrics
            metrics = trainer.evaluate(dataset['test'] if 'test' in dataset else dataset['train'])
            mlflow.log_metrics(metrics)

        return {
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "metrics": metrics
        }

    def _deploy_model(self, model_info: Dict[str, Any], docker_image: str):
        """Deploy model to serving infrastructure"""
        print(f"Deploying model {model_info['run_id']}...")

        # Update model registry
        model_name = self.config['model_registry']['model_name']
        model_version = self.mlflow_client.create_model_version(
            name=model_name,
            source=model_info['model_uri'],
            run_id=model_info['run_id']
        )

        # Transition model stage
        self.mlflow_client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        # Deploy to Kubernetes (if configured)
        if self.config.get('kubernetes', {}).get('enabled', False):
            self._deploy_to_kubernetes(model_name, model_version.version, docker_image)

        # Update API gateway
        self._update_api_gateway(model_name, model_version.version)

        print(f"Model deployed: {model_name} v{model_version.version}")

    def _deploy_to_kubernetes(self, model_name: str, version: str, docker_image: str):
        """Deploy to Kubernetes using infrastructure-as-code"""
        from ..infra.kubernetes.deployer import KubernetesDeployer

        deployer = KubernetesDeployer(self.config['kubernetes'])

        # Generate deployment manifests
        manifests = deployer.generate_manifests(
            model_name=model_name,
            model_version=version,
            docker_image=docker_image,
            replicas=self.config['kubernetes'].get('replicas', 2),
            resources=self.config['kubernetes'].get('resources', {})
        )

        # Apply manifests
        deployer.apply_manifests(manifests)

    def _update_api_gateway(self, model_name: str, version: str):
        """Update API gateway routing"""
        # Implement API gateway update logic
        pass

    def _get_git_sha(self) -> str:
        """Get current git commit SHA"""
        return self.git_repo.head.object.hexsha

    def _parse_coverage(self, pytest_output: str) -> float:
        """Parse coverage from pytest output"""
        # Simplified parsing
        import re
        match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', pytest_output)
        return float(match.group(1)) if match else 0.0


class ModelVersioning:
    """Model versioning and management"""

    def __init__(self, registry_uri: str):
        mlflow.set_tracking_uri(registry_uri)
        self.client = mlflow.tracking.MlflowClient()

    def register_model(self, run_id: str, model_name: str,
                       description: str = "") -> str:
        """Register model in MLflow"""
        model_uri = f"runs:/{run_id}/model"

        # Create model if doesn't exist
        try:
            self.client.get_registered_model(model_name)
        except mlflow.exceptions.RestException:
            self.client.create_registered_model(model_name, description)

        # Create model version
        model_version = self.client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        return model_version.version

    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to different stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )

    def get_production_model(self, model_name: str) -> Optional[Dict]:
        """Get current production model"""
        try:
            model = self.client.get_registered_model(model_name)
            for mv in model.latest_versions:
                if mv.current_stage == "Production":
                    return {
                        "version": mv.version,
                        "run_id": mv.run_id,
                        "source": mv.source,
                        "status": mv.status
                    }
        except mlflow.exceptions.RestException:
            return None

        return None

    def compare_models(self, model_name: str, version1: str, version2: str) -> Dict:
        """Compare two model versions"""
        v1 = self.client.get_model_version(model_name, version1)
        v2 = self.client.get_model_version(model_name, version2)

        # Get metrics for comparison
        run1 = mlflow.get_run(v1.run_id)
        run2 = mlflow.get_run(v2.run_id)

        comparison = {
            "version1": {
                "version": version1,
                "metrics": run1.data.metrics,
                "params": run1.data.params,
                "tags": run1.data.tags
            },
            "version2": {
                "version": version2,
                "metrics": run2.data.metrics,
                "params": run2.data.params,
                "tags": run2.data.tags
            },
            "differences": self._calculate_differences(run1, run2)
        }

        return comparison

    def _calculate_differences(self, run1, run2) -> Dict:
        """Calculate differences between runs"""
        differences = {}

        # Compare metrics
        for metric in set(run1.data.metrics.keys()) | set(run2.data.metrics.keys()):
            val1 = run1.data.metrics.get(metric, 0)
            val2 = run2.data.metrics.get(metric, 0)
            if val1 != val2:
                differences[f"metric_{metric}"] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": val2 - val1
                }

        return differences