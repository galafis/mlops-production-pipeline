"""
Unit tests for the MLOps Production Pipeline.

Tests cover the MLOpsPipeline class including initialization,
training, deployment, monitoring, and evaluation workflows.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.pipeline import MLOpsPipeline


class TestMLOpsPipelineInit:
    """Test MLOpsPipeline initialization."""

    def test_default_config(self):
        """Test pipeline initializes with default configuration."""
        pipeline = MLOpsPipeline()
        assert pipeline.config is not None
        assert pipeline.config['model_registry'] == 'mlflow'
        assert pipeline.config['monitoring'] is True
        assert pipeline.config['auto_retrain'] is False
        assert pipeline.config['deployment_target'] == 'kubernetes'

    def test_custom_config(self):
        """Test pipeline initializes with custom configuration."""
        custom_config = {
            'model_registry': 'custom_registry',
            'monitoring': False,
            'auto_retrain': True,
            'deployment_target': 'docker'
        }
        pipeline = MLOpsPipeline(config=custom_config)
        assert pipeline.config == custom_config

    def test_initial_model_is_none(self):
        """Test that model is None before training."""
        pipeline = MLOpsPipeline()
        assert pipeline.model is None

    def test_initial_metrics_empty(self):
        """Test that metrics dict is empty before training."""
        pipeline = MLOpsPipeline()
        assert pipeline.metrics == {}


class TestMLOpsPipelineTraining:
    """Test MLOpsPipeline training workflow."""

    def test_train_sets_model(self):
        """Test that training sets the model attribute."""
        pipeline = MLOpsPipeline()
        pipeline.train(data=[1, 2, 3])
        assert pipeline.model is not None

    def test_train_populates_metrics(self):
        """Test that training populates metrics."""
        pipeline = MLOpsPipeline()
        pipeline.train(data=[1, 2, 3])
        assert 'accuracy' in pipeline.metrics
        assert 'loss' in pipeline.metrics
        assert 'training_time' in pipeline.metrics

    def test_train_accuracy_value(self):
        """Test training accuracy is in valid range."""
        pipeline = MLOpsPipeline()
        pipeline.train(data=[1, 2, 3])
        assert 0.0 <= pipeline.metrics['accuracy'] <= 1.0

    def test_train_loss_value(self):
        """Test training loss is non-negative."""
        pipeline = MLOpsPipeline()
        pipeline.train(data=[1, 2, 3])
        assert pipeline.metrics['loss'] >= 0.0


class TestMLOpsPipelineDeployment:
    """Test MLOpsPipeline deployment workflow."""

    def test_register_model(self):
        """Test model registration returns valid URI."""
        pipeline = MLOpsPipeline()
        uri = pipeline.register_model("test_model", "v1")
        assert "mlflow" in uri
        assert "test_model" in uri
        assert "v1" in uri

    def test_deploy_returns_info(self):
        """Test deployment returns deployment information."""
        pipeline = MLOpsPipeline()
        info = pipeline.deploy("mlflow://test_model/v1")
        assert info['status'] == 'deployed'
        assert info['replicas'] == 3
        assert 'endpoint' in info

    def test_deploy_to_staging(self):
        """Test deployment to staging environment."""
        pipeline = MLOpsPipeline()
        info = pipeline.deploy("mlflow://test_model/v1", environment='staging')
        assert info['environment'] == 'staging'

    def test_deploy_to_production(self):
        """Test deployment to production environment."""
        pipeline = MLOpsPipeline()
        info = pipeline.deploy("mlflow://test_model/v1", environment='production')
        assert info['environment'] == 'production'


class TestMLOpsPipelineMonitoring:
    """Test MLOpsPipeline monitoring capabilities."""

    def test_monitor_returns_metrics(self):
        """Test monitoring returns expected metrics."""
        pipeline = MLOpsPipeline()
        metrics = pipeline.monitor()
        assert 'requests_per_second' in metrics
        assert 'latency_p95' in metrics
        assert 'error_rate' in metrics
        assert 'model_drift' in metrics

    def test_monitor_error_rate_low(self):
        """Test that error rate is within acceptable range."""
        pipeline = MLOpsPipeline()
        metrics = pipeline.monitor()
        assert metrics['error_rate'] < 0.01

    def test_monitor_latency_acceptable(self):
        """Test that P95 latency is acceptable."""
        pipeline = MLOpsPipeline()
        metrics = pipeline.monitor()
        assert metrics['latency_p95'] < 100  # ms


class TestMLOpsPipelineProcess:
    """Test MLOpsPipeline data processing."""

    def test_process_auto_trains(self):
        """Test process auto-trains if no model exists."""
        pipeline = MLOpsPipeline()
        assert pipeline.model is None
        result = pipeline.process(data=[1, 2, 3])
        assert pipeline.model is not None
        assert result is not None

    def test_process_returns_metrics(self):
        """Test process returns metrics dict."""
        pipeline = MLOpsPipeline()
        result = pipeline.process(data=[1, 2, 3])
        assert isinstance(result, dict)
        assert 'accuracy' in result


class TestMLOpsPipelineEvaluation:
    """Test MLOpsPipeline evaluation."""

    def test_evaluate_returns_metrics(self):
        """Test evaluation returns standard ML metrics."""
        pipeline = MLOpsPipeline()
        metrics = pipeline.evaluate(test_data=[1, 2, 3])
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

    def test_evaluate_metrics_in_range(self):
        """Test all evaluation metrics are between 0 and 1."""
        pipeline = MLOpsPipeline()
        metrics = pipeline.evaluate(test_data=[1, 2, 3])
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range"


class TestProjectStructure:
    """Test project file structure."""

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_path = Path(__file__).parent.parent / "src"
        assert src_path.exists(), "src directory should exist"

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        assert req_path.exists(), "requirements.txt should exist"

    def test_readme_exists_and_substantial(self):
        """Test that README.md exists and has substantial content."""
        readme_path = Path(__file__).parent.parent / "README.md"
        assert readme_path.exists(), "README.md should exist"
        content = readme_path.read_text(encoding='utf-8')
        assert len(content) > 500, "README should have substantial content"

    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile should exist"

    def test_license_exists(self):
        """Test that LICENSE file exists."""
        license_path = Path(__file__).parent.parent / "LICENSE"
        assert license_path.exists(), "LICENSE should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
