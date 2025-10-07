# MLOps Production Pipeline

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED.svg)
![Kubernetes](https://img.shields.io/badge/Kubernetes-1.27+-326CE5.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-0194E2.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Complete MLOps pipeline with experiment tracking, model versioning, deployment, and monitoring**

[English](#english) | [Português](#português)

</div>

---

## English

### 📋 Overview

Production-ready MLOps pipeline implementing best practices for machine learning operations. Includes data versioning (DVC), experiment tracking (MLflow), model registry, containerization (Docker), orchestration (Kubernetes), CI/CD (GitHub Actions), monitoring (Prometheus/Grafana), and automated retraining.

### 🎯 Key Features

- **Data Versioning**: DVC for dataset and feature versioning
- **Experiment Tracking**: MLflow for metrics, parameters, and artifacts
- **Model Registry**: Centralized model management and versioning
- **Containerization**: Docker for reproducible environments
- **Orchestration**: Kubernetes for scalable deployment
- **CI/CD**: Automated testing and deployment pipelines
- **Monitoring**: Real-time model performance tracking
- **A/B Testing**: Gradual rollout and comparison

### 🚀 Quick Start

```bash
git clone https://github.com/galafis/mlops-production-pipeline.git
cd mlops-production-pipeline

# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Train model with tracking
python src/models/train.py --experiment-name my-experiment

# Build Docker image
docker build -t ml-model:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
```

### 📊 Pipeline Architecture

```
Data → DVC → Feature Engineering → Model Training → MLflow
                                         ↓
                                   Model Registry
                                         ↓
                              Docker Containerization
                                         ↓
                              Kubernetes Deployment
                                         ↓
                            Monitoring & Retraining
```

### 👤 Author

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)

---

## Português

### 📋 Visão Geral

Pipeline MLOps pronto para produção implementando melhores práticas para operações de machine learning. Inclui versionamento de dados (DVC), tracking de experimentos (MLflow), registro de modelos, containerização (Docker), orquestração (Kubernetes), CI/CD (GitHub Actions), monitoramento (Prometheus/Grafana) e retreinamento automatizado.

### 🎯 Características Principais

- **Versionamento de Dados**: DVC para versionamento de datasets e features
- **Tracking de Experimentos**: MLflow para métricas, parâmetros e artefatos
- **Registro de Modelos**: Gerenciamento centralizado e versionamento de modelos
- **Containerização**: Docker para ambientes reproduzíveis
- **Orquestração**: Kubernetes para deployment escalável
- **CI/CD**: Pipelines automatizados de teste e deployment
- **Monitoramento**: Tracking de performance do modelo em tempo real
- **Testes A/B**: Rollout gradual e comparação

### 👤 Autor

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)
