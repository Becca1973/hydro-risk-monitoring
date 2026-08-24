# Hydro Risk Monitoring

An end-to-end machine learning system for hydrological monitoring, water-level forecasting, and risk classification using hydrological and weather data.

The system combines automated data collection and preprocessing, time-series forecasting, risk classification, data validation and monitoring, REST API development, interactive visualization, and automated ML workflows.

## Overview

Hydro Risk Monitoring processes hydrological and weather data through a reproducible machine learning pipeline to provide water-level predictions and hydrological risk estimates.

### Key Features

* Automated collection and preprocessing of hydrological and weather data
* Water-level forecasting using an LSTM neural network
* Hydrological risk classification using a neural network classifier
* Data validation with Great Expectations
* Experiment tracking with MLflow
* Data and model monitoring with Evidently
* Data and model versioning with DVC
* REST API built with FastAPI
* Interactive visualization with Streamlit and Folium
* Automated workflows with GitHub Actions
* Containerized deployment with Docker
* Model export to ONNX

## Tech Stack

### Backend & API

* Python
* FastAPI
* Uvicorn

### Machine Learning & Data

* TensorFlow / Keras
* scikit-learn
* pandas
* NumPy
* ONNX / ONNX Runtime

### MLOps & Data Quality

* DVC
* MLflow
* Great Expectations
* Evidently

### Visualization

* Streamlit
* Folium

### DevOps & Automation

* Docker
* GitHub Actions
* Poetry
* Git

## Machine Learning Pipeline

The project uses **DVC** to define and reproduce the data and machine learning pipeline.

The pipeline includes:

1. fetching hydrological data
2. preprocessing hydrological data
3. fetching weather data
4. preprocessing weather data
5. validating data with Great Expectations
6. testing and preparing reference data
7. training the global water-level forecasting model
8. training the hydrological risk classifier
9. preparing production data
10. exporting trained models to ONNX

DVC is used to manage pipeline dependencies and version data and model artifacts.

## Models

### Water-Level Forecasting

Water levels are predicted using a global **LSTM (Long Short-Term Memory)** neural network implemented with TensorFlow/Keras.

The model uses historical sequences of hydrological measurements to learn temporal patterns and predict future water levels.

Input features include:

* water level
* water flow
* water temperature

The data is processed as time-series windows before being passed to the LSTM model.

### Hydrological Risk Classification

Hydrological risk is estimated using a separate **feed-forward neural network classifier**.

The classifier combines hydrological measurements with weather observations, including features such as:

* water level and flow
* water temperature
* air temperature
* relative humidity
* precipitation
* wind information
* snow observations

Hydrological measurements are matched with corresponding weather observations before being used for risk classification.

The classifier produces hydrological risk-level predictions using a multi-class softmax output.

## Data Collection

Hydrological and weather observations are collected from **ARSO (Slovenian Environment Agency)** data sources.

Automated workflows were used to periodically retrieve new measurements and update the datasets used by the system.

The collected data is then cleaned and transformed before being passed to validation and modeling stages.

## Data Quality & Monitoring

Data validation and monitoring are integrated into the machine learning workflow.

**Great Expectations** is used to validate preprocessed hydrological and weather data before downstream processing.

**Evidently** is used for data and model monitoring and for generating monitoring reports.

These components help detect data-quality problems and changes in incoming data.

## Experiment Tracking

**MLflow** is used to track machine learning experiments, including model parameters, evaluation metrics, and trained model artifacts.

This makes it possible to compare model runs and maintain a record of the modeling process.

## REST API

The trained models are integrated into a **FastAPI** backend that exposes model functionality through REST endpoints.

This separates the machine learning layer from the user interface and allows predictions to be consumed programmatically.

## User Interface

An interactive user interface was developed using **Streamlit** and **Folium**.

The interface provides a map-based visualization of hydrological monitoring locations and allows model predictions and risk information to be presented in a more accessible way.

## MLOps & Automation

The project incorporates several MLOps practices:

* reproducible ML pipelines with DVC
* data and model artifact versioning
* experiment tracking with MLflow
* automated data validation
* data and model monitoring
* automated workflows with GitHub Actions
* model export to ONNX
* containerized deployment with Docker

GitHub Actions were also used to automate recurring data collection and parts of the data and model workflow.

## Project Structure

```text
hydro-risk-monitoring/
├── .dvc/                  # DVC configuration
├── .github/workflows/     # GitHub Actions workflows
├── data/                  # Data and metadata
├── gx/                    # Great Expectations configuration
├── reports/               # Modeling and monitoring reports
├── src/
│   ├── api/               # FastAPI backend
│   ├── data/              # Data collection and preprocessing
│   ├── model/             # Model training and export
│   └── ui/                # Streamlit user interface
├── Dockerfile             # Backend container
├── Dockerfile.ui          # UI container
├── dvc.yaml               # DVC pipeline definition
├── dvc.lock               # Reproducible pipeline state
├── params.yaml            # Pipeline and model parameters
├── pyproject.toml         # Dependencies and project configuration
└── start.sh               # Application startup script
```

## Author

**Rebeka Cep**
Master's student in Informatics and Data Technologies
Faculty of Electrical Engineering and Computer Science (FERI), University of Maribor
