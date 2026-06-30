# 🌪️ Cyclone Track Forecasting System

## Overview

Cyclone Track Forecasting System is an end-to-end Machine Learning project that predicts future tropical cyclone locations using historical cyclone observations from the IBTrACS dataset.

The project includes data preprocessing, feature engineering, machine learning model development, model evaluation, REST API deployment using FastAPI, and an interactive Streamlit dashboard for real-time predictions.

---

## Project Architecture

```text
IBTrACS Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
Storm-Based Train/Test Split
       │
       ▼
Machine Learning Models
(Random Forest, XGBoost, LSTM)
       │
       ▼
Model Evaluation
       │
       ▼
FastAPI Backend
       │
       ▼
Streamlit Dashboard
```

---

## Dataset

**Source:** IBTrACS (International Best Track Archive for Climate Stewardship)

### Dataset Statistics

* Total Records: 267,387
* Tropical Cyclones: 4,859+
* Time Period: 1980–Present
* Forecast Horizons:

  * 6 Hours
  * 12 Hours
  * 24 Hours
  * 48 Hours
  * 72 Hours

---

## Feature Engineering

### Temporal Features

* Month
* Day
* Hour
* Day of Year

### Cyclic Features

* Month Sin/Cos
* Hour Sin/Cos

### Lag Features

* Latitude Lag 1–3
* Longitude Lag 1–3
* Wind Lag 1–3
* Pressure Lag 1–3

### Motion Features

* Delta Latitude
* Delta Longitude
* Delta Wind
* Delta Pressure
* Movement Distance

---

## Models

### Random Forest Regressor

Best Performing Model

**Performance**

| Metric           | Value    |
| ---------------- | -------- |
| MAE              | 0.4319   |
| RMSE             | 5.3203   |
| R² Score         | 0.9972   |
| Mean Track Error | 64.56 km |

---

### XGBoost Regressor

| Metric           | Value    |
| ---------------- | -------- |
| MAE              | 0.4447   |
| RMSE             | 5.9046   |
| R² Score         | 0.9970   |
| Mean Track Error | 68.28 km |

---

### LSTM Network

Implemented using sequence modeling with historical cyclone trajectories.

---

## FastAPI Deployment

The trained model is deployed using FastAPI.

### Run API

```bash
uvicorn app.main:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Streamlit Dashboard

Interactive web application for cyclone track prediction.

### Run Dashboard

```bash
streamlit run streamlit_app.py
```

Features:

* Real-Time Prediction
* Interactive Map Visualization
* Model Information Display
* FastAPI Integration

---

## Project Structure

```text
Cyclone/
│
├── app/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
│
├── streamlit_app.py
├── Dockerfile
├── README.md
└── requirements.txt
```

---

## Results

Random Forest achieved the best performance with a mean cyclone track prediction error of **64.56 km** while maintaining strict storm-wise train-test separation to prevent data leakage.

---

## Future Improvements

* Multi-Horizon Forecasting (6h, 12h, 24h, 48h, 72h)
* Attention-Based LSTM Models
* Transformer-Based Track Prediction
* Docker Deployment
* Cloud Deployment (AWS/Azure/GCP)
* Real-Time Weather Data Integration

---

## Author

Om Prakash Gaini

M.Tech – AI for Sustainability
Indian Institute of Technology Kanpur
