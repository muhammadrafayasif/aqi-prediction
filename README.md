# AQI Prediction

A repository for predicting the next 72 hours of AQI using a machine learning model.

## Repository Structure

```
.
├── backend/                # Python API backend for AQI prediction
│   ├── main.py             # Main FastAPI server
│   └── .gitignore
├── data_analysis/          # Jupyter Notebooks for data exploration & analysis
│   ├── correlation.ipynb   # Correlation analysis notebook
│   ├── length.ipynb        # Dataset length/stats notebook
│   ├── line_plot.ipynb     # Line plots of data
│   └── summary.ipynb       # Summary statistics notebook
├── frontend/               # Web frontend (Vite + React)
│   ├── README.md           # Frontend usage and dev instructions
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   ├── eslint.config.js
│   ├── .gitignore
│   ├── public/             # Public assets
│   └── src/                # Application source code
├── .github/
│   ├── actions/            # [Custom composite GH actions, if present]
│   └── workflows/          # GitHub Actions workflow config
├── report.pdf              # Final project report (PDF)
├── .gitignore
├── LICENSE
```

## Model & Training Approach

This project uses a Python backend (FastAPI) for AQI prediction. 
- The backend (`main.py`) provides an API for generating AQI predictions, using a trained machine learning model.
- The data analysis notebooks in `data_analysis/` explore correlations, dataset stats, visualize time series, and summarize project data.
- Model deployment is handled by the backend API, which exposes endpoints for prediction.

## Frontend

The `frontend` directory contains a Vite + React web application for interacting with the prediction API:
- View AQI predictions and the graph of AQI prediction trends
- Source code, build configs and assets for the web UI.

## GitHub Actions

- `.github/workflows/`: contains workflow YAML files for CI/CD automation.
  - Typical roles include training the model, and gathering weather + pollutant data through web scraping
- `.github/actions/`: Detailed instrucutions for data gathering and model training pipeline

## CI/CD Pipelines
- `.github/workflows/add_to_fs.yml` scrapes weather data and air pollutant data and adds them to an online feature store through Hopsworks
- `.github/workflows/train_model.yml` trains the model using `XGBoost` and trains separate horizons `(1h, 6h, 24h, 48h and 72h)` which are combined into one 72h prediction using interpolation

## Analysis Notebooks

- `correlation.ipynb`: Explore how weather and pollution features correlate with AQI values.
- `line_plot.ipynb`: Visualize AQI and other variables over time.
- `length.ipynb`: Dataset stat analysis.
- `summary.ipynb`: Broad feature and outcome summaries.

## License

Distributed under the terms of the [MIT License](https://github.com/muhammadrafayasif/aqi-prediction/blob/main/LICENSE).

## Report

See [`report.pdf`](https://github.com/muhammadrafayasif/aqi-prediction/blob/main/report.pdf) for understanding the motivation for using XGBoost for the model and the correlation of each feature with each other.