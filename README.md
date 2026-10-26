# MachinePulse AI — Industrial Sensor Anomaly Monitoring

**A student-built data science prototype aligned with data preparation, machine learning, visualization, and model monitoring concepts.**

## Problem
Industrial equipment produces time-series readings such as temperature, vibration, pressure, and energy consumption. Unusual combinations can warrant investigation. This demo illustrates how anomaly detection can help surface readings for review.

## Features
- CSV upload and required-column validation
- Data cleaning and numeric conversion
- Standardization and Isolation Forest anomaly detection
- Sensor time-series chart with flagged points
- Correlation matrix and severity summary
- Alert table with simple, interpretable IQR-based evidence
- Downloadable alert log

## Dataset
`sample_sensor_data.csv` contains **synthetic demonstration readings**. Some values were intentionally perturbed to demonstrate anomaly detection. It is not real Siemens data or real factory telemetry.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## GitHub quick start
1. Create a new public or private repository named `machinepulse-ai`.
2. Upload `app.py`, `requirements.txt`, `sample_sensor_data.csv`, and this `README.md`.
3. To run locally, follow the instructions above. For a hosted demo, use Streamlit Community Cloud and select `app.py` as the entry point.

## Suggested screenshots for the report
1. Main dashboard with metric cards
2. Sensor trend with flagged anomalies
3. Correlation matrix
4. Explainable alert log
5. CSV download and testing examples

Only include screenshots from your own running implementation.

## Method
- Validate expected columns and convert timestamps/numeric fields
- Drop rows missing required values
- Standardize the four sensor features
- Fit an Isolation Forest model
- Add IQR-based evidence for readings outside robust bounds

## Limitations
- Synthetic data only; results are illustrative
- No validated fault labels, so accuracy/recall are not claimed
- The model is fitted on the uploaded dataset for demonstration
- Alerts are not equipment diagnoses and must not be used for safety decisions

## Attribution and academic honesty
This is a starter implementation for an independent student project. If you reuse or adapt third-party code or datasets, review their licenses and acknowledge the original sources. Do not describe this as a project assigned by Siemens/EduSkills unless it actually was. The topic is aligned with general data science concepts (data engineering/preparation, machine learning, and monitoring), not an official Siemens project.
