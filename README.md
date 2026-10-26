MachinePulse AI

Industrial Sensor Anomaly Detection and Monitoring

MachinePulse AI is an independent student project that demonstrates how machine learning can help identify unusual patterns in industrial sensor readings.

The application provides an interactive dashboard for exploring sensor trends, detecting potential anomalies, examining sensor relationships, and reviewing explainable alerts.

Features

- Sensor monitoring: Visualize temperature, vibration, pressure, and energy readings.
- Anomaly detection: Use the Isolation Forest machine-learning algorithm to flag unusual sensor patterns.
- Interactive visualizations: Explore sensor trends and correlations through charts.
- Alert severity: Review flagged readings grouped by severity.
- Explainable alert log: Inspect anomaly scores and the sensor evidence associated with alerts.
- CSV support: Load sample sensor data and download the alert log.

Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- NumPy

Project Files

- "app.py" — Main Streamlit application.
- "requirements.txt" — Python package dependencies.
- "sample_sensor_data.csv" — Sample sensor dataset.
- ".gitignore" — Files excluded from version control.

Installation and Usage

1. Clone the repository

git clone https://github.com/Sayandana-v/MachinePulse-AI.git
cd MachinePulse-AI

2. Create a virtual environment

python -m venv .venv

Activate it on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r requirements.txt

4. Run the application

python -m streamlit run app.py

Open the local URL displayed in the terminal, usually "http://localhost:8501".

How It Works

1. Sensor readings are loaded from a CSV file.
2. The application prepares the sensor data for analysis.
3. An Isolation Forest model identifies readings that differ from typical patterns.
4. The dashboard presents trends, flagged anomalies, severity summaries, and alert evidence.

Project Status

Status: Student prototype.

This project uses synthetic demonstration data and is not connected to live factory equipment. Its alerts are intended for educational demonstration, not as a substitute for industrial safety systems or professional engineering assessments.

Author

Sayandana V

GitHub: "Sayandana-v" (https://github.com/Sayandana-v)
