# XGBoost Phishing Email Detection

This project implements a high-performance phishing email detection system using an XGBoost classifier. It analyzes the subject and body of emails, extracts a rich set of text-based features, and predicts whether an email is phishing or legitimate.

The system includes scripts for training, evaluation, batch prediction, and a FastAPI server for real-time inference with Splunk integration.

## Features

- **High-Performance Model**: Utilizes XGBoost for fast and accurate classification.
- **Rich Feature Engineering**: Extracts over 50 features from email text, including:
  - Length and word counts.
  - Ratios of special characters, digits, and uppercase letters.
  - Shannon entropy to detect randomness.
  - Counts of suspicious keywords (urgency, financial, security).
  - URL analysis (count, length, IP addresses, suspicious TLDs).
  - Structural patterns (`click here`, `verify your account`, etc.).
- **Command-Line Interface**: Easy-to-use scripts for training, evaluation, and batch prediction.
- **Real-time API**: A FastAPI server to serve predictions over HTTP.
- **Splunk Integration**: The API server logs prediction events and errors to Splunk for monitoring and security analytics.
- **Comprehensive Evaluation**: Training script generates a detailed performance report (`metrics_report.json`) and saves the model for deployment.

## Project Structure

```
XgBoost/
├── Enron.csv                   # Dataset for training (must be obtained separately)
├── api_server_fastapi.py       # FastAPI server for real-time predictions
├── feature_extraction_text.py  # Core logic for text feature extraction
├── predict_phishing.py         # CLI for model evaluation and batch prediction
├── train_text_phishing.py      # Script to train the XGBoost model
├── requirements.txt            # Python dependencies
├── phishing_text_model.joblib  # Saved model file (output of training)
└── metrics_report.json         # Performance metrics report (output of training)
```

## Setup and Installation

### 1. Prerequisites

- Python 3.8+
- Git

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd security-analytics-2/XgBoost
```

### 3. Install Dependencies

# Core ML libraries
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# Feature extraction
tldextract>=3.4.0

# API server
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Splunk integration
requests>=2.31.0

It is recommended to use a virtual environment to manage dependencies.

```bash
# Create and activate a virtual environment (e.g., using venv)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install -r requirements.txt
```

### 4. Download the Dataset

The training script uses the `Enron.csv` dataset. You can find a version of this dataset on Kaggle: Phishing Email Dataset.

Download `Enron.csv` and place it in the `XgBoost/` directory.

## Usage

### 1. Train the Model

To train the XGBoost classifier, run the training script. This will process `Enron.csv`, train the model, and save `phishing_text_model.joblib` and `metrics_report.json`.

```bash
python train_text_phishing.py
```

The script will output a detailed evaluation summary on the holdout test set.

### 2. Evaluate the Model

You can evaluate the trained model against any labeled CSV file containing `subject`, `body`, and `label` columns.

```bash
# Example: Evaluate on the test portion of the Enron dataset
python predict_phishing.py evaluate --input Enron.csv --model phishing_text_model.joblib
```

### 3. Run Batch Predictions

To predict on a CSV file containing `subject` and `body` columns, use the `batch` command. The predictions will be saved to a new CSV file.

```bash
python predict_phishing.py batch --input your_emails.csv --output predictions.csv
```

## API Server

The project includes a FastAPI server for serving real-time predictions. It also logs detailed events to a configured Splunk instance.

### 1. Configure Splunk (Optional)

The API server sends logs to Splunk via the HTTP Event Collector (HEC). Set the following environment variables to enable this feature:

```bash
export SPLUNK_HEC_URL="https://your-splunk-server:8088/services/collector"
export SPLUNK_HEC_TOKEN="your-hec-token"
export SPLUNK_INDEX="phishing_detection"
```

### 2. Run the Server

Start the server using `uvicorn`. You can specify a port as a command-line argument.

```bash
# Run on default port 8000
python api_server_fastapi.py

# Run on a custom port
python api_server_fastapi.py 8080
```

Once running, the API documentation will be available at `http://localhost:8000/docs`.

### API Endpoints

#### `POST /predict`

Predicts a single email.

- **Request Body**:
  ```json
  {
    "subject": "Urgent: Verify Your Account",
    "body": "Click here to confirm your details or your account will be suspended."
  }
  ```
- **Response**:
  ```json
  {
    "is_phishing": true,
    "phishing_probability": 0.987,
    "confidence": 0.974,
    "label": "Phishing"
  }
  ```

#### `POST /predict/csv`

Performs batch predictions on an uploaded CSV file.

- **Request**: `multipart/form-data` request with a file attached. The CSV must contain `subject` and `body` columns.
- **Example `curl` command**:
  ```bash
  curl -X POST "http://localhost:8000/predict/csv" -F "file=@/path/to/your_emails.csv"
  ```
- **Response**: A JSON object containing a summary and a list of predictions for each row.

#### `GET /health`

A simple health check endpoint.

- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```