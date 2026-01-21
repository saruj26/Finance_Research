# Backend Setup and Usage Guide

## Installation

1. Navigate to the backend directory:

```bash
cd backend
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the Backend Server

Start the Flask API server:

```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check

```
GET /health
```

Returns server status

### 2. Predict from Text

```
POST /predict
Content-Type: application/json

{
  "text": "Your financial news article text here..."
}
```

Returns:

```json
{
  "prediction": "up",
  "probability": 0.75,
  "confidence": 0.75,
  "probabilities": {
    "up": 0.75,
    "down": 0.25
  },
  "features": {...},
  "timestamp": "2026-01-20T..."
}
```

### 3. Predict from PDF

```
POST /predict/pdf
Content-Type: multipart/form-data

file: [PDF file]
```

Returns same format as `/predict`

## Architecture

The backend consists of:

1. **app.py** - Flask REST API server with CORS enabled
2. **inference_pipeline.py** - ML inference pipeline that:
   - Loads trained XGBoost model
   - Loads FinBERT for sentiment analysis
   - Extracts features (sentiment, topics, keywords)
   - Makes predictions
   - Handles PDF text extraction

## Features

- ✅ Real-time article analysis
- ✅ PDF text extraction
- ✅ FinBERT sentiment analysis
- ✅ Topic modeling (simplified keyword-based)
- ✅ TF-IDF keyword extraction
- ✅ XGBoost prediction
- ✅ Probability scores
- ✅ CORS enabled for frontend

## Notes

- The current implementation uses simplified topic/keyword extraction
- For production, integrate actual LDA model and TF-IDF vectorizer from research pipeline
- Model is loaded once on startup for faster predictions
- FinBERT model is cached after first load
