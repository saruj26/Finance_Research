# Full Stack Integration Guide

# S&P 500 Index Movement Prediction System

## System Architecture

```
Frontend (React + TypeScript)
      ↓
   HTTP API
      ↓
Backend Flask Server (Port 5000)
      ↓
Inference Pipeline
   ├── Text Preprocessing
   ├── FinBERT Sentiment Analysis
   ├── Topic Feature Extraction (LDA)
   ├── TF-IDF Keyword Extraction
   └── XGBoost Model Prediction
      ↓
   Result (UP/DOWN)
```

## Quick Start

### 1. Start Backend Server

```bash
# Navigate to project root
cd C:\Research\Finance_Research

# Activate virtual environment
.venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Start Flask server
python app.py
```

Server will start at: `http://localhost:5000`

### 2. Start Frontend Development Server

```bash
# Open new terminal
cd C:\Research\Finance_Research\frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Frontend will start at: `http://localhost:5173`

## Usage

1. Open browser to `http://localhost:5173`
2. Choose upload method:
   - **Upload PDF**: Click to upload financial news PDF
   - **Paste Text**: Paste article text directly
3. Click "Analyze Article"
4. Watch the prediction lights:
   - **Green UP**: Index predicted to move up
   - **Red DOWN**: Index predicted to move down

## Files Created/Modified

### Backend (New)

- `backend/app.py` - Flask API server
- `backend/inference_pipeline.py` - ML inference pipeline
- `backend/requirements.txt` - Python dependencies
- `backend/README.md` - Backend documentation

### Frontend (Modified)

- `frontend/src/App.tsx` - Added state management for predictions
- `frontend/src/components/ArticleUpload.tsx` - Added API integration
- `frontend/src/components/ArticleUpload.css` - Added loading/error styles
- `frontend/src/components/PredictionIndicator.tsx` - Added live prediction display
- `frontend/src/components/PredictionIndicator.css` - Added probability bars

## API Examples

### Test with curl:

```bash
# Health check
curl http://localhost:5000/health

# Predict from text
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Federal Reserve raises interest rates amid inflation concerns. Markets show volatility as investors react to monetary policy changes.\"}"

# Predict from PDF
curl -X POST http://localhost:5000/predict/pdf \
  -F "file=@article.pdf"
```

## Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError`

```bash
# Solution: Install missing packages
pip install -r backend/requirements.txt
```

**Problem**: Model not found

```bash
# Solution: Train the model first
cd Code
python XGBoost_training.py
```

**Problem**: Port 5000 already in use

```python
# Solution: Change port in backend/app.py
app.run(host='0.0.0.0', port=5001, debug=True)
# Also update frontend: src/components/ArticleUpload.tsx
const API_BASE_URL = 'http://localhost:5001';
```

### Frontend Issues

**Problem**: CORS error

```bash
# Solution: Make sure flask-cors is installed
pip install flask-cors
```

**Problem**: Cannot connect to backend

- Verify backend is running on port 5000
- Check browser console for errors
- Ensure API_BASE_URL in ArticleUpload.tsx is correct

## Next Steps for Production

1. **Enhance Feature Extraction**:
   - Load actual LDA model (currently using keyword-based)
   - Load actual TF-IDF vectorizer (currently using simple counting)
   - Add caching for faster inference

2. **Improve Backend**:
   - Add authentication
   - Add rate limiting
   - Add request logging
   - Deploy to cloud (AWS, GCP, Azure)
   - Use gunicorn for production

3. **Improve Frontend**:
   - Add article history
   - Add feature visualization
   - Add SHAP explanations
   - Build production bundle
   - Deploy to Vercel/Netlify

4. **Model Improvements**:
   - Retrain with more data
   - Add ensemble methods
   - Add real-time market data
   - Add sentiment time-series features

## Performance Notes

- First prediction takes ~5-10 seconds (loading FinBERT)
- Subsequent predictions take ~1-2 seconds
- PDF extraction adds ~1 second
- Model loaded once on server startup

## Support

For issues or questions:

1. Check backend logs in terminal
2. Check browser console (F12)
3. Verify all dependencies installed
4. Ensure trained model exists at `Data/results/xgboost_sp500_model.pkl`
