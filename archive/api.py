# api.py
# FastAPI REST endpoint for fake news detection model serving

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uvicorn
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import asyncio
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import redis
import hashlib
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions', ['model', 'result'])
prediction_duration = Histogram('prediction_duration_seconds', 'Time spent processing prediction', ['model'])
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])

# Request/Response Models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="News article text to analyze")
    model: Optional[str] = Field("naive_bayes", description="Model to use for prediction")
    return_confidence: Optional[bool] = Field(True, description="Return confidence scores")
    return_explanation: Optional[bool] = Field(False, description="Return LIME explanation")
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Text must be at least 10 characters long')
        return v
    
    @validator('model')
    def validate_model(cls, v):
        allowed_models = ['naive_bayes', 'random_forest', 'lstm', 'bert', 'ensemble']
        if v not in allowed_models:
            raise ValueError(f'Model must be one of {allowed_models}')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    model_used: str
    processing_time: float
    explanation: Optional[Dict[str, Any]] = None
    timestamp: str

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: Optional[str] = "naive_bayes"

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime: float
    total_predictions: int

class ModelManager:
    """Manages model loading and prediction"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        self.vectorizers = {}
        self.cache = {}  # Simple in-memory cache
        
    async def load_models(self):
        """Load all models asynchronously"""
        logger.info("Loading models...")
        
        try:
            # Load preprocessor
            with open('preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Load TF-IDF vectorizer
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizers['tfidf'] = pickle.load(f)
            
            # Load Naive Bayes
            with open('naive_bayes_model.pkl', 'rb') as f:
                self.models['naive_bayes'] = pickle.load(f)
            
            # Load Random Forest
            with open('random_forest_model.pkl', 'rb') as f:
                self.models['random_forest'] = pickle.load(f)
            
            # Try to load deep learning models
            try:
                from tensorflow.keras.models import load_model
                
                # Load LSTM
                if Path('best_lstm_model.h5').exists():
                    self.models['lstm'] = load_model('best_lstm_model.h5')
                    
                    with open('lstm_tokenizer.pkl', 'rb') as f:
                        self.vectorizers['lstm_tokenizer'] = pickle.load(f)
                
                # Load BERT
                if Path('best_bert_lstm_model.h5').exists():
                    self.models['bert'] = load_model('best_bert_lstm_model.h5')
                    
                    from transformers import BertTokenizer, BertModel
                    self.vectorizers['bert_tokenizer'] = BertTokenizer.from_pretrained('bert-base-uncased')
                    self.models['bert_base'] = BertModel.from_pretrained('bert-base-uncased')
            
            except ImportError:
                logger.warning("TensorFlow not available - deep learning models not loaded")
            
            logger.info(f"Loaded models: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for prediction"""
        if self.preprocessor:
            return self.preprocessor.clean_text(text)
        else:
            # Basic preprocessing if preprocessor not available
            return text.lower().strip()
    
    def get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for prediction"""
        return hashlib.md5(f"{text}:{model}".encode()).hexdigest()
    
    async def predict(self, text: str, model_name: str) -> Dict[str, Any]:
        """Make prediction with specified model"""
        
        # Check cache
        cache_key = self.get_cache_key(text, model_name)
        if cache_key in self.cache:
            logger.info(f"Cache hit for {cache_key[:8]}...")
            return self.cache[cache_key]
        
        start_time = time.time()
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        result = {
            'prediction': None,
            'confidence': None,
            'model_used': model_name
        }
        
        try:
            if model_name == 'naive_bayes' and 'naive_bayes' in self.models:
                # Transform text using TF-IDF
                text_tfidf = self.vectorizers['tfidf'].transform([cleaned_text])
                
                # Make prediction
                prediction = self.models['naive_bayes'].predict(text_tfidf)[0]
                confidence = self.models['naive_bayes'].predict_proba(text_tfidf)[0].max()
                
                result['prediction'] = 'FAKE' if prediction == 1 else 'REAL'
                result['confidence'] = float(confidence)
                
            elif model_name == 'random_forest' and 'random_forest' in self.models:
                # Transform text
                text_tfidf = self.vectorizers['tfidf'].transform([cleaned_text])
                
                # Make prediction
                prediction = self.models['random_forest'].predict(text_tfidf.toarray())[0]
                confidence = self.models['random_forest'].predict_proba(text_tfidf.toarray())[0].max()
                
                result['prediction'] = 'FAKE' if prediction == 1 else 'REAL'
                result['confidence'] = float(confidence)
                
            elif model_name == 'lstm' and 'lstm' in self.models:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                
                # Tokenize and pad
                tokenizer = self.vectorizers['lstm_tokenizer']
                sequence = tokenizer.texts_to_sequences([cleaned_text])
                padded = pad_sequences(sequence, maxlen=100)
                
                # Predict
                pred_prob = self.models['lstm'].predict(padded, verbose=0)[0][0]
                
                result['prediction'] = 'FAKE' if pred_prob > 0.5 else 'REAL'
                result['confidence'] = float(pred_prob if pred_prob > 0.5 else 1 - pred_prob)
                
            elif model_name == 'ensemble':
                # Ensemble prediction (majority voting)
                predictions = []
                confidences = []
                
                for model in ['naive_bayes', 'random_forest']:
                    if model in self.models:
                        pred = await self.predict(text, model)
                        predictions.append(1 if pred['prediction'] == 'FAKE' else 0)
                        confidences.append(pred['confidence'])
                
                if predictions:
                    final_pred = 1 if sum(predictions) > len(predictions) / 2 else 0
                    result['prediction'] = 'FAKE' if final_pred == 1 else 'REAL'
                    result['confidence'] = float(np.mean(confidences))
                    result['model_used'] = 'ensemble'
            
            else:
                raise ValueError(f"Model {model_name} not available")
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
        
        # Record metrics
        duration = time.time() - start_time
        prediction_duration.labels(model=model_name).observe(duration)
        prediction_counter.labels(model=model_name, result=result['prediction']).inc()
        
        # Cache result
        self.cache[cache_key] = result
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries
            self.cache = dict(list(self.cache.items())[-500:])
        
        return result
    
    async def explain_prediction(self, text: str, model_name: str) -> Dict[str, Any]:
        """Generate LIME explanation for prediction"""
        try:
            from lime.lime_text import LimeTextExplainer
            
            explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
            
            # Create prediction function for LIME
            def predict_proba(texts):
                results = []
                for t in texts:
                    if model_name == 'naive_bayes':
                        cleaned = self.preprocess_text(t)
                        tfidf = self.vectorizers['tfidf'].transform([cleaned])
                        proba = self.models['naive_bayes'].predict_proba(tfidf)[0]
                        results.append(proba)
                return np.array(results)
            
            # Generate explanation
            exp = explainer.explain_instance(
                text, 
                predict_proba, 
                num_features=10
            )
            
            return {
                'important_words': exp.as_list(),
                'score': exp.score
            }
            
        except Exception as e:
            logger.error(f"Explanation error: {e}")
            return None

# Create model manager instance
model_manager = ModelManager()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up API server...")
    await model_manager.load_models()
    app.state.start_time = time.time()
    app.state.total_predictions = 0
    yield
    # Shutdown
    logger.info("Shutting down API server...")

# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="REST API for detecting fake news using machine learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    api_requests.labels(endpoint="/", method="GET").inc()
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    api_requests.labels(endpoint="/health", method="GET").inc()
    
    return HealthResponse(
        status="healthy",
        models_loaded=list(model_manager.models.keys()),
        uptime=time.time() - app.state.start_time,
        total_predictions=app.state.total_predictions
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Single text prediction endpoint"""
    api_requests.labels(endpoint="/predict", method="POST").inc()
    
    start_time = time.time()
    
    try:
        # Make prediction
        result = await model_manager.predict(request.text, request.model)
        
        # Add explanation if requested
        explanation = None
        if request.return_explanation:
            explanation = await model_manager.explain_prediction(request.text, request.model)
        
        # Update counter
        app.state.total_predictions += 1
        
        response = PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'] if request.return_confidence else None,
            model_used=result['model_used'],
            processing_time=time.time() - start_time,
            explanation=explanation,
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction in background
        background_tasks.add_task(log_prediction, request.text, result)
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    api_requests.labels(endpoint="/predict/batch", method="POST").inc()
    
    results = []
    
    for text in request.texts:
        try:
            result = await model_manager.predict(text, request.model)
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
        except Exception as e:
            results.append({
                'text': text[:100] + '...',
                'error': str(e)
            })
    
    app.state.total_predictions += len(request.texts)
    
    return {
        'results': results,
        'processed': len(results),
        'model_used': request.model,
        'timestamp': datetime.now().isoformat()
    }

@app.post("/upload", tags=["Prediction"])
async def upload_and_predict(file: UploadFile = File(...)):
    """Upload CSV file for batch prediction"""
    api_requests.labels(endpoint="/upload", method="POST").inc()
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        if 'text' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must have 'text' column")
        
        # Process predictions
        results = []
        for text in df['text'].head(100):  # Limit to 100 for demo
            result = await model_manager.predict(str(text), 'naive_bayes')
            results.append(result)
        
        # Add predictions to dataframe
        df['prediction'] = [r['prediction'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        
        # Save results
        output_path = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        
        return {
            'processed': len(results),
            'output_file': output_path,
            'summary': df['prediction'].value_counts().to_dict()
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", tags=["Models"])
async def list_models():
    """List available models"""
    api_requests.labels(endpoint="/models", method="GET").inc()
    
    models_info = []
    for name, model in model_manager.models.items():
        info = {
            'name': name,
            'type': type(model).__name__,
            'available': True
        }
        models_info.append(info)
    
    return {'models': models_info}

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint"""
    api_requests.labels(endpoint="/metrics", method="GET").inc()
    
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/cache/stats", tags=["Monitoring"])
async def cache_stats():
    """Get cache statistics"""
    api_requests.labels(endpoint="/cache/stats", method="GET").inc()
    
    return {
        'size': len(model_manager.cache),
        'max_size': 1000,
        'hit_rate': 0.0  # Would need to track this properly
    }

@app.delete("/cache/clear", tags=["Monitoring"])
async def clear_cache():
    """Clear prediction cache"""
    api_requests.labels(endpoint="/cache/clear", method="DELETE").inc()
    
    size = len(model_manager.cache)
    model_manager.cache.clear()
    
    return {
        'message': 'Cache cleared',
        'items_removed': size
    }

# Helper functions
async def log_prediction(text: str, result: Dict[str, Any]):
    """Log prediction to file (async)"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_sample': text[:100],
            'prediction': result['prediction'],
            'confidence': result.get('confidence'),
            'model': result.get('model_used')
        }
        
        # Append to log file
        with open('predictions.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    except Exception as e:
        logger.error(f"Logging error: {e}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )