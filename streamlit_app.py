# streamlit_app.py
# Fake News Detection System - Web Interface
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ML/DL Libraries (only import what we actually use)
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
# Note: We don't import torch, transformers, keras since we use our model_utils instead

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main content area with better contrast */
    .main {
        padding: 2rem;
        background-color: #2c2c2c;
        color: #F5F5F0;
    }

    /* Dark mode overrides for better text visibility */
    .stApp {
        background-color: #2c2c2c;
    }

    /* Text areas and inputs with better contrast */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #3c3c3c !important;
        color: #E6D8C3 !important;
        border: 2px solid #C2A68C !important;
        border-radius: 8px !important;
    }

    /* Alert boxes */
    .stAlert {
        background-color: #3c3c3c;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #E6D8C3;
    }

    /* Prediction boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .real-news {
        background-color: #2d4a2d;
        border: 2px solid #38a169;
        color: #90ee90;
    }

    .fake-news {
        background-color: #4a2d2d;
        border: 2px solid #e53e3e;
        color: #ff6b6b;
    }

    /* Better heading contrast */
    h1 {
        color: #E6D8C3 !important;
        text-align: center;
        padding: 1rem 0;
        font-weight: 700;
    }

    h2, h3 {
        color: #C2A68C !important;
        border-bottom: 2px solid #C2A68C;
        padding-bottom: 0.5rem;
    }

    /* Sidebar styling with better text contrast */
    .css-1d391kg {
        background-color: #3c3c3c;
    }

    /* Sidebar text elements */
    .sidebar .stMarkdown, .sidebar .stMarkdown p, .sidebar .stMarkdown div,
    .css-1d391kg .stMarkdown, .css-1d391kg .stMarkdown p, .css-1d391kg .stMarkdown div,
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span, .css-1d391kg label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #C2A68C !important;
        font-weight: 500 !important;
    }
    /* Additional sidebar text styling for specific data-testid */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stText"],
    [data-testid="stSidebar"] [data-testid="element-container"] {
      color: #44444E !important;
    }
    /* Sidebar metrics and cards */
    [data-testid="stSidebar"] .metric-value, [data-testid="stSidebar"] .metric-label {
        color: #C2A68C !important;
        font-weight: 600 !important;
    }

    /* Content text for better readability - Force dark text */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    .stText, p, div, span, label, .stRadio > div, .stSelectbox > div,
    .element-container, .block-container, [data-testid="stMarkdownContainer"] {
        color: #C2A68C !important;
        font-weight: 500 !important;
    }

    /* Force dark text on all Streamlit elements */
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label {
        color: #E6D8C3 !important;
    }

    /* Tab labels and other text elements */
    .stTabs [data-baseweb="tab"] {
        color: #E6D8C3 !important;
        font-weight: 600 !important;
    }

    /* Radio button and selectbox text */
    .stRadio > div > div > div > label,
    .stSelectbox > div > div > div {
        color: #1a202c !important;
        font-weight: 500 !important;
    }

    /* Metric styling */
    .metric-card {
        background-color: #3c3c3c;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid #C2A68C;
        color: #E6D8C3;
    }

    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 3rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .footer h3 {
        color: white !important;
        border-bottom: 2px solid rgba(255, 255, 255, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    .footer p, .footer div {
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 0.5rem 0;
    }

    .team-member {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }

    /* Better button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Text preprocessing class
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

# Model loading functions
@st.cache_resource
def load_models():
    """Load pre-trained models and preprocessors"""
    models = {}
    preprocessor = TextPreprocessor()

    try:
        # Use our actual model loader
        from model_utils import get_model_loader
        loader = get_model_loader()

        if loader and loader.get_available_models():
            models['model_loader'] = loader
            st.success(f"✅ Loaded models: {', '.join(loader.get_available_models())}")
        else:
            st.warning("⚠️ No trained models found. Please run the training script first.")

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")

    return models, preprocessor

# Import our working model utilities
from model_utils import get_model_loader

# Check if transformer is available
def is_transformer_available():
    """Check if DistilBERT model is available."""
    try:
        from transformer_model import TransformerClassifier, is_transformer_available as check_transformer
        if not check_transformer():
            return False
        # Check if fine-tuned model exists
        model_path = Path('models/distilbert')
        return model_path.exists() and (model_path / 'config.json').exists()
    except ImportError:
        return False

# Cache transformer model loading
@st.cache_resource
def load_transformer_model():
    """Load transformer model (cached)."""
    try:
        from transformer_model import TransformerClassifier
        classifier = TransformerClassifier(model_path='models/distilbert')
        if classifier.load_model():
            return classifier
    except Exception as e:
        logger.warning(f"Could not load transformer: {e}")
    return None

# Import Path for file checks
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

# Import Gemini AI for enhanced explanations
try:
    from gemini_utils import is_gemini_available, get_gemini_explainer
    GEMINI_AVAILABLE = is_gemini_available()
except ImportError:
    GEMINI_AVAILABLE = False

# Prediction function
def make_prediction(text, models, preprocessor, model_choice):
    """Make prediction using selected model"""

    try:
        # Check if transformer model is selected
        if "DistilBERT" in model_choice or "Transformer" in model_choice:
            transformer = load_transformer_model()
            if transformer is None:
                st.warning("DistilBERT not available. Train it first with: `python train_transformer.py`")
                return None, 0.5, "Transformer model not available"

            with st.spinner("Running DistilBERT inference..."):
                result = transformer.predict(text)

            prediction = 1 if result['prediction'] == 'FAKE' else 0
            confidence = result['confidence']
            explanation = f"Prediction made using DistilBERT transformer with {confidence:.1%} confidence"

            return prediction, confidence, explanation

        # Use our working model_utils for sklearn models
        loader = get_model_loader()

        if not loader or not loader.get_available_models():
            return None, 0.5, "No trained models available"

        # Map UI choices to our actual models
        model_map = {
            "Naive Bayes (94.4%)": "naive_bayes",
            "Random Forest (98.0%)": "random_forest",
            "Logistic Regression (97.4%)": "logistic",
            # Legacy mappings for backwards compatibility
            "Naive Bayes": "naive_bayes",
            "Random Forest": "random_forest",
            "Logistic Regression": "logistic"
        }

        actual_model = model_map.get(model_choice, "logistic")

        # Make prediction using our working model
        result = loader.predict(text, actual_model)

        # Convert to expected format
        prediction = 1 if result['prediction'] == 'FAKE' else 0
        confidence = result['confidence']
        explanation = f"Prediction made using {actual_model} model with {confidence:.1%} confidence"

        return prediction, confidence, explanation

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, 0.5, f"Error: {str(e)}"

# LIME Explainability function
def render_lime_explanation(text, model_name, prediction):
    """
    Render LIME explainability visualization in Streamlit.

    Args:
        text: The input text that was analyzed
        model_name: Internal model name (logistic, naive_bayes, random_forest)
        prediction: The prediction result (0 for REAL, 1 for FAKE)
    """
    with st.expander("🔍 Why this prediction? (AI Explanation)", expanded=False):
        try:
            # Check if LIME is available
            try:
                import lime
            except ImportError:
                st.warning("LIME not installed. Run: `pip install lime`")
                st.info("Install LIME to see word-level explanations for predictions.")
                return

            loader = get_model_loader()

            with st.spinner("Generating explanation (this may take a few seconds)..."):
                # Get explainer and generate explanation
                explainer = loader.get_explainer(model_name)
                explanation = explainer.explain(text, num_features=10, num_samples=300)

                # Create tabs for different visualization views
                tab1, tab2, tab3 = st.tabs(["📊 Word Impact", "📝 Highlighted Text", "📋 Details"])

                with tab1:
                    # Get visualization data
                    viz_data = explainer.get_word_importance_data(explanation)

                    # Create horizontal bar chart
                    fig = go.Figure(go.Bar(
                        x=viz_data['weights'],
                        y=viz_data['words'],
                        orientation='h',
                        marker_color=viz_data['colors'],
                        text=[f"{w:.3f}" for w in viz_data['weights']],
                        textposition='outside'
                    ))

                    fig.update_layout(
                        title="Word Impact on Prediction",
                        xaxis_title="Impact (+ = FAKE, - = REAL)",
                        yaxis_title="Words",
                        height=400,
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#E6D8C3')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""
                    **How to read this chart:**
                    - 🔴 **Red bars (positive)**: Words that pushed the prediction toward **FAKE**
                    - 🟢 **Green bars (negative)**: Words that pushed the prediction toward **REAL**
                    - **Longer bars** = stronger influence on the prediction
                    """)

                with tab2:
                    st.markdown("### Text with Key Words Highlighted")

                    # Get highlighted HTML
                    highlighted_html = explainer.get_highlighted_html(text, explanation)

                    st.markdown(f"""
                    <div style="background-color: #3c3c3c; padding: 20px; border-radius: 10px;
                                line-height: 1.8; color: #E6D8C3; max-height: 400px; overflow-y: auto;">
                    {highlighted_html}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    **Legend:**
                    - <span style="background-color: rgba(255,107,107,0.6); padding: 2px 6px; border-radius: 3px;">Red highlight</span> = Indicates **FAKE** news characteristics
                    - <span style="background-color: rgba(81,207,102,0.6); padding: 2px 6px; border-radius: 3px;">Green highlight</span> = Indicates **REAL** news characteristics
                    """, unsafe_allow_html=True)

                with tab3:
                    st.markdown("### Explanation Details")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Fidelity Score", f"{explanation['score']:.3f}")
                        st.caption("How well LIME approximates the model locally (higher = better)")
                    with col2:
                        st.metric("Features Analyzed", len(explanation['words']))
                        st.caption("Number of words considered in explanation")

                    st.markdown("---")
                    st.markdown("**Prediction Probabilities:**")
                    prob_col1, prob_col2 = st.columns(2)
                    with prob_col1:
                        st.metric("REAL Probability", f"{explanation['prediction_proba']['REAL']:.1%}")
                    with prob_col2:
                        st.metric("FAKE Probability", f"{explanation['prediction_proba']['FAKE']:.1%}")

                    st.markdown("---")
                    st.markdown("**Top 5 Most Influential Words:**")
                    for i, (word, weight) in enumerate(zip(explanation['words'][:5], explanation['weights'][:5])):
                        direction = "FAKE" if weight > 0 else "REAL"
                        color = "#ff6b6b" if weight > 0 else "#51cf66"
                        st.markdown(
                            f"{i+1}. **{word}**: <span style='color: {color}'>{weight:.4f}</span> "
                            f"(pushes toward {direction})",
                            unsafe_allow_html=True
                        )

        except ImportError as e:
            st.warning(f"LIME library not available: {e}")
            st.info("Install LIME to enable explainability: `pip install lime`")
        except Exception as e:
            st.error(f"Could not generate explanation: {str(e)}")
            st.info("Try using Logistic Regression or Naive Bayes for best explainability results.")


# Gemini AI Explanation function
def render_gemini_explanation(text, prediction, confidence):
    """
    Render Gemini AI-powered explanation for the prediction.

    Args:
        text: The input text that was analyzed
        prediction: The prediction result (0 for REAL, 1 for FAKE)
        confidence: The confidence score
    """
    if not GEMINI_AVAILABLE:
        return

    with st.expander("🤖 AI-Powered Analysis (Gemini)", expanded=True):
        try:
            explainer = get_gemini_explainer()

            if explainer is None:
                st.warning("Gemini AI not available. Install with: `pip install google-generativeai`")
                return

            prediction_label = "FAKE" if prediction == 1 else "REAL"

            with st.spinner("Generating AI analysis... This may take a few seconds."):
                result = explainer.generate_explanation(
                    text=text,
                    prediction=prediction_label,
                    confidence=confidence,
                    key_words=None
                )

            if result.get("success"):
                # Display the AI explanation with markdown formatting
                st.markdown(result["explanation"])

                # Add a note about AI limitations
                st.markdown("---")
                st.caption("*This analysis is generated by Gemini AI to help explain the prediction. "
                          "Always verify important information through multiple reliable sources.*")
            else:
                error_msg = result.get("error", "Unknown error")
                st.error(f"Could not generate AI explanation: {error_msg}")

        except Exception as e:
            st.error(f"Error with Gemini AI: {str(e)}")
            st.info("The AI explanation feature requires the google-generativeai package.")


# Fetch news from API (placeholder function)
def fetch_news_from_api(query, api_key=None):
    """Fetch news articles from NewsAPI"""
    if not api_key:
        return []
    
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize=5"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if data['status'] == 'ok':
            articles = []
            for article in data['articles']:
                articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'url': article['url'],
                    'source': article['source']['name']
                })
            return articles
    except:
        return []
    
    return []

# Web scraping function
def scrape_article_from_url(url):
    """
    Scrape article content from a given URL
    Returns: dict with title, text, and metadata or None if failed
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return {"error": "Invalid URL format. Please include http:// or https://"}

        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        # Make request with timeout
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'menu']):
            element.decompose()

        # Try to extract title
        title = None
        title_selectors = ['h1', 'title', '.article-title', '.headline', '.entry-title']
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text(strip=True):
                title = title_elem.get_text(strip=True)
                break

        # Try to extract main content using common selectors
        content_selectors = [
            'article', '.article-content', '.post-content', '.entry-content',
            '.story-body', '.article-body', '.content', 'main', '.main-content',
            '.post-body', '.text-content', '.story-text'
        ]

        article_text = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Get all paragraph text
                paragraphs = content_elem.find_all(['p', 'div'], recursive=True)
                if paragraphs:
                    texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                    if texts:
                        article_text = ' '.join(texts)
                        break

        # Fallback: extract all paragraph text if no specific content found
        if not article_text:
            paragraphs = soup.find_all('p')
            if paragraphs:
                texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
                article_text = ' '.join(texts)

        # Final fallback: get all text content
        if not article_text:
            article_text = soup.get_text()

        # Clean up the text
        article_text = re.sub(r'\s+', ' ', article_text).strip()

        # Validate extracted content
        if len(article_text) < 100:
            return {"error": "Could not extract sufficient article content. The page might be protected or have unusual formatting."}

        # Get domain for source info
        source_domain = parsed_url.netloc

        return {
            "title": title or "Article Title Not Found",
            "text": article_text,
            "source": source_domain,
            "url": url,
            "word_count": len(article_text.split()),
            "char_count": len(article_text)
        }

    except requests.RequestException as e:
        return {"error": f"Failed to access URL: {str(e)}"}
    except Exception as e:
        return {"error": f"Error parsing article: {str(e)}"}

# Main app
def main():
    st.markdown("<h1>🔍 Fake News Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #C2A68C;'>AI-powered system to detect misinformation</p>", unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.model_loaded:
        with st.spinner("Loading models..."):
            st.session_state.models, st.session_state.preprocessor = load_models()
            st.session_state.model_loaded = True
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model selection - get available models dynamically
        loader = get_model_loader()
        if loader and loader.get_available_models():
            # Map internal model names to user-friendly names
            model_display_names = {
                "logistic": "Logistic Regression (97.4%)",
                "naive_bayes": "Naive Bayes (94.4%)",
                "random_forest": "Random Forest (98.0%)"
            }
            available_models = [model_display_names.get(model, model.title())
                              for model in loader.get_available_models()]

            # Add DistilBERT if available
            if is_transformer_available():
                available_models.append("DistilBERT Transformer (Deep Learning)")
                st.success("🤖 DistilBERT model available!")
        else:
            available_models = ["No models available"]

        # Show model type info
        st.markdown("**Model Types:**")
        st.markdown("- 🔹 *Traditional ML*: Fast, interpretable")
        if is_transformer_available():
            st.markdown("- 🔸 *Transformer*: Higher accuracy, slower")
        else:
            st.info("💡 Train DistilBERT for better accuracy:\n`python train_transformer.py`")

        model_choice = st.selectbox(
            "Select Model",
            available_models,
            help="Choose between different trained models with varying accuracy"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.5, 1.0, 0.7,
            help="Minimum confidence level for predictions"
        )
        
        st.markdown("---")
        
        # NewsAPI settings
        st.markdown("## 📰 News API Settings")

        # Try to get API key from secrets or env
        default_api_key = ""
        try:
            # Try Streamlit secrets first (for cloud deployment)
            default_api_key = st.secrets.get("NEWS_API_KEY", "")
        except:
            # Fallback to environment variable (for local development)
            try:
                import os
                from dotenv import load_dotenv
                load_dotenv()
                default_api_key = os.getenv("NEWS_API_KEY", "")
            except:
                pass

        if default_api_key:
            st.success("✅ NewsAPI key loaded automatically")
            api_key = default_api_key
            st.markdown("*Using configured API key for news fetching*")
        else:
            api_key = st.text_input(
                "NewsAPI Key",
                type="password",
                help="Enter your NewsAPI key to fetch real-time news"
            )
        
        if st.button("Fetch Latest News"):
            if api_key:
                with st.spinner("Fetching news..."):
                    articles = fetch_news_from_api("latest", api_key)
                    if articles:
                        st.success(f"Fetched {len(articles)} articles")
                        st.session_state.fetched_articles = articles
            else:
                st.error("Please enter an API key")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("## 📊 Session Statistics")
        st.metric("Total Predictions", len(st.session_state.history))
        
        if st.session_state.history:
            fake_count = sum(1 for h in st.session_state.history if h['prediction'] == 'Fake')
            real_count = len(st.session_state.history) - fake_count
            st.metric("Fake News Detected", fake_count)
            st.metric("Real News Detected", real_count)
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Text Analysis", "📰 URL Analysis", "🖼️ Deepfake Detection", "📊 Analytics", "❓ About"])
    
    with tab1:
        st.markdown("### Enter News Article Text")
        
        # Text input methods
        input_method = st.radio(
            "Input Method",
            ["Type/Paste Text", "Upload File", "Use Example"],
            horizontal=True
        )
        
        text_input = ""
        
        if input_method == "Type/Paste Text":
            text_input = st.text_area(
                "Article Text",
                height=200,
                placeholder="Paste your news article here..."
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a text file",
                type=['txt', 'csv']
            )
            if uploaded_file:
                text_input = str(uploaded_file.read(), 'utf-8')
                st.text_area("Uploaded Content", text_input, height=200, disabled=True)
        
        else:  # Use Example
            examples = {
                "Real News Example": """The Senate passed a bipartisan infrastructure bill on Tuesday with a vote of 69-30, marking a significant legislative victory for President Biden. The $1.2 trillion package includes funding for roads, bridges, broadband internet, and electric vehicle charging stations. Senator Rob Portman, a Republican from Ohio who helped negotiate the deal, said the bill represents a historic investment in American infrastructure. The legislation now moves to the House of Representatives for consideration. Congressional Budget Office estimates suggest the bill would add approximately $256 billion to the federal deficit over the next decade. Transportation Secretary Pete Buttigieg praised the bipartisan effort, calling it a model for future cooperation.""",

                "Fake News Example": "BREAKING: Scientists discover that drinking lemon water every morning can cure cancer completely! Doctors are shocked and pharmaceutical companies are trying to hide this information from the public. Share this before it gets deleted! One weird trick that Big Pharma doesn't want you to know about. Thousands have already been cured using this simple method that costs less than $1. The government doesn't want you to know this secret!"
            }
            
            selected_example = st.selectbox("Select Example", list(examples.keys()))
            text_input = examples[selected_example]
            st.text_area("Example Text", text_input, height=200, disabled=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            analyze_button = st.button(
                "🔍 Analyze Text",
                use_container_width=True,
                type="primary"
            )
        
        if analyze_button and text_input:
            with st.spinner("Analyzing..."):
                # Make prediction
                prediction, confidence, explanation = make_prediction(
                    text_input,
                    st.session_state.models,
                    st.session_state.preprocessor,
                    model_choice
                )
                
                if prediction is not None:
                    # Store in history
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'text': text_input[:100] + "...",
                        'prediction': 'Fake' if prediction == 1 else 'Real',
                        'confidence': confidence,
                        'model': model_choice
                    })
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        if prediction == 1:  # Fake news
                            st.markdown(
                                f"""
                                <div class='prediction-box fake-news'>
                                    <h2 style='color: #ff6b6b;'>⚠️ FAKE NEWS DETECTED</h2>
                                    <h3>Confidence: {confidence:.1%}</h3>
                                    <p>This article appears to contain misinformation.</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:  # Real news
                            st.markdown(
                                f"""
                                <div class='prediction-box real-news'>
                                    <h2 style='color: #90ee90;'>✅ REAL NEWS</h2>
                                    <h3>Confidence: {confidence:.1%}</h3>
                                    <p>This article appears to be legitimate.</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Confidence meter
                    st.markdown("### Confidence Meter")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence * 100,
                        title={'text': f"Model: {model_choice}"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': confidence_threshold * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # LIME Explainability section
                    st.markdown("### 🔍 Explanation")

                    # Map UI model choice to internal model name
                    model_map = {
                        "Naive Bayes (94.4%)": "naive_bayes",
                        "Random Forest (98.0%)": "random_forest",
                        "Logistic Regression (97.4%)": "logistic",
                        "Naive Bayes": "naive_bayes",
                        "Random Forest": "random_forest",
                        "Logistic Regression": "logistic"
                    }
                    internal_model_name = model_map.get(model_choice, "logistic")

                    # Render LIME explanation
                    render_lime_explanation(text_input, internal_model_name, prediction)

                    # Render Gemini AI explanation
                    render_gemini_explanation(text_input, prediction, confidence)

                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    
                    if prediction == 1:  # Fake news
                        st.warning("""
                        **What to do next:**
                        - Verify information from multiple reliable sources
                        - Check the publication date and author credentials
                        - Look for citations and references
                        - Be cautious about sharing this article
                        - Report misinformation to platform moderators
                        """)
                    else:  # Real news
                        st.success("""
                        **This appears to be legitimate news, but always:**
                        - Cross-reference with other reputable sources
                        - Check for recent updates on the topic
                        - Consider the source's reliability and bias
                        - Read beyond headlines for full context
                        """)
                else:
                    st.error("Could not analyze the text. Please try again.")
        
        elif analyze_button:
            st.warning("Please enter some text to analyze")
    
    with tab2:
        st.markdown("### Analyze News from URL")

        url_input = st.text_input(
            "Enter News Article URL",
            placeholder="https://example.com/news-article"
        )

        # URL validation helper
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**Supported sites:** Most news websites (BBC, CNN, Reuters, Guardian, etc.)")
        with col2:
            st.markdown("**Max time:** 15 seconds")

        if st.button("🔍 Analyze URL", use_container_width=True, type="primary"):
            if url_input:
                with st.spinner("🌐 Fetching article content..."):
                    # Scrape the article
                    scraped_data = scrape_article_from_url(url_input)

                    if "error" in scraped_data:
                        st.error(f"❌ {scraped_data['error']}")
                        st.markdown("""
                        **Troubleshooting tips:**
                        - Ensure the URL is complete and accessible
                        - Some sites may block automated access
                        - Try a different news article URL
                        - Check if the site requires JavaScript (not supported)
                        """)
                    else:
                        # Display article info
                        st.success("✅ Article successfully extracted!")

                        # Show article metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📝 Word Count", f"{scraped_data['word_count']:,}")
                        with col2:
                            st.metric("📰 Source", scraped_data['source'])
                        with col3:
                            st.metric("🔗 Characters", f"{scraped_data['char_count']:,}")

                        # Display article title and preview
                        st.markdown("**📰 Article Title:**")
                        st.markdown(f"*{scraped_data['title']}*")

                        # Show article preview
                        st.markdown("**📄 Article Preview:**")
                        preview_text = scraped_data['text'][:500] + "..." if len(scraped_data['text']) > 500 else scraped_data['text']
                        st.text_area("", preview_text, height=150, disabled=True)

                        # Analyze the scraped content
                        with st.spinner("🤖 Analyzing article for fake news..."):
                            prediction, confidence, explanation = make_prediction(
                                scraped_data['text'],
                                st.session_state.models,
                                st.session_state.preprocessor,
                                model_choice
                            )

                            if prediction is not None:
                                # Store in history with URL source
                                st.session_state.history.append({
                                    'timestamp': datetime.now(),
                                    'text': f"URL: {scraped_data['title'][:50]}...",
                                    'prediction': 'Fake' if prediction == 1 else 'Real',
                                    'confidence': confidence,
                                    'model': model_choice,
                                    'source': 'URL',
                                    'url': url_input
                                })

                                # Display results
                                st.markdown("---")
                                st.markdown("### 📊 Prediction Results")

                                col1, col2, col3 = st.columns([1, 2, 1])

                                with col2:
                                    if prediction == 1:  # Fake news
                                        st.markdown(
                                            f"""
                                            <div class='prediction-box fake-news'>
                                                <h2 style='color: #ff6b6b;'>⚠️ FAKE NEWS DETECTED</h2>
                                                <h3>Confidence: {confidence:.1%}</h3>
                                                <p>This article appears to contain misinformation.</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    else:  # Real news
                                        st.markdown(
                                            f"""
                                            <div class='prediction-box real-news'>
                                                <h2 style='color: #90ee90;'>✅ REAL NEWS</h2>
                                                <h3>Confidence: {confidence:.1%}</h3>
                                                <p>This article appears to be legitimate.</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                # Confidence meter
                                st.markdown("### Confidence Meter")
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=confidence * 100,
                                    title={'text': f"Model: {model_choice}"},
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    gauge={
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgray"},
                                            {'range': [50, 80], 'color': "yellow"},
                                            {'range': [80, 100], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': confidence_threshold * 100
                                        }
                                    }
                                ))
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)

                                # LIME Explainability section for URL analysis
                                st.markdown("### 🔍 AI Explanation")

                                # Map UI model choice to internal model name
                                url_model_map = {
                                    "Naive Bayes (94.4%)": "naive_bayes",
                                    "Random Forest (98.0%)": "random_forest",
                                    "Logistic Regression (97.4%)": "logistic",
                                    "Naive Bayes": "naive_bayes",
                                    "Random Forest": "random_forest",
                                    "Logistic Regression": "logistic"
                                }
                                url_internal_model = url_model_map.get(model_choice, "logistic")

                                # Render LIME explanation for URL content
                                render_lime_explanation(scraped_data['text'], url_internal_model, prediction)

                                # Render Gemini AI explanation for URL content
                                render_gemini_explanation(scraped_data['text'], prediction, confidence)

                                # Source analysis
                                st.markdown("### 📰 Source Analysis")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Source Domain:** `{scraped_data['source']}`")
                                    st.markdown(f"**Article Length:** {scraped_data['word_count']} words")
                                with col2:
                                    st.markdown(f"**Original URL:** [Visit Article]({url_input})")
                                    st.markdown(f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                                # Recommendations based on prediction
                                st.markdown("### 💡 Recommendations")

                                if prediction == 1:  # Fake news
                                    st.warning(f"""
                                    **⚠️ This article from {scraped_data['source']} appears to be fake news:**
                                    - Cross-check this information with multiple reliable sources
                                    - Look for the original source of the claims
                                    - Check the publication date and author credentials
                                    - Be cautious about sharing this article
                                    - Consider fact-checking websites like Snopes, FactCheck.org
                                    """)
                                else:  # Real news
                                    st.success(f"""
                                    **✅ This article from {scraped_data['source']} appears legitimate:**
                                    - Still recommended to cross-reference with other sources
                                    - Check for recent updates on this topic
                                    - Consider the source's overall reliability and potential bias
                                    - Read the full article for complete context
                                    """)
                            else:
                                st.error("Could not analyze the extracted text. Please try again.")
            else:
                st.warning("Please enter a URL to analyze")

    # NEW TAB: Deepfake Detection
    with tab3:
        st.markdown("### 🖼️ Deepfake Face Detection")
        st.markdown("""
        Detect AI-generated or manipulated human faces using a Vision Transformer model.
        Upload an image to check if it's a real photograph or a deepfake.
        """)

        # Check if detector is available
        try:
            from src.image_detection import is_detector_available, get_image_detector
            detector_available = is_detector_available()
        except ImportError:
            detector_available = False

        if not detector_available:
            st.warning("⚠️ Deepfake detection requires PyTorch and Transformers. Install with: `pip install torch transformers`")
        else:
            # Image input method
            image_input_method = st.radio(
                "Input Method",
                ["Upload Image", "Use Example"],
                horizontal=True,
                key="image_input_method"
            )

            uploaded_image = None
            image_to_analyze = None

            if image_input_method == "Upload Image":
                uploaded_image = st.file_uploader(
                    "Choose an image file",
                    type=['jpg', 'jpeg', 'png', 'webp'],
                    help="Supported formats: JPG, JPEG, PNG, WEBP"
                )

                if uploaded_image:
                    from PIL import Image
                    image_to_analyze = Image.open(uploaded_image)

            else:  # Use Example
                st.markdown("**Example Images:**")

                # Map example names to actual files
                example_images = {
                    "Real Face 1": "sample_images/real_face_1.jpg",
                    "Real Face 2": "sample_images/real_face_2.jpg",
                    "Real Face 3": "sample_images/real_face_3.jpg",
                    "AI Generated Face 1": "sample_images/ai_generated_1.jpg",
                    "AI Generated Face 2": "sample_images/ai_generated_2.jpg",
                }

                example_choice = st.selectbox(
                    "Select an example",
                    list(example_images.keys()),
                    key="deepfake_example"
                )

                # Load the selected example image
                import os
                from PIL import Image
                example_path = example_images[example_choice]
                if os.path.exists(example_path):
                    image_to_analyze = Image.open(example_path)
                    st.success(f"✅ Loaded: {example_choice}")
                else:
                    st.warning(f"⚠️ Image not found: {example_path}")
                    st.info("*For demo: Try uploading images from thispersondoesnotexist.com*")

            # Display uploaded image and analyze
            if image_to_analyze:
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.image(image_to_analyze, caption="Uploaded Image", use_container_width=True)

                # Analyze button
                if st.button("🔍 Detect Deepfake", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image with Vision Transformer..."):
                        try:
                            detector = get_image_detector()

                            if detector is None:
                                st.error("Failed to load deepfake detector model.")
                            else:
                                result = detector.predict(image_to_analyze)

                                # Display result
                                with col2:
                                    if result['prediction'] == 'DEEPFAKE':
                                        st.markdown(
                                            f"""
                                            <div class='prediction-box fake-news'>
                                                <h2 style='color: #ff6b6b;'>🤖 DEEPFAKE DETECTED</h2>
                                                <h3>Confidence: {result['confidence']:.1%}</h3>
                                                <p>This face appears to be AI-generated or manipulated.</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                    else:
                                        st.markdown(
                                            f"""
                                            <div class='prediction-box real-news'>
                                                <h2 style='color: #90ee90;'>📷 REAL FACE</h2>
                                                <h3>Confidence: {result['confidence']:.1%}</h3>
                                                <p>This appears to be a genuine photograph.</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                # Confidence gauge
                                st.markdown("### Confidence Meter")
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=result['confidence'] * 100,
                                    title={'text': "Detection Confidence"},
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    gauge={
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "lightgray"},
                                            {'range': [50, 80], 'color': "yellow"},
                                            {'range': [80, 100], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 70
                                        }
                                    }
                                ))
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)

                                # Probability breakdown
                                st.markdown("### Probability Breakdown")
                                prob_col1, prob_col2 = st.columns(2)
                                with prob_col1:
                                    st.metric("Real Probability", f"{result['probability']['REAL']:.1%}")
                                with prob_col2:
                                    st.metric("Deepfake Probability", f"{result['probability']['DEEPFAKE']:.1%}")

                                # Recommendations
                                st.markdown("### 💡 Recommendations")
                                if result['prediction'] == 'DEEPFAKE':
                                    st.warning("""
                                    **⚠️ This image may be artificially generated:**
                                    - Be cautious about trusting this image
                                    - Look for other verification sources
                                    - Check for inconsistencies in lighting, hair, or background
                                    - Reverse image search to find the original
                                    - Report suspected deepfakes on social media platforms
                                    """)
                                else:
                                    st.success("""
                                    **✅ This appears to be a genuine photograph:**
                                    - The image shows characteristics of real photography
                                    - Still verify important images from multiple sources
                                    - No AI generation patterns detected
                                    """)

                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")
                            st.info("Make sure PyTorch and Transformers are installed correctly.")

    with tab4:
        st.markdown("### 📊 Analytics Dashboard")
        
        if st.session_state.history:
            # Create DataFrame from history
            df_history = pd.DataFrame(st.session_state.history)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Analyzed",
                    len(df_history),
                    delta=f"+{len(df_history)}" if len(df_history) > 0 else "0"
                )
            
            with col2:
                fake_percentage = (df_history['prediction'] == 'Fake').mean() * 100
                st.metric(
                    "Fake News %",
                    f"{fake_percentage:.1f}%",
                    delta=f"{fake_percentage-50:.1f}%" if fake_percentage != 50 else "0%"
                )
            
            with col3:
                avg_confidence = df_history['confidence'].mean() * 100
                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence:.1f}%"
                )
            
            with col4:
                most_used_model = df_history['model'].mode()[0] if not df_history.empty else "N/A"
                st.metric(
                    "Top Model",
                    most_used_model
                )
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of predictions
                fig = px.pie(
                    values=df_history['prediction'].value_counts().values,
                    names=df_history['prediction'].value_counts().index,
                    title="Prediction Distribution",
                    color_discrete_map={'Fake': '#dc3545', 'Real': '#28a745'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Model usage chart
                fig = px.bar(
                    x=df_history['model'].value_counts().index,
                    y=df_history['model'].value_counts().values,
                    title="Model Usage",
                    labels={'x': 'Model', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence distribution
            fig = px.histogram(
                df_history,
                x='confidence',
                nbins=20,
                title="Confidence Distribution",
                labels={'confidence': 'Confidence Level', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # History table
            st.markdown("### Recent Predictions")
            
            # Format the dataframe for display
            df_display = df_history.copy()
            df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.1%}")
            df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display with custom styling
            st.dataframe(
                df_display[['timestamp', 'text', 'prediction', 'confidence', 'model']].tail(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            if st.button("📥 Export History as CSV"):
                csv = df_history.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"fake_news_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No predictions yet. Analyze some articles to see analytics.")
    
    with tab5:
        st.markdown("### About This System")
        
        st.markdown("""
        #### 🎯 **Purpose**
        This Fake News Detection System is developed as part of the ITM-360 Artificial Intelligence course 
        at American University of Phnom Penh. It aims to combat misinformation by using advanced AI techniques
        to identify potentially fake news articles.
        
        #### 🧠 **How It Works**
        The system uses multiple machine learning models:
        
        1. **Naive Bayes**: A probabilistic classifier based on word frequencies
        2. **Random Forest**: An ensemble method using multiple decision trees
        3. **LSTM**: A deep learning model that understands sequence and context
        4. **BERT**: State-of-the-art transformer model for language understanding
        
        #### 📊 **Features**
        - **Multi-model Support**: Choose from different AI models
        - **Explainable AI**: Understand why a prediction was made (LIME)
        - **Real-time Analysis**: Instant predictions on text or URLs
        - **Deepfake Detection**: Detect AI-generated faces using Vision Transformer
        - **Analytics Dashboard**: Track and visualize prediction history
        - **Confidence Scoring**: See how certain the model is about its prediction
        
        #### 👥 **Team Members**
        - **Lim Petnikola** (Team Leader) - Model Development & Integration
        - **Rim Sovichey** - Documentation & Testing
        - **Pha Lyheng** - Data Preprocessing & Feature Engineering
        - **Sokhonn Raksmeidaravid** - UI/UX & Deployment
        
        #### ⚠️ **Disclaimer**
        This system is designed to assist in identifying potential misinformation but should not be the sole
        basis for determining the veracity of news. Always verify important information through multiple
        reliable sources.
        
        #### 📚 **References**
        - Shu et al. (2017) - Fake News Detection on Social Media
        - Ahmed et al. (2018) - Detecting opinion spams and fake news
        - Devlin et al. (2019) - BERT: Pre-training of deep bidirectional transformers
        """)
        
        # Model performance comparison
        st.markdown("### 🏆 Model Performance Comparison")
        
        performance_data = {
            'Model': ['Logistic Regression'],
            'Accuracy': [0.974],
            'Precision': [0.97],
            'Recall': [0.98],
            'F1-Score': [0.97]
        }
        
        df_performance = pd.DataFrame(performance_data)
        
        # Create grouped bar chart
        fig = px.bar(
            df_performance.melt(id_vars='Model', var_name='Metric', value_name='Score'),
            x='Model',
            y='Score',
            color='Metric',
            title='Model Performance Metrics',
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical details expander
        with st.expander("📖 Technical Implementation Details"):
            st.markdown("""
            **Data Processing Pipeline:**
            1. Text cleaning (lowercase, remove URLs, punctuation, stopwords)
            2. Lemmatization using WordNet
            3. Feature extraction (TF-IDF, Word2Vec, BERT embeddings)
            4. Model training with cross-validation
            5. Hyperparameter optimization using GridSearchCV
            
            **Dataset:**
            - Source: Kaggle Fake News Dataset (~40,000 articles)
            - Additional data from NewsAPI for real-time testing
            - 80-10-10 train-validation-test split
            
            **Technologies Used:**
            - Python 3.8+
            - Scikit-learn for machine learning
            - NLTK for text processing
            - Streamlit for web interface
            - BeautifulSoup for web scraping
            """)

    # Team Footer - appears on all pages
    st.markdown("---")
    st.markdown("### 🎓 Development Team")

    # Team members in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Lim Petnikola** (Team Leader)  \nModel Development & Integration")
        st.markdown("**Pha Lyheng**  \nData Preprocessing & Feature Engineering")
    with col2:
        st.markdown("**Rim Sovichey**  \nDocumentation & Testing")
        st.markdown("**Sokhonn Raksmeidaravid**  \nUI/UX & Deployment")

    st.markdown("---")
    st.markdown("**👨‍🏫 Course Instructor:** Professor Kunthea PIN")
    st.markdown("🏛️ American University of Phnom Penh  \n📚 ITM-360 Artificial Intelligence  \n📅 Academic Year 2024-2025")

if __name__ == "__main__":
    main()