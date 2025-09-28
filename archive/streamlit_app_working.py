# streamlit_app_simple.py
# Simplified Fake News Detection System - Web Interface
# Run with: streamlit run streamlit_app_simple.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Import model utilities
from model_utils import get_model_loader, predict_text, get_available_models, get_model_info

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = None

# Try to load saved models on startup
@st.cache_resource
def load_saved_models():
    """Load saved models if available"""
    try:
        loader = get_model_loader()
        if loader.get_available_models():
            return loader
    except Exception as e:
        st.warning(f"No saved models found: {e}")
    return None

# Load models
saved_models = load_saved_models()
if saved_models and saved_models.get_available_models():
    st.session_state.model_trained = True
    st.session_state.model_loader = saved_models

def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

def load_dataset():
    """Load the dataset"""
    try:
        # Try to load the processed dataset
        df = pd.read_csv('data/fake_news_data.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please run the data preparation script first.")
        return None

def train_model(df, model_type='logistic'):
    """Train a machine learning model"""

    # Preprocess text
    with st.spinner("Preprocessing text data..."):
        df['processed_text'] = df['text'].apply(preprocess_text)

        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]

    # Split the data
    X = df['processed_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize text
    with st.spinner("Creating text features..."):
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

    # Train model
    with st.spinner(f"Training {model_type} model..."):
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42)
        elif model_type == 'naive_bayes':
            model = MultinomialNB()
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train_vec, y_train)

    # Make predictions
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)

    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()

    return model, vectorizer, accuracy, y_test, y_pred, y_pred_proba

def predict_text_local(text, model, vectorizer):
    """Predict if text is fake or real using local models"""
    processed_text = preprocess_text(text)
    text_vec = vectorizer.transform([processed_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    return prediction, probability

def predict_with_saved_model(text, model_name='logistic'):
    """Predict using saved models"""
    if st.session_state.model_loader:
        try:
            result = st.session_state.model_loader.predict(text, model_name)
            return result['prediction'], [result['probability']['FAKE'], result['probability']['REAL']]
        except Exception as e:
            st.error(f"Error with saved model prediction: {e}")
            return None, None
    return None, None

def main():
    st.set_page_config(
        page_title="Fake News Detection System",
        page_icon="📰",
        layout="wide"
    )

    # Header
    st.title("📰 Fake News Detection System")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔍 Detection", "📊 Analytics", "🎯 Model Training"]
    )

    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Detection":
        show_detection_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "🎯 Model Training":
        show_training_page()

def show_home_page():
    """Show the home page"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Welcome to the Fake News Detection System")
        st.write("""
        This application helps you identify whether a news article is real or fake using machine learning.

        **Features:**
        - 🔍 Real-time fake news detection
        - 📊 Data analytics and visualization
        - 🎯 Multiple machine learning models
        - 📈 Performance metrics and evaluation

        **How it works:**
        1. **Train a Model**: Use our dataset to train a machine learning model
        2. **Detect Fake News**: Input any news article to get a prediction
        3. **Analyze Results**: View detailed analytics and model performance
        """)

        # Dataset info
        df = load_dataset()
        if df is not None:
            st.success(f"✅ Dataset loaded: {len(df):,} articles ready for analysis")

            col_real, col_fake = st.columns(2)
            with col_real:
                real_count = (df['label'] == 'REAL').sum()
                st.metric("Real News Articles", f"{real_count:,}")
            with col_fake:
                fake_count = (df['label'] == 'FAKE').sum()
                st.metric("Fake News Articles", f"{fake_count:,}")

    with col2:
        st.image("https://via.placeholder.com/300x200/1f77b4/ffffff?text=Fake+News+Detection", caption="AI-Powered Detection")

def show_detection_page():
    """Show the detection page"""
    st.header("🔍 Fake News Detection")

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Model Training' section.")
        return

    # Model selection if saved models are available
    available_models = ['logistic']  # Default
    selected_model = 'logistic'

    if st.session_state.model_loader:
        available_models = st.session_state.model_loader.get_available_models()
        if len(available_models) > 1:
            selected_model = st.selectbox("Choose model:", available_models)

    st.write("Enter a news article below to check if it's real or fake:")

    # Text input methods
    input_method = st.radio("Choose input method:", ["✍️ Type/Paste Text", "📄 Upload File"])

    text_to_analyze = ""

    if input_method == "✍️ Type/Paste Text":
        text_to_analyze = st.text_area(
            "Enter news article text:",
            height=200,
            placeholder="Paste your news article here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        if uploaded_file is not None:
            text_to_analyze = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", text_to_analyze, height=200)

    if text_to_analyze.strip():
        if st.button("🔍 Analyze Article", type="primary"):
            with st.spinner("Analyzing article..."):
                # Try to use saved models first
                prediction, probability = predict_with_saved_model(text_to_analyze, selected_model)

                # Fall back to session models if saved models fail
                if prediction is None and st.session_state.model and st.session_state.vectorizer:
                    prediction, probability = predict_text_local(
                        text_to_analyze,
                        st.session_state.model,
                        st.session_state.vectorizer
                    )

                if prediction is not None:
                    # Display results
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        if prediction == 'REAL':
                            st.success("✅ This article appears to be REAL")
                            confidence = probability[1] * 100  # Probability of being REAL
                        else:
                            st.error("❌ This article appears to be FAKE")
                            confidence = probability[0] * 100  # Probability of being FAKE

                        st.metric("Confidence", f"{confidence:.1f}%")

                        # Show model used
                        st.info(f"Model used: {selected_model}")

                    with col2:
                        # Probability chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['FAKE', 'REAL'],
                                y=[probability[0] * 100, probability[1] * 100],
                                marker_color=['red', 'green']
                            )
                        ])
                        fig.update_layout(
                            title="Prediction Probabilities",
                            yaxis_title="Probability (%)",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Error making prediction. Please check the model.")

def show_analytics_page():
    """Show the analytics page"""
    st.header("📊 Data Analytics")

    df = load_dataset()
    if df is None:
        return

    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Articles", f"{len(df):,}")
    with col2:
        real_pct = (df['label'] == 'REAL').mean() * 100
        st.metric("Real News %", f"{real_pct:.1f}%")
    with col3:
        fake_pct = (df['label'] == 'FAKE').mean() * 100
        st.metric("Fake News %", f"{fake_pct:.1f}%")

    # Class distribution
    st.subheader("Class Distribution")
    label_counts = df['label'].value_counts()

    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Distribution of Real vs Fake News",
        color_discrete_map={'REAL': 'green', 'FAKE': 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Text length analysis
    st.subheader("Text Length Analysis")
    df['text_length'] = df['text'].str.len()

    fig = px.histogram(
        df,
        x='text_length',
        color='label',
        title="Distribution of Article Lengths",
        nbins=50,
        color_discrete_map={'REAL': 'green', 'FAKE': 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sample articles
    st.subheader("Sample Articles")
    sample_real = df[df['label'] == 'REAL'].sample(1).iloc[0]
    sample_fake = df[df['label'] == 'FAKE'].sample(1).iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.success("**Sample Real News**")
        st.write(sample_real['text'][:500] + "..." if len(sample_real['text']) > 500 else sample_real['text'])

    with col2:
        st.error("**Sample Fake News**")
        st.write(sample_fake['text'][:500] + "..." if len(sample_fake['text']) > 500 else sample_fake['text'])

def show_training_page():
    """Show the model training page"""
    st.header("🎯 Model Training")

    df = load_dataset()
    if df is None:
        return

    st.write("Choose a machine learning model to train:")

    # Model selection
    model_type = st.selectbox(
        "Select Model:",
        ["logistic", "naive_bayes", "random_forest"],
        format_func=lambda x: {
            "logistic": "Logistic Regression",
            "naive_bayes": "Naive Bayes",
            "random_forest": "Random Forest"
        }[x]
    )

    if st.button("🚀 Train Model", type="primary"):
        try:
            model, vectorizer, accuracy, y_test, y_pred, y_pred_proba = train_model(df, model_type)

            # Save to session state
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.model_trained = True

            st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.3f}")

            # Show detailed results
            st.subheader("Model Performance")

            col1, col2 = st.columns(2)

            with col1:
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

            with col2:
                # Classification Report
                report = classification_report(y_test, y_pred, output_dict=True)

                st.metric("Accuracy", f"{accuracy:.3f}")
                st.metric("Precision (Fake)", f"{report['FAKE']['precision']:.3f}")
                st.metric("Recall (Fake)", f"{report['FAKE']['recall']:.3f}")
                st.metric("F1-Score (Fake)", f"{report['FAKE']['f1-score']:.3f}")

        except Exception as e:
            st.error(f"Error training model: {str(e)}")

if __name__ == "__main__":
    main()