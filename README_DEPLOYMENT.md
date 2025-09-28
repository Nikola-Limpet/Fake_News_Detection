# Fake News Detection System - Streamlit Cloud Deployment

## 🚀 Quick Deploy to Streamlit Cloud

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at share.streamlit.io)

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `streamlit_app.py`
   - Use `requirements_streamlit.txt` for dependencies
   - Deploy!

### 📁 Key Files for Deployment

- **`streamlit_app.py`** - Main application file
- **`requirements_streamlit.txt`** - Streamlined dependencies for cloud
- **`packages.txt`** - System-level dependencies
- **`.gitignore`** - Excludes unnecessary files
- **`models/`** - Pre-trained ML models (included in repo)
- **`model_utils.py`** - Model loading utilities

### 🔧 Environment Variables (Optional)

If using NewsAPI features, add in Streamlit Cloud secrets:
```toml
NEWS_API_KEY = "your_newsapi_key_here"
```

### 🎯 Features Available

✅ **Text Analysis** - Paste news article text for analysis
✅ **URL Analysis** - Enter news article URL for automatic scraping
✅ **Analytics Dashboard** - View prediction history and statistics
✅ **Web Scraping** - Extract content from most news websites
✅ **Dark Theme** - Professional UI with warm color scheme
✅ **Mobile Responsive** - Works on all devices

### 📊 Models Included

- **Logistic Regression** - Fast, accurate baseline model (97.4% accuracy)
- **TF-IDF Vectorizer** - Text feature extraction
- **NLTK Preprocessing** - Text cleaning and normalization

### 🛠️ Local Development

```bash
# Clone and setup
git clone <your-repo-url>
cd fake_news_detection
pip install -r requirements_streamlit.txt

# Run locally
streamlit run streamlit_app.py
```

### 🔍 Troubleshooting

**Common Issues:**
1. **Memory errors** - Use `requirements_streamlit.txt` (lighter dependencies)
2. **NLTK data** - App downloads required data automatically
3. **Model loading** - Models are included in repository
4. **Web scraping** - Some sites may block requests (normal behavior)

### 🎓 Academic Credit

**Team:** Lim Petnikola, Rim Sovichey, Pha Lyheng, Sokhonn Raksmeidaravid
**Instructor:** Professor Kunthea PIN
**Institution:** American University of Phnom Penh
**Course:** ITM-360 Artificial Intelligence
**Academic Year:** 2024-2025