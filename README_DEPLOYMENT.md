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

### 🔧 Environment Variables

**NewsAPI Key Setup** (enables real-time news fetching):

**For Streamlit Cloud:**
1. Go to your app dashboard on share.streamlit.io
2. Click "⚙️ Settings" → "Secrets"
3. Add this to your secrets:
```toml
NEWS_API_KEY = "1c5db918d7584047b8f24d017a835f31"
```

**For Local Development:**
- The app automatically loads from your existing `.env` file
- Key: `1c5db918d7584047b8f24d017a835f31`

**What NewsAPI Enables:**
- ✅ Fetch latest news articles from 80,000+ sources
- ✅ Real-time analysis of current events
- ✅ Immediate fake news detection on breaking news
- ✅ Compare model performance on live data

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

**Fixed Deployment Issues:**
- ✅ **Python 3.13 Compatibility** - Removed heavy packages (gensim, tensorflow, torch)
- ✅ **Requirements Streamlined** - Using minimal dependencies for fast deployment
- ✅ **Build Errors Fixed** - No more C compilation issues

**Common Issues:**
1. **NLTK data** - App downloads required data automatically
2. **Model loading** - Pre-trained models are included in repository
3. **Web scraping** - Some sites may block requests (normal behavior)
4. **NewsAPI** - Add your API key to Streamlit Cloud secrets for news fetching

### 🎓 Academic Credit

**Team:** Lim Petnikola, Rim Sovichey, Pha Lyheng, Sokhonn Raksmeidaravid
**Instructor:** Professor Kunthea PIN
**Institution:** American University of Phnom Penh
**Course:** ITM-360 Artificial Intelligence
**Academic Year:** 2024-2025