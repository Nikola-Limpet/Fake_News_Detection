#!/bin/bash
# quickstart.sh - Quick setup and launch script for Fake News Detection System

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "==========================================="
echo "   FAKE NEWS DETECTION SYSTEM"
echo "   Quick Start Setup"
echo "==========================================="
echo -e "${NC}"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $PYTHON_VERSION found"
    
    # Check if Python version is 3.8+
    MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [ "$MAJOR" -lt 3 ] || [ "$MAJOR" -eq 3 -a "$MINOR" -lt 8 ]; then
        print_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi
else
    print_error "Python 3 is not installed"
    exit 1
fi

# Create virtual environment
print_info "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

# Install requirements
print_info "Installing requirements (this may take a while)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    print_success "Requirements installed"
else
    print_warning "requirements.txt not found, installing basic packages..."
    pip install streamlit pandas numpy scikit-learn tensorflow torch transformers nltk -q
fi

# Download NLTK data
print_info "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"
print_success "NLTK data downloaded"

# Create necessary directories
print_info "Creating project directories..."
mkdir -p data models logs notebooks src

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file..."
    cat > .env << EOL
# Environment Variables for Fake News Detection System
NEWS_API_KEY=your_api_key_here
KAGGLE_USERNAME=your_username_here
KAGGLE_KEY=your_key_here
MODEL_PATH=models/
DATA_PATH=data/
LOG_PATH=logs/
EOL
    print_warning ".env file created - please update with your API keys"
else
    print_info ".env file already exists"
fi

# Check for data
print_info "Checking for dataset..."
if [ ! -f "data/fake_news_data.csv" ]; then
    print_warning "Dataset not found"
    echo -e "${YELLOW}Would you like to:${NC}"
    echo "1) Download from Kaggle (requires API key)"
    echo "2) Create sample dataset"
    echo "3) Skip for now"
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            print_info "Setting up Kaggle download..."
            python3 data_preparation.py
            ;;
        2)
            print_info "Creating sample dataset..."
            python3 -c "
import pandas as pd
import numpy as np

# Create sample data
real_news = ['Scientists discover new treatment showing promise in clinical trials'] * 10
fake_news = ['SHOCKING: Miracle cure found that doctors dont want you to know'] * 10

df = pd.DataFrame({
    'text': real_news + fake_news,
    'label': ['REAL']*10 + ['FAKE']*10
})
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('data/fake_news_data.csv', index=False)
print('Sample dataset created')
            "
            print_success "Sample dataset created"
            ;;
        3)
            print_info "Skipping dataset setup"
            ;;
    esac
else
    print_success "Dataset found"
fi

# Menu for actions
echo ""
echo -e "${BLUE}==========================================="
echo "   SETUP COMPLETE"
echo "==========================================="
echo -e "${NC}"
echo "What would you like to do?"
echo "1) Train models"
echo "2) Launch Streamlit app"
echo "3) Launch training monitor"
echo "4) Run both apps"
echo "5) Run with Docker"
echo "6) Exit"
read -p "Enter choice (1-6): " action

case $action in
    1)
        print_info "Starting model training..."
        python3 fake_news_detection.py
        ;;
    2)
        print_info "Launching Streamlit app..."
        print_success "App will be available at http://localhost:8501"
        streamlit run streamlit_app.py
        ;;
    3)
        print_info "Launching training monitor..."
        print_success "Monitor will be available at http://localhost:8502"
        streamlit run training_monitor.py --server.port 8502
        ;;
    4)
        print_info "Launching both apps..."
        print_success "Main app: http://localhost:8501"
        print_success "Monitor: http://localhost:8502"
        
        # Start both apps in background
        streamlit run streamlit_app.py &
        APP_PID=$!
        streamlit run training_monitor.py --server.port 8502 &
        MONITOR_PID=$!
        
        print_info "Press Ctrl+C to stop both apps"
        
        # Wait for interrupt
        trap "kill $APP_PID $MONITOR_PID 2>/dev/null" INT
        wait
        ;;
    5)
        print_info "Starting with Docker..."
        if command -v docker-compose &> /dev/null; then
            docker-compose up --build
        else
            print_error "Docker Compose is not installed"
            print_info "Install Docker and Docker Compose from https://docs.docker.com/get-docker/"
        fi
        ;;
    6)
        print_success "Goodbye!"
        exit 0
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

---

# quickstart.bat - Windows version
@echo off
setlocal EnableDelayedExpansion

echo ==========================================
echo    FAKE NEWS DETECTION SYSTEM
echo    Quick Start Setup (Windows)
echo ==========================================
echo.

:: Check Python
echo [INFO] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

:: Create virtual environment
echo [INFO] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [SUCCESS] Virtual environment created
) else (
    echo [WARNING] Virtual environment already exists
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

:: Install requirements
echo [INFO] Installing requirements...
if exist "requirements.txt" (
    pip install -r requirements.txt
    echo [SUCCESS] Requirements installed
) else (
    echo [WARNING] requirements.txt not found
    pip install streamlit pandas numpy scikit-learn tensorflow torch transformers nltk
)

:: Download NLTK data
echo [INFO] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

:: Create directories
echo [INFO] Creating project directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "notebooks" mkdir notebooks
if not exist "src" mkdir src

:: Create .env file
if not exist ".env" (
    echo [INFO] Creating .env file...
    (
        echo # Environment Variables
        echo NEWS_API_KEY=your_api_key_here
        echo KAGGLE_USERNAME=your_username_here
        echo KAGGLE_KEY=your_key_here
    ) > .env
    echo [WARNING] Please update .env with your API keys
)

:: Menu
echo.
echo ==========================================
echo    SETUP COMPLETE
echo ==========================================
echo.
echo What would you like to do?
echo 1) Train models
echo 2) Launch Streamlit app
echo 3) Launch training monitor
echo 4) Exit
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo [INFO] Starting model training...
    python fake_news_detection.py
) else if "%choice%"=="2" (
    echo [INFO] Launching Streamlit app...
    echo [SUCCESS] App will be available at http://localhost:8501
    streamlit run streamlit_app.py
) else if "%choice%"=="3" (
    echo [INFO] Launching training monitor...
    echo [SUCCESS] Monitor will be available at http://localhost:8502
    streamlit run training_monitor.py --server.port 8502
) else if "%choice%"=="4" (
    echo [SUCCESS] Goodbye!
    exit /b 0
) else (
    echo [ERROR] Invalid choice
    pause
    exit /b 1
)

pause