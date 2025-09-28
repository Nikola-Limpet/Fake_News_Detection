# data_preparation.py
# Script to download and prepare the fake news dataset

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
import zipfile
from pathlib import Path
import kaggle
import json

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("Setting up Kaggle API...")
    
    # Check if kaggle.json exists
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("\nKaggle API credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click on 'Create New API Token'")
        print("3. Save the downloaded kaggle.json to ~/.kaggle/")
        print("   (Windows: C:\\Users\\YourName\\.kaggle\\)")
        
        username = input("\nEnter your Kaggle username: ")
        key = input("Enter your Kaggle API key: ")
        
        # Create kaggle directory if it doesn't exist
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        # Save credentials
        credentials = {"username": username, "key": key}
        
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        # Set permissions (Unix-like systems only)
        try:
            os.chmod(kaggle_json, 0o600)
        except:
            pass
    
    return True

def download_kaggle_dataset():
    """Download the fake news dataset from Kaggle"""
    try:
        setup_kaggle_api()
        
        print("\nDownloading dataset from Kaggle...")
        
        # Import kaggle after setup
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download the dataset
        dataset = 'clmentbisaillon/fake-and-real-news-dataset'
        api.dataset_download_files(dataset, path='data/', unzip=True)
        
        print("Dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False

def prepare_combined_dataset():
    """Combine and prepare the dataset"""
    print("\nPreparing dataset...")
    
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Try to load Kaggle dataset
    fake_path = data_dir / 'Fake.csv'
    real_path = data_dir / 'True.csv'
    
    if fake_path.exists() and real_path.exists():
        print("Loading Kaggle dataset...")
        
        # Load datasets
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Add labels
        fake_df['label'] = 'FAKE'
        real_df['label'] = 'REAL'
        
        # Combine datasets
        df = pd.concat([fake_df, real_df], ignore_index=True)
        
        # Rename columns if necessary
        if 'title' in df.columns and 'text' in df.columns:
            df['text'] = df['title'] + ' ' + df['text']
        
        print(f"Dataset loaded: {len(df)} samples")
        
    else:
        print("Kaggle dataset not found. Creating sample dataset...")
        df = create_sample_dataset()
    
    # Clean the dataset
    df = clean_dataset(df)
    
    # Split the dataset
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    # Save datasets
    train_df.to_csv(data_dir / 'train.csv', index=False)
    val_df.to_csv(data_dir / 'validation.csv', index=False)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    df.to_csv(data_dir / 'fake_news_data.csv', index=False)
    
    print(f"\nDataset split:")
    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Print class distribution
    print(f"\nClass distribution in training set:")
    print(train_df['label'].value_counts())
    
    return df

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    
    real_news = [
        "Scientists at Stanford University have developed a new machine learning algorithm that can predict protein structures with unprecedented accuracy. The breakthrough, published in Nature, could accelerate drug discovery and our understanding of biological processes.",
        
        "The Federal Reserve announced a quarter-point interest rate increase today, citing persistent inflation concerns. The decision marks the third rate hike this year as the central bank attempts to balance economic growth with price stability.",
        
        "A new study published in The Lancet shows that regular exercise can reduce the risk of developing Alzheimer's disease by up to 30%. Researchers followed 10,000 participants over 15 years and found significant cognitive benefits from moderate physical activity.",
        
        "Tesla reported record quarterly earnings, beating Wall Street expectations with revenue of $23.5 billion. The electric vehicle manufacturer delivered 435,000 vehicles in the quarter, representing a 35% year-over-year increase.",
        
        "NASA's Perseverance rover has discovered organic molecules in Martian rock samples, suggesting the planet may have once harbored conditions suitable for life. The findings will be further analyzed when the samples return to Earth in 2033.",
        
        "The European Union has approved new regulations requiring all smartphones to use USB-C charging ports by 2024. The legislation aims to reduce electronic waste and simplify charging solutions for consumers.",
        
        "Researchers at MIT have developed a new battery technology that could charge electric vehicles in under 10 minutes. The solid-state battery design uses a ceramic electrolyte that enables faster ion movement while maintaining safety.",
        
        "The World Health Organization reports that global malaria cases have declined by 20% over the past five years, thanks to improved prevention methods and treatment accessibility in affected regions.",
        
        "Amazon Web Services experienced a major outage affecting thousands of websites and services worldwide. The company attributed the 4-hour disruption to a configuration error in their Northern Virginia data center.",
        
        "A federal judge has ruled that social media companies can be held liable for algorithmic recommendations that promote harmful content, setting a potential precedent for platform accountability."
    ]
    
    fake_news = [
        "BREAKING: Scientists confirm that drinking hot lemon water every morning cures cancer! Big Pharma has been hiding this simple trick that costs less than $1. Share before they delete this!",
        
        "Bill Gates admits to putting microchips in vaccines during secret interview! The billionaire finally confessed his plan for global population control through 5G activated nanobots.",
        
        "NASA insider reveals the moon landing was filmed in Hollywood! Stanley Kubrick's widow has documents proving the entire Apollo mission was an elaborate hoax funded by the government.",
        
        "Doctors HATE this one weird trick that reverses aging by 20 years! Local mom discovers fountain of youth in common kitchen spice. Click here before Big Pharma shuts us down!",
        
        "URGENT: New world order announces mandatory digital currency by next month! All cash will be worthless - convert your savings NOW or lose everything! Forward to everyone you know!",
        
        "Celebrity dies from vaccine, mainstream media covers it up! The truth about what really happened is being censored across all platforms. Only we have the courage to tell you the real story.",
        
        "Ancient aliens built the pyramids, government finally admits! Classified documents reveal extraterrestrial technology has been reverse-engineered for decades.",
        
        "Miracle fruit from Amazon rainforest makes you lose 30 pounds in 10 days! Celebrities are using this secret method to stay thin. Limited supply - order now!",
        
        "Weather control devices cause recent hurricanes, whistleblower reveals! Government using HAARP technology to manipulate climate for political gain.",
        
        "Schools secretly teaching children to worship Satan, parent's video goes viral! Shocking footage shows ritual in classroom. Mainstream media refuses to cover this story!"
    ]
    
    # Create DataFrame
    data = []
    
    for text in real_news:
        data.append({'text': text, 'label': 'REAL'})
    
    for text in fake_news:
        data.append({'text': text, 'label': 'FAKE'})
    
    # Add more synthetic samples
    print("Generating synthetic samples...")
    
    # Real news patterns
    real_templates = [
        "Researchers at {university} have published a study in {journal} showing that {finding}.",
        "The {organization} announced {policy} aimed at {goal}.",
        "A new report from {source} indicates that {trend} over the past {timeframe}.",
        "{company} reported {financial_metric} of {amount}, {performance} analyst expectations.",
        "Scientists have discovered {discovery} that could lead to {application}."
    ]
    
    # Fake news patterns
    fake_templates = [
        "SHOCKING: {celebrity} caught in {scandal}! You won't believe what happens next!",
        "Doctors HATE him! Local man discovers {miracle_cure} using this one simple trick!",
        "BREAKING: Government admits {conspiracy}! Share before they delete this!",
        "{product} causes {disease}, study hidden by Big Pharma reveals!",
        "You've been lied to! The truth about {topic} will SHOCK you!"
    ]
    
    # Generate more samples
    universities = ["Harvard", "MIT", "Oxford", "Cambridge", "Yale"]
    journals = ["Nature", "Science", "The Lancet", "Cell", "NEJM"]
    organizations = ["WHO", "UN", "CDC", "FDA", "EPA"]
    companies = ["Apple", "Google", "Microsoft", "Amazon", "Meta"]
    
    for _ in range(50):
        # Generate real news
        template = np.random.choice(real_templates)
        if "university" in template:
            text = template.format(
                university=np.random.choice(universities),
                journal=np.random.choice(journals),
                finding="a correlation between sleep quality and cognitive performance"
            )
        elif "organization" in template:
            text = template.format(
                organization=np.random.choice(organizations),
                policy="new guidelines",
                goal="improving public health"
            )
        elif "company" in template:
            text = template.format(
                company=np.random.choice(companies),
                financial_metric="quarterly revenue",
                amount="$50 billion",
                performance="exceeding"
            )
        else:
            text = template.format(
                source="industry analysts",
                trend="market growth",
                timeframe="five years",
                discovery="a new material",
                application="improved solar panels"
            )
        data.append({'text': text, 'label': 'REAL'})
    
    for _ in range(50):
        # Generate fake news
        template = np.random.choice(fake_templates)
        text = template.format(
            celebrity="Famous Actor",
            scandal="alien conspiracy",
            miracle_cure="cancer cure",
            conspiracy="mind control program",
            product="5G towers",
            disease="autism",
            topic="vaccines"
        )
        data.append({'text': text, 'label': 'FAKE'})
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Sample dataset created: {len(df)} samples")
    
    return df

def clean_dataset(df):
    """Clean and prepare the dataset"""
    
    # Remove duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=['text'])
    if len(df) < initial_size:
        print(f"Removed {initial_size - len(df)} duplicate entries")
    
    # Remove empty texts
    df = df[df['text'].notna()]
    df = df[df['text'].str.strip() != '']
    
    # Remove very short texts (likely not real articles)
    df = df[df['text'].str.len() > 100]
    
    # Standardize labels
    df['label'] = df['label'].str.upper()
    df['label'] = df['label'].replace({'TRUE': 'REAL', 'FALSE': 'FAKE', '1': 'FAKE', '0': 'REAL'})
    
    # Keep only REAL and FAKE labels
    df = df[df['label'].isin(['REAL', 'FAKE'])]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Final dataset size: {len(df)} samples")
    
    return df

def download_newsapi_samples(api_key=None):
    """Download additional samples from NewsAPI (optional)"""
    if not api_key:
        print("\nSkipping NewsAPI download (no API key provided)")
        return None
    
    print("\nFetching additional samples from NewsAPI...")
    
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'apiKey': api_key,
        'language': 'en',
        'pageSize': 100
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == 'ok':
            articles = []
            for article in data['articles']:
                if article['content']:
                    articles.append({
                        'text': f"{article['title']} {article['description']} {article['content']}",
                        'label': 'REAL',  # Assuming NewsAPI provides real news
                        'source': article['source']['name']
                    })
            
            df = pd.DataFrame(articles)
            print(f"Fetched {len(df)} articles from NewsAPI")
            return df
    
    except Exception as e:
        print(f"Error fetching from NewsAPI: {e}")
    
    return None

def main():
    """Main function to prepare the dataset"""
    
    print("="*60)
    print("FAKE NEWS DETECTION - DATA PREPARATION")
    print("="*60)
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Try to download from Kaggle
    choice = input("\nDo you want to download the dataset from Kaggle? (y/n): ").lower()
    
    if choice == 'y':
        success = download_kaggle_dataset()
        if not success:
            print("\nFalling back to sample dataset...")
    
    # Prepare the dataset
    df = prepare_combined_dataset()
    
    # Optional: Add NewsAPI samples
    api_key = input("\nEnter NewsAPI key (press Enter to skip): ").strip()
    if api_key:
        news_df = download_newsapi_samples(api_key)
        if news_df is not None:
            # Save NewsAPI samples separately
            news_df.to_csv('data/newsapi_samples.csv', index=False)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nYou can now run the training script:")
    print("  python fake_news_detection.py")
    print("\nOr start the Streamlit app:")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()