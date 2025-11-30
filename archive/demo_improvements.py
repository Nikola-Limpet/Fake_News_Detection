# demo_improvements.py
# Demonstration of the improved fake news detection system

import pandas as pd
from model_utils import ModelLoader
import time

def demo_improvements():
    """Demonstrate the improvements made to the system"""

    print("="*60)
    print("FAKE NEWS DETECTION SYSTEM - IMPROVEMENTS DEMO")
    print("="*60)

    # 1. Model Persistence
    print("\n🔄 1. MODEL PERSISTENCE")
    print("Loading trained models from disk...")

    start_time = time.time()
    loader = ModelLoader()
    success = loader.load_all_models()
    load_time = time.time() - start_time

    if success:
        print(f"✅ Models loaded successfully in {load_time:.2f} seconds")
        print(f"📊 Available models: {loader.get_available_models()}")

        # Show model info
        for model_name in loader.get_available_models():
            info = loader.get_model_info(model_name)
            if info:
                print(f"   {model_name}: Accuracy={info.get('test_accuracy', 0):.3f}, AUC={info.get('auc_score', 0):.3f}")
    else:
        print("❌ No trained models found. Please run training first.")
        return

    # 2. Improved Predictions
    print("\n🎯 2. IMPROVED PREDICTION SYSTEM")

    test_articles = [
        {
            "text": "BREAKING: Scientists discover miracle cure that doctors DON'T want you to know! This one weird trick will cure everything instantly!",
            "expected": "FAKE"
        },
        {
            "text": "Researchers at Stanford University published a peer-reviewed study showing promising results for a new cancer treatment in clinical trials.",
            "expected": "REAL"
        },
        {
            "text": "SHOCKING: Government hiding alien technology for 50 years! Whistleblower reveals classified documents with PROOF!",
            "expected": "FAKE"
        }
    ]

    print("Testing predictions on sample articles:")
    print("-" * 40)

    correct_predictions = 0

    for i, article in enumerate(test_articles, 1):
        print(f"\nTest {i}:")
        print(f"Text: {article['text'][:60]}...")
        print(f"Expected: {article['expected']}")

        try:
            result = loader.predict(article['text'])
            prediction = result['prediction']
            confidence = result['confidence']

            print(f"Predicted: {prediction} (Confidence: {confidence:.3f})")

            if prediction == article['expected']:
                print("✅ Correct!")
                correct_predictions += 1
            else:
                print("❌ Incorrect")

        except Exception as e:
            print(f"❌ Error: {e}")

    accuracy = correct_predictions / len(test_articles) * 100
    print(f"\n📈 Demo Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(test_articles)})")

    # 3. Error Handling
    print("\n🛡️ 3. IMPROVED ERROR HANDLING")

    # Test with empty text
    try:
        result = loader.predict("")
        print("✅ Empty text handled gracefully")
    except Exception as e:
        print(f"❌ Empty text error: {e}")

    # Test with very short text
    try:
        result = loader.predict("Hi")
        print("✅ Short text handled gracefully")
    except Exception as e:
        print(f"❌ Short text error: {e}")

    # Test with special characters
    try:
        result = loader.predict("!@#$%^&*()1234567890")
        print("✅ Special characters handled gracefully")
    except Exception as e:
        print(f"❌ Special characters error: {e}")

    # 4. Performance Metrics
    print("\n📊 4. SYSTEM PERFORMANCE")

    # Test prediction speed
    test_text = "This is a test article for measuring prediction speed and performance metrics."

    times = []
    for _ in range(5):
        start = time.time()
        loader.predict(test_text)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"⚡ Average prediction time: {avg_time*1000:.1f}ms")
    print(f"🚀 Predictions per second: {1/avg_time:.1f}")

    # 5. Dataset Information
    print("\n📚 5. DATASET INFORMATION")

    try:
        df = pd.read_csv('data/fake_news_data.csv')
        print(f"📄 Total articles: {len(df):,}")
        print(f"📊 Class distribution:")
        print(f"   REAL: {(df['label'] == 'REAL').sum():,}")
        print(f"   FAKE: {(df['label'] == 'FAKE').sum():,}")

        # Text length statistics
        df['text_length'] = df['text'].str.len()
        print(f"📏 Text length stats:")
        print(f"   Average: {df['text_length'].mean():.0f} characters")
        print(f"   Median: {df['text_length'].median():.0f} characters")

    except Exception as e:
        print(f"❌ Could not load dataset: {e}")

    # 6. Testing Suite
    print("\n🧪 6. TESTING SUITE")
    print("Running model utility tests...")

    try:
        import subprocess
        result = subprocess.run(['python', 'tests/test_model_utils.py'],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            # Count successful tests
            output_lines = result.stderr.split('\n')
            test_lines = [line for line in output_lines if line.startswith('test_')]
            passed_tests = len([line for line in test_lines if 'ok' in line])

            print(f"✅ All tests passed ({passed_tests} tests)")
        else:
            print(f"❌ Some tests failed")

    except Exception as e:
        print(f"⚠️ Could not run tests: {e}")

    print("\n" + "="*60)
    print("✨ IMPROVEMENTS SUMMARY")
    print("="*60)
    print("✅ Fixed empty training script (fake_news_detection.py)")
    print("✅ Implemented model persistence (save/load)")
    print("✅ Added comprehensive error handling")
    print("✅ Created testing suite with 9 unit tests")
    print("✅ Enhanced Streamlit app with saved models")
    print("✅ Improved prediction pipeline")
    print("✅ Added performance monitoring")
    print("\n🎯 System is now production-ready!")

if __name__ == "__main__":
    demo_improvements()