#!/usr/bin/env python3
"""
Movie Recommendation System Startup Script
This script checks if the data is preprocessed and starts the Flask application.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import nltk
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'tmdb_5000_movies.csv',
        'tmdb_5000_credits.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing data files: {', '.join(missing_files)}")
        return False
    
    print("✓ All required data files found")
    return True

def check_processed_data():
    """Check if data has been preprocessed"""
    processed_files = [
        'movies.pkl',
        'movie_dict.pkl'
    ]
    
    missing_files = []
    for file in processed_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠ Data not preprocessed. Missing: {', '.join(missing_files)}")
        return False
    
    print("✓ Data is preprocessed and ready")
    return True

def preprocess_data():
    """Run the data preprocessing script"""
    print("\n🔄 Preprocessing data...")
    print("This may take a few minutes for the first time...")
    
    try:
        result = subprocess.run([sys.executable, 'preprocess_data.py'], 
                              capture_output=True, text=True, check=True)
        print("✓ Data preprocessing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during preprocessing: {e}")
        print(f"Error output: {e.stderr}")
        return False

def start_app():
    """Start the Flask application"""
    print("\n🚀 Starting Movie Recommendation System...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"✗ Error starting server: {e}")

def main():
    """Main function to orchestrate the startup process"""
    print("🎬 Movie Recommendation System")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check data files
    if not check_data_files():
        return
    
    # Check if data is preprocessed
    if not check_processed_data():
        print("\n📊 Data preprocessing required...")
        user_input = input("Do you want to preprocess the data now? (y/n): ").lower().strip()
        
        if user_input in ['y', 'yes']:
            if not preprocess_data():
                return
        else:
            print("Please run 'python preprocess_data.py' manually before starting the app.")
            return
    
    # Start the application
    start_app()

if __name__ == "__main__":
    main()

