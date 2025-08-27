# 🚀 Quick Start Guide

## Get Your Movie Recommendation System Running in 3 Steps!

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Preprocess Data (First Time Only)
```bash
python preprocess_data.py
```
*This may take a few minutes the first time*

### Step 3: Start the System
```bash
python app.py
```

### 🌐 Open Your Browser
Go to: **http://localhost:5000**

---

## 🎯 What You Can Do

1. **Search by Movie Title**: Type "The Dark Knight" to get similar movies
2. **Search by Genre**: Type "action", "comedy", "horror"
3. **Search by Actor/Director**: Type "Christopher Nolan", "Tom Hanks"
4. **Search by Keywords**: Type "superhero", "space", "war"

## 🔧 Alternative Ways to Start

### Option 1: Use the Startup Script
```bash
python run.py
```

### Option 2: Windows Users
Double-click `start.bat`

### Option 3: Manual Start
```bash
# Check if data is ready
ls *.pkl

# If files exist, start the app
python app.py
```

## 🆘 Troubleshooting

- **"Module not found"**: Run `pip install -r requirements.txt`
- **"Model not loaded"**: Run `python preprocess_data.py`
- **Port 5000 busy**: Change port in `app.py` or kill existing process

---

**🎬 Enjoy discovering your next favorite movies!**

