from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import re
import requests
import json

app = Flask(__name__)

# Load the pre-trained model and data
try:
    with open('api/movies.pkl', 'rb') as f:
        new_df = pickle.load(f)
    with open('api/movie_dict.pkl', 'rb') as f:
        movie_dict = pickle.load(f)
    # Recreate the similarity matrix
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please run the preprocessing script first.")
    new_df = None
    movie_dict = None
    similarity = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/movies')
def get_movies():
    if new_df is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get all movie titles for search suggestions
    movies = new_df['title'].tolist()
    return jsonify({'movies': movies})

@app.route('/api/recommend', methods=['POST'])
def recommend_movies():
    if new_df is None or similarity is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Find the movie index
        movie_index = new_df[new_df['title'] == query].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
        
        recommendations = []
        for i in movies_list:
            movie_title = new_df.iloc[i[0]]['title']
            # Get movie details from the original data
            movie_details = get_movie_details(movie_title)
            recommendations.append(movie_details)
        
        return jsonify({
            'query': query,
            'recommendations': recommendations
        })
    
    except (IndexError, KeyError):
        # If exact match not found, search by genre or keyword
        return search_by_genre_or_keyword(query)

def search_by_genre_or_keyword(query):
    """Search movies by genre, keyword, or partial title match"""
    query_lower = query.lower()
    
    # Search in genres, keywords, cast, crew, and overview
    matching_movies = []
    
    for idx, row in new_df.iterrows():
        title = row['title'].lower()
        tags = row['tags'].lower()
        
        # Check if query matches title, genres, or other tags
        if (query_lower in title or 
            query_lower in tags or
            any(query_lower in tag for tag in tags.split())):
            movie_details = get_movie_details(row['title'])
            matching_movies.append(movie_details)
            
            if len(matching_movies) >= 10:  # Limit results
                break
    
    # Sort by relevance (title matches first)
    matching_movies.sort(key=lambda x: query_lower in x['title'].lower(), reverse=True)
    
    return jsonify({
        'query': query,
        'recommendations': matching_movies[:10],
        'search_type': 'keyword_search'
    })

def get_movie_details(title):
    """Get detailed movie information including poster"""
    try:
        # Load original movie data
        movies_df = pd.read_csv('api/tmdb_5000_movies.csv')
        movie_row = movies_df[movies_df['title'] == title]
        
        if not movie_row.empty:
            movie = movie_row.iloc[0]
            
            # Parse genres into names array string for frontend
            genres_value = '[]'
            try:
                if pd.notna(movie.get('genres')):
                    raw_genres = movie.get('genres')
                    # CSV contains a JSON-like string array of objects with name
                    genres_objs = json.loads(raw_genres.replace("'", '"')) if isinstance(raw_genres, str) else []
                    if isinstance(genres_objs, list):
                        genre_names = [g.get('name') for g in genres_objs if isinstance(g, dict) and g.get('name')]
                        genres_value = json.dumps(genre_names)
            except Exception:
                genres_value = '[]'

            # Try to get poster from free movie poster API (OMDb demo key)
            poster_url = get_movie_poster(title, movie.get('release_date'))
            
            return {
                'title': movie['title'],
                'overview': movie['overview'] if pd.notna(movie['overview']) else 'No overview available',
                'genres': genres_value,
                'release_date': movie['release_date'] if pd.notna(movie['release_date']) else 'Unknown',
                'vote_average': float(movie['vote_average']) if pd.notna(movie['vote_average']) else 0.0,
                'poster_path': None,
                'poster_url': poster_url
            }
    except Exception as e:
        print(f"Error getting movie details for {title}: {e}")
    
    # Fallback to basic info with poster search
    poster_url = get_movie_poster(title, None)
    return {
        'title': title,
        'overview': 'No overview available',
        'genres': '[]',
        'release_date': 'Unknown',
        'vote_average': 0.0,
        'poster_path': None,
        'poster_url': poster_url
    }

def get_movie_poster(title, release_date):
    """Get movie poster from free API services"""
    try:
        # Try to get poster from a working free movie poster service
        poster_url = fetch_movie_poster(title, release_date)
        if poster_url and poster_url != "placeholder":
            return poster_url
            
        # If no poster found, return placeholder
        return "placeholder"
        
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
        return "placeholder"

def fetch_movie_poster(title, release_date):
    """Fetch movie poster from a working free service"""
    try:
        # Clean the title for better search
        clean_title = re.sub(r'[^\\w\\s]', '', title).strip()
        
        # Try OMDb first (demo key). It often provides direct poster URLs
        year = None
        if release_date and isinstance(release_date, str) and len(release_date) >= 4:
            year = release_date[:4]
        try:
            params = {
                't': clean_title,
                'y': year or '',
                'apikey': 'thewdb'
            }
            resp = requests.get('http://www.omdbapi.com/', params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                poster = data.get('Poster')
                if poster and poster != 'N/A':
                    return poster
        except Exception:
            pass

        # Fallback to a simple generated poster (none -> placeholder handled by UI)
        poster_url = generate_custom_poster(title, release_date)
        if poster_url:
            return poster_url
            
        # If no poster found, return placeholder
        return "placeholder"
        
    except Exception as e:
        print(f"Poster fetch error for {title}: {e}")
        return "placeholder"

def generate_custom_poster(title, release_date):
    """Generate a custom movie poster using a free service"""
    try:
        # We'll use a free image generation service to create custom posters
        # This approach doesn't require API keys and creates unique posters
        
        # Clean the title for the poster
        clean_title = re.sub(r'[^\\w\\s]', '', title).strip()
        
        # Try to create a poster using a free service
        # For now, let's return None to use the attractive placeholder
        # In a production system, you could:
        # 1. Use a free image generation service
        # 2. Create posters with movie titles and themes
        # 3. Cache generated posters for performance
        
        return None
        
    except Exception as e:
        print(f"Custom poster generation error for {title}: {e}")
        return None

def generate_placeholder_poster(title):
    """Generate a placeholder poster URL or use a default image service"""
    # Try to get a movie poster from a free API service
    # For now, let's create an attractive placeholder with the movie title
    
    # Option 1: Use a movie-themed placeholder
    # Option 2: Use a gradient with movie icon
    # Option 3: Use a free movie poster service
    
    # Create a nice-looking placeholder using CSS gradients and text
    # This will be handled by the frontend CSS
    
    # For now, return a flag that the frontend can use to show a placeholder
    return "placeholder"

@app.route('/api/search')
def search_movies():
    if new_df is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'movies': []})
    
    # Search for movies that contain the query
    matching_movies = []
    query_lower = query.lower()
    
    for idx, row in new_df.iterrows():
        title = row['title']
        if query_lower in title.lower():
            movie_details = get_movie_details(title)
            matching_movies.append(movie_details)
            
            if len(matching_movies) >= 20:  # Limit search results
                break
    
    return jsonify({'movies': matching_movies})
