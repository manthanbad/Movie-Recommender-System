import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re

def clean_text(text):
    """Simple text cleaning function"""
    if pd.isna(text):
        return ""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Loading data...")
    # Load the CSV files
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
    
    print("Merging datasets...")
    # Merge credits and movies on title
    movies = movies.merge(credits, on='title')
    
    # Select relevant columns
    movies = movies[['movie_id', 'cast', 'crew', 'keywords', 'title', 'overview', 'genres']]
    
    print("Cleaning data...")
    # Remove null values
    movies.dropna(inplace=True)
    
    # Remove duplicates
    movies.drop_duplicates(inplace=True)
    
    print("Processing genres...")
    # Convert genres from string to list
    def convert_genres(obj):
        try:
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        except:
            return []
    
    movies['genres'] = movies['genres'].apply(convert_genres)
    
    print("Processing keywords...")
    # Convert keywords from string to list
    def convert_keywords(obj):
        try:
            L = []
            for i in ast.literal_eval(obj):
                L.append(i['name'])
            return L
        except:
            return []
    
    movies['keywords'] = movies['keywords'].apply(convert_keywords)
    
    print("Processing overview...")
    # Convert overview to list of words and clean
    movies['overview'] = movies['overview'].apply(lambda x: clean_text(x).split())
    
    print("Processing cast...")
    # Convert cast from string to list (top 3 actors)
    def convert_cast(obj):
        try:
            L = []
            counter = 0
            for i in ast.literal_eval(obj):
                if counter != 3:
                    L.append(i['name'])
                    counter += 1
                else:
                    break
            return L
        except:
            return []
    
    movies['cast'] = movies['cast'].apply(convert_cast)
    
    print("Processing crew...")
    # Extract director from crew
    def fetch_director(obj):
        try:
            L = []
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
            return L
        except:
            return []
    
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    print("Cleaning text data...")
    # Clean and combine all text data
    movies['genres'] = movies['genres'].apply(lambda x: [clean_text(i) for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [clean_text(i) for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [clean_text(i) for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [clean_text(i) for i in x])
    
    print("Creating tags...")
    # Combine all text data into tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create new dataframe with relevant columns
    new_df = movies[['movie_id', 'title', 'tags']]
    
    # Convert tags to string
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    
    # Convert to lowercase
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    print("Creating feature vectors...")
    # Create feature vectors using CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    print("Calculating similarity matrix...")
    # Calculate cosine similarity
    similarity = cosine_similarity(vectors)
    
    print("Saving processed data...")
    # Save the processed data
    pickle.dump(new_df, open('movies.pkl', 'wb'))
    pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
    
    print("Data preprocessing completed successfully!")
    print(f"Total movies processed: {len(new_df)}")
    print(f"Feature vector shape: {vectors.shape}")
    print(f"Similarity matrix shape: {similarity.shape}")
    
    # Test the recommendation system
    print("\nTesting recommendation system...")
    test_movie = new_df.iloc[0]['title']
    print(f"Testing with movie: {test_movie}")
    
    movie_index = new_df[new_df['title'] == test_movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    print("Top 5 recommendations:")
    for i in movies_list:
        print(f"- {new_df.iloc[i[0]]['title']}")

if __name__ == "__main__":
    main()
