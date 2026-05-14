import os
import pandas as pd
from ml.services.web_api_service import WebAPIService

DB_DATA_PATH = "ml/data/database_data"

GENRES = []

def convert_db_user_features_to_csv():
    # Lấy dữ liệu từ DB
    user_features_data = WebAPIService.get_all_user_features()
    db_genre_columns = [f"{name.lower().replace('-', '_').replace(' ', '_')}_avg" for _, name in GENRES]
    
    user_features_df = pd.DataFrame(
        user_features_data,
        columns=['user_id'] + db_genre_columns
    )
    TARGET_COLUMNS = [
        'user_id', 'action_avg', 'adventure_avg', 'animation_avg', 'children_avg', 
        'comedy_avg', 'crime_avg', 'documentary_avg', 'drama_avg', 'fantasy_avg', 
        'film_noir_avg', 'horror_avg', 'imax_avg', 'musical_avg', 'mystery_avg', 
        'romance_avg', 'sci_fi_avg', 'thriller_avg', 'war_avg', 'western_avg'
    ]

    for col in TARGET_COLUMNS:
        if col not in user_features_df.columns:
            user_features_df[col] = 0.0
    user_features_df = user_features_df[TARGET_COLUMNS]
    
    # Lưu ra file
    user_features_df.to_csv(os.path.join(DB_DATA_PATH, 'db_user_features.csv'), index=False)

def convert_db_ratings_to_csv():
    # Lấy dữ liệu từ DB
    ratings_data = WebAPIService.get_ratings()
    ratings_df = pd.DataFrame(ratings_data, columns=['userId', 'movieId', 'timestamp'])
    
    ratings_df['rating'] = 5
    
    TARGET_COLUMNS = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_df = ratings_df[TARGET_COLUMNS]
    
    # Lưu ra file
    ratings_df.to_csv(os.path.join(DB_DATA_PATH, 'db_ratings.csv'), index=False)

def convert_db_favourites_movies_to_csv():
    # Lấy dữ liệu từ DB
    favourites_data = WebAPIService.get_favourite_movies()
    favourites_df = pd.DataFrame(favourites_data, columns=['movieId', 'title', 'genres'])
    
    # Lưu ra file
    favourites_df.to_csv(os.path.join(DB_DATA_PATH, 'db_favourites_movies.csv'), index=False)
