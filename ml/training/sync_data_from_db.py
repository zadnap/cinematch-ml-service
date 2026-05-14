import os
import pandas as pd
import numpy as np
import re

from ml.utils.convert_to_csv import (
    DB_DATA_PATH,
    convert_db_ratings_to_csv,
    convert_db_user_features_to_csv,
    convert_db_favourites_movies_to_csv
)

PROCESSED_DATA_PATH = "ml/data/processed_data"
RAW_DATA_PATH = "ml/data/raw_data"

def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None

convert_db_user_features_to_csv()
convert_db_ratings_to_csv()
convert_db_favourites_movies_to_csv()

def sync_database_to_training_data():
    print("Bắt đầu đồng bộ dữ liệu DB...")

    # ==========================================
    # LOAD DATA
    # ==========================================
    db_favourites_movies = pd.read_csv(
        os.path.join(DB_DATA_PATH, 'db_favourites_movies.csv')
    )

    db_ratings = pd.read_csv(
        os.path.join(DB_DATA_PATH, 'db_ratings.csv')
    )

    links_df = pd.read_csv(
        os.path.join(RAW_DATA_PATH, 'links.csv')
    )

    movies_df = pd.read_csv(
        os.path.join(RAW_DATA_PATH, 'movies.csv')
    )

    pos_interactions = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH, 'positive_interactions.csv')
    )

    # ==========================================
    # CLEAN DUPLICATES
    # ==========================================
    print("Đang cleanup duplicate data...")

    links_df = links_df.drop_duplicates(
        subset=['tmdbId'],
        keep='first'
    ).reset_index(drop=True)

    movies_df = movies_df.drop_duplicates(
        subset=['movieId'],
        keep='first'
    ).reset_index(drop=True)

    print(f"Links count: {len(links_df)}")
    print(f"Movies count: {len(movies_df)}")

    # ==========================================
    # XỬ LÝ PHIM MỚI
    # ==========================================
    db_tmdb_ids = set(db_ratings['movieId'].unique())
    existing_tmdb_ids = set(links_df['tmdbId'].dropna())

    new_tmdb_ids = [
        tmdb_id
        for tmdb_id in db_tmdb_ids
        if tmdb_id not in existing_tmdb_ids
    ]

    if len(new_tmdb_ids) > 0:
        print(f"Phát hiện {len(new_tmdb_ids)} phim mới.")

        new_links = []
        new_movies = []

        next_movie_id = int(movies_df['movieId'].max()) + 1

        for tmdb_id in new_tmdb_ids:

            movie_row = db_favourites_movies[
                db_favourites_movies['movieId'] == tmdb_id
            ]

            # Nếu không tìm thấy metadata phim
            if movie_row.empty:
                print(f"Không tìm thấy metadata cho TMDB ID: {tmdb_id}")
                continue

            title = movie_row.iloc[0]['title']
            genres = movie_row.iloc[0]['genres']

            new_links.append({
                'movieId': next_movie_id,
                'imdbId': '',
                'tmdbId': tmdb_id
            })

            new_movies.append({
                'movieId': next_movie_id,
                'title': title,
                'genres': genres
            })

            next_movie_id += 1

        # Append phim mới
        if len(new_links) > 0:
            links_df = pd.concat(
                [links_df, pd.DataFrame(new_links)],
                ignore_index=True
            )

            movies_df = pd.concat(
                [movies_df, pd.DataFrame(new_movies)],
                ignore_index=True
            )

    # ==========================================
    # CLEAN AGAIN AFTER APPEND
    # ==========================================
    links_df = links_df.drop_duplicates(
        subset=['tmdbId'],
        keep='first'
    ).reset_index(drop=True)

    movies_df = movies_df.drop_duplicates(
        subset=['movieId'],
        keep='first'
    ).reset_index(drop=True)

    # ==========================================
    # SAVE CLEAN RAW DATA
    # ==========================================
    links_df.to_csv(
        os.path.join(RAW_DATA_PATH, 'links.csv'),
        index=False
    )

    movies_df.to_csv(
        os.path.join(RAW_DATA_PATH, 'movies.csv'),
        index=False
    )

    print("Đã cập nhật links.csv và movies.csv")

    # ==========================================
    # ENCODE MOVIE IDs
    # ==========================================
    movie_ids = movies_df['movieId'].unique()

    movie2movie_encoded = {
        x: i for i, x in enumerate(movie_ids)
    }

    movies_df['movie_id'] = movies_df['movieId'].map(
        movie2movie_encoded
    )

    # ==========================================
    # CREATE MOVIE FEATURES
    # ==========================================
    genres_dummies = movies_df['genres'].str.get_dummies(sep='|')

    if '(no genres listed)' in genres_dummies.columns:
        genres_dummies.drop(
            columns=['(no genres listed)'],
            inplace=True
        )

    genres_dummies.columns = [
        col.lower()
        .replace('-', '_')
        .replace(' ', '_')
        for col in genres_dummies.columns
    ]

    movies_df['year'] = movies_df['title'].apply(extract_year)

    movies_df['year'] = movies_df['year'].fillna(
        movies_df['year']
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    ).astype(int)

    movie_features = pd.concat(
        [movies_df[['movie_id', 'year']], genres_dummies],
        axis=1
    )

    movie_features.to_csv(
        os.path.join(PROCESSED_DATA_PATH, 'movie_features.csv'),
        index=False
    )

    print("Cập nhật movie_features.csv thành công!")

    # ==========================================
    # UPDATE POSITIVE INTERACTIONS
    # ==========================================
    tmdb2real = dict(
        zip(links_df['tmdbId'], links_df['movieId'])
    )

    db_ratings['real_movieId'] = db_ratings['movieId'].map(
        tmdb2real
    )

    db_ratings['internal_movie_id'] = db_ratings[
        'real_movieId'
    ].map(movie2movie_encoded)

    db_ratings = db_ratings.dropna(
        subset=['internal_movie_id']
    )

    new_interactions = db_ratings[
        ['userId', 'internal_movie_id', 'timestamp']
    ].copy()

    new_interactions = new_interactions.rename(columns={
        'userId': 'user_id',
        'internal_movie_id': 'movie_id'
    })

    new_interactions['movie_id'] = (
        new_interactions['movie_id'].astype(int)
    )

    pos_interactions = pd.concat(
        [pos_interactions, new_interactions],
        ignore_index=True
    )

    pos_interactions = pos_interactions.drop_duplicates(
        subset=['user_id', 'movie_id'],
        keep='last'
    )

    pos_interactions.to_csv(
        os.path.join(PROCESSED_DATA_PATH,
        'positive_interactions.csv'),
        index=False
    )

    print("Cập nhật positive_interactions.csv thành công!")

    # ==========================================
    # UPDATE USER FEATURES
    # ==========================================
    print("Đang cập nhật user_features...")

    user_features = pd.read_csv(
        os.path.join(PROCESSED_DATA_PATH,
        'user_features.csv')
    )

    db_user_feature = pd.read_csv(
        os.path.join(DB_DATA_PATH,
        "db_user_features.csv")
    )

    user_features = pd.concat(
        [user_features, db_user_feature],
        ignore_index=True
    )

    user_features = user_features.drop_duplicates(
        subset=['user_id'],
        keep='first'
    )

    user_features = user_features.reset_index(drop=True)

    print(f"Tổng số user: {len(user_features)}")

    # ==========================================
    # RECALCULATE GENRE AVERAGES
    # ==========================================
    merged_df = pd.merge(
        pos_interactions[['user_id', 'movie_id']],
        movie_features,
        on='movie_id'
    )

    total_ratings_per_user = merged_df.groupby(
        'user_id'
    )['movie_id'].count()

    genre_cols = [
        col for col in movie_features.columns
        if col not in ['movie_id', 'year']
    ]

    user_features.set_index('user_id', inplace=True)

    for genre in genre_cols:

        genre_scores = merged_df[genre] * 5.0

        sum_genre_ratings = genre_scores.groupby(
            merged_df['user_id']
        ).sum()

        custom_avg_series = (
            sum_genre_ratings / total_ratings_per_user
        )

        avg_col_name = genre + '_avg'

        if avg_col_name not in user_features.columns:
            user_features[avg_col_name] = 0.0

        user_features[avg_col_name] = custom_avg_series

        user_features[avg_col_name] = (
            user_features[avg_col_name]
            .fillna(0.0)
        )

    user_features.reset_index(inplace=True)

    user_features.to_csv(
        os.path.join(PROCESSED_DATA_PATH,
        'user_features.csv'),
        index=False
    )

    print("Cập nhật genres_avg thành công!")