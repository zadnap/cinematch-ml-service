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
    
    # Đọc các file cần thiết
    db_favourites_movies = pd.read_csv(os.path.join(DB_DATA_PATH, 'db_favourites_movies.csv'))
    db_ratings = pd.read_csv(os.path.join(DB_DATA_PATH, 'db_ratings.csv'))
    links_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'links.csv'))
    movies_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'))
    pos_interactions = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'positive_interactions.csv'))
    
    # ==========================================
    # BƯỚC 1 & 2: XỬ LÝ PHIM MỚI VÀ CẬP NHẬT MOVIE_FEATURES
    # ==========================================
    # Lấy các tmdbId từ DB (trong db_ratings bạn lưu nó ở cột movieId)
    db_tmdb_ids = db_ratings['movieId'].unique()
    existing_tmdb_ids = links_df['tmdbId'].dropna().values
    
    # Tìm các tmdbId chưa từng tồn tại trong hệ thống
    new_tmdb_ids = np.setdiff1d(db_tmdb_ids, existing_tmdb_ids)
    
    if len(new_tmdb_ids) > 0:
        print(f"Phát hiện {len(new_tmdb_ids)} phim mới từ DB. Đang cập nhật kho phim...")
        new_links = []
        new_movies = []
        
        # Khởi tạo ID gốc mới (tăng dần từ ID lớn nhất hiện tại)
        next_movie_id = movies_df['movieId'].max() + 1
        
        for tmdb_id in new_tmdb_ids:
            title = db_favourites_movies[db_favourites_movies['movieId'] == tmdb_id]['title'].values[0]
            genres = db_favourites_movies[db_favourites_movies['movieId'] == tmdb_id]['genres'].values[0]
            
            new_links.append({'movieId': next_movie_id, 'imdbId': '', 'tmdbId': tmdb_id})
            new_movies.append({'movieId': next_movie_id, 'title': title, 'genres': genres})
            
            next_movie_id += 1
            
        # Nối data mới và lưu lại
        links_df = pd.concat([links_df, pd.DataFrame(new_links)], ignore_index=True)
        movies_df = pd.concat([movies_df, pd.DataFrame(new_movies)], ignore_index=True)
        
        links_df.to_csv(os.path.join(RAW_DATA_PATH, 'links.csv'), index=False)
        movies_df.to_csv(os.path.join(RAW_DATA_PATH, 'movies.csv'), index=False)
        
    # Tạo lại từ điển map movieId thật sang movie_id (0 -> N-1)
    movie_ids = movies_df['movieId'].unique()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movies_df['movie_id'] = movies_df['movieId'].map(movie2movie_encoded)
    
    # Cập nhật lại toàn bộ movie_features.csv (One-hot encoding)
    genres_dummies = movies_df['genres'].str.get_dummies(sep='|')
    if '(no genres listed)' in genres_dummies.columns:
        genres_dummies.drop(columns=['(no genres listed)'], inplace=True)
    genres_dummies.columns = [col.lower().replace('-', '_').replace(' ', '_') for col in genres_dummies.columns]
    
    movies_df['year'] = movies_df['title'].apply(extract_year)
    movies_df['year'] = movies_df['year'].fillna(
        movies_df['year'].rolling(window=5, center=True, min_periods=1).mean()
    ).astype(int)
    
    movie_features = pd.concat([movies_df[['movie_id', 'year']], genres_dummies], axis=1)
    movie_features.to_csv(os.path.join(PROCESSED_DATA_PATH, 'movie_features.csv'), index=False)
    print("Cập nhật movie_features.csv thành công!")

    # ==========================================
    # BƯỚC 3: CẬP NHẬT POSITIVE_INTERACTIONS
    # ==========================================
    # Chuyển tmdbId (từ DB) sang internal movie_id (0 -> N-1)
    tmdb2real = dict(zip(links_df['tmdbId'], links_df['movieId']))
    
    db_ratings['real_movieId'] = db_ratings['movieId'].map(tmdb2real)
    db_ratings['internal_movie_id'] = db_ratings['real_movieId'].map(movie2movie_encoded)
    db_ratings = db_ratings.dropna(subset=['internal_movie_id'])
    
    # Đổi tên cột cho khớp với positive_interactions
    new_interactions = db_ratings[['userId', 'internal_movie_id', 'timestamp']].copy()
    new_interactions = new_interactions.rename(columns={'userId': 'user_id', 'internal_movie_id': 'movie_id'})
    new_interactions['movie_id'] = new_interactions['movie_id'].astype(int)
    
    # Gộp và xóa trùng lặp (nếu user đã đánh giá phim này rồi thì không thêm lại)
    pos_interactions = pd.concat([pos_interactions, new_interactions], ignore_index=True)
    pos_interactions = pos_interactions.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
    pos_interactions.to_csv(os.path.join(PROCESSED_DATA_PATH, 'positive_interactions.csv'), index=False)
    print("Cập nhật positive_interactions.csv thành công!")

    # ============================================
    # BƯỚC 4: CẬP NHẬT USER MỚI VÀO USER FEATURES
    # ============================================
    print("Đang chạy main_train: Gộp thêm dữ liệu db_user_features...")
    user_features = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'user_features.csv'))
    db_user_feature = pd.read_csv(os.path.join(DB_DATA_PATH, "db_user_features.csv"))

    user_features = pd.concat([user_features, db_user_feature], ignore_index=True)
    user_features = user_features.drop_duplicates(subset=['user_id'], keep='first')
    user_features = user_features.reset_index(drop=True)
    print(f"Đã gộp và xóa trùng lặp. Tổng số user hiện tại: {len(user_features)}")

    # ==========================================
    # BƯỚC 5: TÍNH TOÁN LẠI GENRES_AVG CHO USER
    # ==========================================
    # Tính TỔNG SỐ LƯỢT ĐÁNH GIÁ của mỗi user
    merged_df = pd.merge(pos_interactions[['user_id', 'movie_id']], movie_features, on='movie_id')
    total_ratings_per_user = merged_df.groupby('user_id')['movie_id'].count()
    
    # Chỉ lấy các cột thể loại (bỏ movie_id và year)
    genre_cols = [col for col in movie_features.columns if col not in ['movie_id', 'year']]
    
    # Set index cho user_features để mapping dữ liệu chuẩn xác
    user_features.set_index('user_id', inplace=True)
    
    for genre in genre_cols:
        genre_scores = merged_df[genre] * 5.0
        
        # Tính genres_avg cho từng user
        sum_genre_ratings = genre_scores.groupby(merged_df['user_id']).sum()
        custom_avg_series = sum_genre_ratings / total_ratings_per_user
        
        # Cập nhật vào user_features
        avg_col_name = genre + '_avg'
        if avg_col_name not in user_features.columns:
            user_features[avg_col_name] = 0.0
            
        user_features[avg_col_name] = custom_avg_series
        user_features[avg_col_name] = user_features[avg_col_name].fillna(0.0)
        
    user_features.reset_index(inplace=True)
    user_features.to_csv(os.path.join(PROCESSED_DATA_PATH, 'user_features.csv'), index=False)
    print("Cập nhật thành công genres_avg cho user_features.csv!")