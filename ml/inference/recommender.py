import numpy as np
import pandas as pd
import os
from ml.models.two_towers_model import (
    TwoTowerModel, user_model, movie_model
)

from ml.training.dataset_loader import (
    NUM_MOVIES, NUM_USERS,
    movie_dataset,
    interactions,
    user_features_df,
    movie_features_df
)

from ml.training.data_preprocessing import (
    movie2movie_encoded, movies_df,
)

ARTIFACTS_PATH = "ml/artifacts"
PROCESSED_DATA_PATH = "ml/data/processed_data"
RAW_DATA_PATH = "ml/data/raw_data"

print("Đang load trọng số mô hình Two-Tower...")
GLOBAL_MODEL = TwoTowerModel(user_model, movie_model, movie_dataset)
GLOBAL_MODEL.load_weights(os.path.join(ARTIFACTS_PATH, "two_tower_best_weights.weights.h5"))

print("Đang pre-compute vector cho toàn bộ hệ thống phim...")
all_movie_ids = np.arange(NUM_MOVIES)

GLOBAL_MOVIE_VECTORS = GLOBAL_MODEL.movie_model.predict(all_movie_ids, batch_size=4096, verbose=1)
print("Khởi tạo hệ thống thành công! Đã sẵn sàng nhận request.")

GLOBAL_MOVIE_SCORES = pd.read_csv(
    os.path.join(
        PROCESSED_DATA_PATH,
        "global_movie_scores.csv"
    )
)

def get_best_k_films_for_user(user_id, k=10):
    liked_genres_set = set()
    # Kiểm tra xem user_id có trong bảng user features chưa
    user_exists = (user_features_df['user_id'] == user_id + NUM_USERS - 1).any()
    
    if user_exists:
        user_row = user_features_df[user_features_df['user_id'] == user_id + NUM_USERS - 1].iloc[0]
        genre_columns = [col for col in user_row.index if col.endswith('_avg')]
        user_genres_avg = user_row[genre_columns]
        liked_genres = user_genres_avg[user_genres_avg > 3.8].index.str.replace('_avg', '').tolist()
        liked_genres_set = set(liked_genres)
    else:
        # TODO: Fetch data from web service API
        # user_features = TrainingDataService.get_user_features(user_id)[1:]  # Bỏ user_id
        user_features = [0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
        genre_cols = user_features_df.columns[1:].str.replace('_avg', '')
        liked_genres = [
            genre for genre, score in zip(genre_cols, user_features) 
            if score > 3.8
        ]
        liked_genres_set = set(liked_genres)
    
    valid_genres = [genre for genre in liked_genres_set if genre in movie_features_df.columns]
    
    if not valid_genres:
        return GLOBAL_MOVIE_SCORES.head(k)["movieId"].tolist()
    
    mask = movie_features_df[valid_genres].sum(axis=1) > 0
    matching_encoded_ids_set = set(movie_features_df.loc[mask, 'movie_id'].tolist())

    filtered_movies = GLOBAL_MOVIE_SCORES[GLOBAL_MOVIE_SCORES['encoded_id'].isin(matching_encoded_ids_set)]
    final_result = filtered_movies.head(k)
    
    return final_result["movieId"].tolist()

def get_top_k_recommendations(model, target_user_id, all_movie_vectors, k=10):    
    user_vector = model.user_model.predict(
        np.array([target_user_id], dtype=np.int32),
        verbose=0
    )
    
    scores = np.dot(all_movie_vectors, user_vector.T).flatten()

    # Lọc bỏ các phim User đã xem/đánh giá
    watched_movie_ids = interactions[interactions['user_id'] == target_user_id]['movie_id'].values
    scores[watched_movie_ids] = -999.0
    
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_scores = scores[top_k_indices]
    
    return top_k_indices, top_k_scores

def mapping_movie_id_to_tmdb_id(movie_ids):
    links_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "links.csv"))
    result = (
        links_df[links_df["movieId"].isin(movie_ids)]
        .set_index("movieId")
        .loc[movie_ids, "tmdbId"]
        .dropna()
        .astype(int)
        .tolist()
    )
    return result

def cold_start_recommendation(user_id, top_k=10):
    print(f"Đang thực hiện đề xuất cold-start cho User ID {user_id}...")
    # Lấy top K phim phổ biến nhất
    best_k_film_ids = get_best_k_films_for_user(user_id, k=top_k)
    top_tmdb_ids = mapping_movie_id_to_tmdb_id(best_k_film_ids)
    return top_tmdb_ids, best_k_film_ids, []

def normal_recommendation(user_id, top_k=10, ratio=0.85):
    ratio_model = ratio
    num_from_model = int(top_k * ratio_model)
    num_from_niche = top_k - num_from_model

    top_k_encoded_ids, top_k_scores = get_top_k_recommendations(
        GLOBAL_MODEL,
        user_id,
        GLOBAL_MOVIE_VECTORS,
        k=num_from_model + 20
    )
    encoded2real_movie_id = {encoded_val: real_id for real_id, encoded_val in movie2movie_encoded.items()}
    model_real_ids = [encoded2real_movie_id[encoded_id] for encoded_id in top_k_encoded_ids]

    niche_real_ids_raw = get_best_k_films_for_user(user_id, k=num_from_niche * 3)

    top_tmdb_ids = []
    unique_niche_ids = []

    for n_id in niche_real_ids_raw:
        if n_id not in model_real_ids and len(unique_niche_ids) < num_from_niche:
            unique_niche_ids.append(n_id)
    
    interval = max(1, num_from_model // len(unique_niche_ids)) if unique_niche_ids else 1
    
    final_real_ids = []
    m_idx, n_idx = 0, 0
    
    while m_idx < num_from_model or n_idx < len(unique_niche_ids):
        for _ in range(interval):
            if m_idx < num_from_model:
                final_real_ids.append(model_real_ids[m_idx])
                m_idx += 1
                
        if n_idx < len(unique_niche_ids):
            final_real_ids.append(unique_niche_ids[n_idx])
            n_idx += 1
    
    final_real_ids = final_real_ids[:top_k]
    top_tmdb_ids = mapping_movie_id_to_tmdb_id(final_real_ids)
    return top_tmdb_ids, final_real_ids, model_real_ids[:num_from_model]

def recommend_movies(user_id, top_k=200):
    is_cold_start = False

    if user_id + NUM_USERS - 1 not in user_features_df['user_id'].values:
        print(f"User ID {user_id} không tồn tại trong dữ liệu người dùng. Sử dụng phương pháp đề xuất cold-start.")
        is_cold_start = True
        top_tmdb_ids, final_real_ids, model_real_ids = cold_start_recommendation(user_id, top_k)
    else:
        ratio = 0.85
        top_tmdb_ids, final_real_ids, model_real_ids = normal_recommendation(user_id, top_k, ratio)
    
    if __name__ == "__main__":
        print(f"\nTOP {top_k} PHIM DÀNH CHO USER {user_id}:")

        for i, real_movie_id in enumerate(final_real_ids):
            movie_info = movies_df[movies_df['movieId'] == real_movie_id]
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
            
            # Gắn tag phân loại linh hoạt dựa trên nguồn
            if is_cold_start:
                source = "Cold-start"
            elif real_movie_id in model_real_ids:
                source = "Model"
            else:
                source = "Niche"
                
            print(f"{i+1:2d}. [{source:<10}] {title[:40]:<42} | Thể loại: {genres}")
    return top_tmdb_ids


if __name__ == "__main__":
    target_user_id = 3
    top_recommend = recommend_movies(target_user_id, top_k=10)
    print("\nDanh sách TMDB IDs của các phim được đề xuất:")
    print(top_recommend)
