import numpy as np
from abc import ABC, abstractmethod
from ml.utils.map_to_tmdb_id import map_to_tmdb_id
from ml.inference.recommender_engine import REC_SYS
from ml.services.web_api_service import WebAPIService

def _get_best_k_films_by_genres(user_id, user_features, is_cold_start, k = 10):
    if not is_cold_start:
        user_row = REC_SYS.user_features_df[REC_SYS.user_features_df['user_id'] == user_id + WebAPIService.ID_OFFSET].iloc[0]
        genre_columns = [col for col in user_row.index if col.endswith('_avg')]
        liked_genres = [col.replace('_avg', '') for col in genre_columns if user_row[col] > 3.8]
    else:
        genre_cols = [col.replace('_avg', '') for col in REC_SYS.user_features_df.columns[1:]]
        liked_genres = [genre for genre, score in zip(genre_cols, user_features) if score > 3.8]
    
    valid_genres = [genre for genre in liked_genres if genre in REC_SYS.movie_features_df.columns]
    if not valid_genres:
        return REC_SYS.movie_scores.head(k)["movieId"].tolist()
    
    mask = REC_SYS.movie_features_df[valid_genres].sum(axis=1) > 0
    matching_encoded_ids_set = set(REC_SYS.movie_features_df.loc[mask, 'movie_id'].tolist())

    filtered_movies = REC_SYS.movie_scores[REC_SYS.movie_scores['encoded_id'].isin(matching_encoded_ids_set)]
    return filtered_movies.head(k)["movieId"].tolist()


def _get_top_k_two_tower_recommendations(target_user_id, k = 10):    
    user_vector = REC_SYS.model.user_model.predict(np.array([target_user_id], dtype=np.int32), verbose=0)
    scores = np.dot(REC_SYS.movie_vectors, user_vector.T).flatten()

    watched_movie_ids = REC_SYS.interactions[REC_SYS.interactions['user_id'] == target_user_id]['movie_id'].values
    scores[watched_movie_ids] = -999.0
    
    top_k_encoded_ids = np.argsort(scores)[::-1][:k]
    return [REC_SYS.encoded_to_real_movie_id[idx] for idx in top_k_encoded_ids if idx in REC_SYS.encoded_to_real_movie_id]


class RecommendationStrategy(ABC):
    @abstractmethod
    def recommend(self, user_id, user_features, is_cold_start: bool, top_k):
        pass


class ColdStartRecommendationStrategy(RecommendationStrategy):
    def recommend(self, user_id, user_features, is_cold_start: bool, top_k):
        print(f"[Strategy] Đang thực hiện đề xuất cold-start cho User ID {user_id}...")
        best_k_film_ids = _get_best_k_films_by_genres(user_id, user_features, is_cold_start, k=top_k)
        return map_to_tmdb_id(best_k_film_ids)


class NormalRecommendationStrategy(RecommendationStrategy):
    def __init__(self, ratio: float = 0.85):
        self.ratio = ratio

    def recommend(self, user_id, user_features, is_cold_start: bool, top_k):
        print(f"[Strategy] Đang thực hiện đề xuất thông thường cho User ID {user_id}...")
        num_from_model = int(top_k * self.ratio)
        num_from_niche = top_k - num_from_model

        # 1. Lấy đề xuất từ Mô hình học sâu (Two-Tower)
        model_real_ids = _get_top_k_two_tower_recommendations(user_id, k=num_from_model + 20)

        # 2. Lấy đề xuất từ Lọc theo danh mục ngách (Niche) và lọc trùng
        niche_real_ids_raw = _get_best_k_films_by_genres(user_id, user_features, is_cold_start, k=num_from_niche * 3)
        unique_niche_ids = [n_id for n_id in niche_real_ids_raw if n_id not in model_real_ids][:num_from_niche]
        
        # 3. Trộn đan xen (Interleaving logic sạch, tránh vòng lặp vô hạn)
        final_real_ids = []
        m_idx, n_idx = 0, 0
        interval = max(1, num_from_model // len(unique_niche_ids)) if unique_niche_ids else num_from_model
        
        while m_idx < num_from_model or n_idx < len(unique_niche_ids):
            # Thêm cụm từ mô hình ML
            for _ in range(interval):
                if m_idx < len(model_real_ids) and m_idx < num_from_model:
                    final_real_ids.append(model_real_ids[m_idx])
                    m_idx += 1
            # Xen kẽ 1 phim từ danh mục ngách
            if n_idx < len(unique_niche_ids):
                final_real_ids.append(unique_niche_ids[n_idx])
                n_idx += 1
        
        return map_to_tmdb_id(final_real_ids[:top_k])


class RecommendationContext:
    def __init__(self, strategy: RecommendationStrategy = None):
        self._strategy = strategy

    @property
    def strategy(self) -> RecommendationStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: RecommendationStrategy):
        self._strategy = strategy

    def execute_recommendation(self, user_id, user_features, is_cold_start, top_k):
        if not self._strategy:
            raise ValueError("Chiến lược gợi ý chưa được thiết lập!")
        return self._strategy.recommend(user_id, user_features, is_cold_start, top_k)


def recommend_movies(user_id, user_features, top_k):
    is_cold_start = (user_id + WebAPIService.ID_OFFSET) not in REC_SYS.user_features_df['user_id'].values
    strategy = ColdStartRecommendationStrategy() if is_cold_start else NormalRecommendationStrategy(ratio=0.85)
    recommender = RecommendationContext(strategy)
        
    return recommender.execute_recommendation(user_id, user_features, is_cold_start, top_k)