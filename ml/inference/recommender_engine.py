import os
import numpy as np
import pandas as pd
import tensorflow as tf
from ml.models.two_towers_model import TwoTowerModel, user_model, movie_model
from ml.training.data_preprocessor import preprocessor
from ml.constants import ARTIFACTS_PATH, PROCESSED_DATA_PATH

class RecommenderEngine:
    def __init__(self):
        print("Đang tải dữ liệu đặc trưng phim trước...")
        self.movie_features_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "movie_features.csv"))
        num_movies = self.movie_features_df['movie_id'].max() + 1
        
        print("Đang khởi tạo cấu trúc hệ thống gợi ý...")
        unique_movie_ids = np.arange(num_movies)
        local_movie_dataset = tf.data.Dataset.from_tensor_slices(unique_movie_ids)

        self.model = TwoTowerModel(user_model, movie_model, local_movie_dataset)
        self.user_model = user_model
        self.movie_model = movie_model

        print("Sinh dữ liệu giả lập độc lập cho từng tháp...")
        dummy_id = np.array([0, 1], dtype=np.int32)
        _ = self.user_model(dummy_id)
        _ = self.movie_model(dummy_id)

        print("Đang load trọng số trực tiếp vào các Layer tương ứng...")
        weights_file = os.path.join(ARTIFACTS_PATH, "two_tower_best_weights.weights.h5")
        self.user_model.load_weights(weights_file, by_name=True, skip_mismatch=True)
        self.movie_model.load_weights(weights_file, by_name=True, skip_mismatch=True)

        print("Đang pre-compute vector cho toàn bộ hệ thống phim...")
        self.movie_vectors = self.movie_model.predict(unique_movie_ids, batch_size=4096, verbose=1)

        print("Đang tải các dữ liệu tương tác còn lại...")
        self.interactions = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "positive_interactions.csv"))
        self.user_features_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "user_features.csv"))
        self.movie_scores = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "global_movie_scores.csv"))

        success = preprocessor.load_mappings()
        if not success:
            preprocessor.run_preprocessing()
        self.encoded_to_real_movie_id = {
            encoded_val: real_id for real_id, encoded_val in preprocessor.movie2movie_encoded.items()
        }
        
        print("Khởi tạo hệ thống thành công! Đã sẵn sàng nhận request.")

REC_SYS = RecommenderEngine()