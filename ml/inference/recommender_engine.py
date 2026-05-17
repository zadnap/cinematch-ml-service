import os
import numpy as np
import pandas as pd
from ml.models.two_towers_model import TwoTowerModel, user_model, movie_model
from ml.training.dataset_loader import NUM_MOVIES, movie_dataset
from ml.constants import ARTIFACTS_PATH, PROCESSED_DATA_PATH

class MovieRecommenderSystem:
    def __init__(self):
        print("Đang khởi tạo cấu trúc hệ thống gợi ý...")
        self.model = TwoTowerModel(user_model, movie_model, movie_dataset)
        
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
        all_movie_ids = np.arange(NUM_MOVIES)
        self.movie_vectors = self.movie_model.predict(all_movie_ids, batch_size=4096, verbose=1)

        print("Đang tải dữ liệu metadata...")
        self.movie_scores = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "global_movie_scores.csv"))
        print("Khởi tạo hệ thống thành công! Đã sẵn sàng nhận request.")

REC_SYS = MovieRecommenderSystem()