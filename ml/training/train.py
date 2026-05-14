import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping

# --- LẤY DỮ LIỆU TỪ DATABASE ---
from ml.training.sync_data_from_db import (
    sync_database_to_training_data
)

print("=== BƯỚC 1: ĐỒNG BỘ DỮ LIỆU TỪ DB ===")
sync_database_to_training_data()
print("Đã cập nhật xong các file CSV tĩnh (movies.csv, user_features.csv, ...)")

# --- LẤY DỮ LIỆU TỪ FILE DATASET_LOADER ---
from ml.training.dataset_loader import (
    train_dataset, val_dataset, test_dataset, 
    test_data, NUM_MOVIES,
    user_features_df,
    PROCESSED_DATA_PATH
)

# --- LẤY MÔ HÌNH TỪ FILE RECOMMENDER_MODEL ---
from ml.models.two_towers_model import model

ARTIFACTS_PATH = "ml/artifacts"

checkpoint = ModelCheckpoint(
    os.path.join(ARTIFACTS_PATH, 'two_tower_best_weights.weights.h5'), 
    monitor='val_loss', 
    mode='min', 
    save_best_only=True,
    save_weights_only=True
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, # Dừng nếu sau 3 epochs val loss không tăng
    restore_best_weights=True
)

print("Bắt đầu huấn luyện mô hình TFRS...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)

print("Training finished! Testing model...")
model.evaluate(test_dataset)

user_features_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "user_features.csv"), index=False)
    
print(f"Đã cập nhật file {os.path.join(PROCESSED_DATA_PATH, 'user_features.csv')} thành công!")
print(f"Tổng số User hiện tại hệ thống nhận diện được: {len(user_features_df)}")

if __name__ == "__main__":
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(history.history['loss']) + 1)
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss')

    plt.title('Sự sụt giảm của Hàm mất mát (Loss) qua các Epoch', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Categorical Crossentropy Loss', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()

    # ==========================================
    # TÍNH TOÁN TOP-K ACCURACY THỦ CÔNG (HIT RATE)
    # ==========================================

    # Tính toán trước Vector của toàn bộ kho phim
    print("Đang tạo Vector cho kho phim...")
    all_movie_ids = np.arange(NUM_MOVIES)
    all_movie_vectors = model.movie_model.predict(all_movie_ids, batch_size=4096, verbose=0)

    # Cài đặt các tham số kiểm tra
    K = [10, 100]
    hits = 0
    total_users_to_test = len(test_data)

    print(f"Đang kiểm tra Top-{K} thủ công cho {total_users_to_test} tương tác...")
    test_user_ids = test_data['user_id'].values
    test_movie_ids = test_data['movie_id'].values
    test_user_vectors = model.user_model.predict(test_user_ids, batch_size=4096, verbose=0)

    # test_user_and_movie_vector_matrix = np.dot(test_user_vectors, all_movie_vectors.T)

    # for K_value in K:
    #     hits = 0
    #     print(f'Đang kiểm tra Top-{K_value}...')
    #     for i in range(total_users_to_test):
    #         actual_movie_id = test_movie_ids[i] # Bộ phim thực tế User đã xem trong tập Test

    #         scores = test_user_and_movie_vector_matrix[i]
    #         top_k_movie_ids = np.argpartition(scores, -K_value)[-K_value:]
    #         if actual_movie_id in top_k_movie_ids:
    #             hits += 1
            
    #     manual_top_k_accuracy = (hits / total_users_to_test) * 100
    #     print(f"\n✅ Kết quả tính toán thủ công:")
    #     print(f"Tổng số lần đoán trúng (Hits) : {hits}")
    #     print(f"Tổng số tương tác đã test     : {total_users_to_test}")
    #     print(f"-> Manual Recall@{K_value}          : {manual_top_k_accuracy:.2f}%")

    # Vòng lặp kiểm tra từng tương tác
    for K_value in K:
        hits = 0
        print(f"Đang kiểm tra Top-{K_value}...")
        for i in range(total_users_to_test):
            user_vec = test_user_vectors[i]
            actual_movie_id = test_movie_ids[i] # Bộ phim thực tế User đã xem trong tập Test

            # Tính điểm số Cosine giữa User này và TOÀN BỘ phim
            scores = np.dot(all_movie_vectors, user_vec)
            top_k_movie_ids = np.argpartition(scores, -K_value)[-K_value:]
            
            # Kiểm tra bộ phim thực tế có lọt vào Top K dự đoán không
            if actual_movie_id in top_k_movie_ids:
                hits += 1
            
        manual_top_k_accuracy = (hits / total_users_to_test) * 100

        print(f"\n✅ Kết quả tính toán thủ công:")
        print(f"Tổng số lần đoán trúng (Hits) : {hits}")
        print(f"Tổng số tương tác đã test     : {total_users_to_test}")
        print(f"-> Manual Recall@{K_value}          : {manual_top_k_accuracy:.2f}%")