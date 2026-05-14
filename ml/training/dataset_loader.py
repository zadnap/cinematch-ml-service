import os
import pandas as pd
import numpy as np
import tensorflow as tf

PROCESSED_DATA_PATH = "ml/data/processed_data"
RAW_DATA_PATH = "ml/data/raw_data"

interactions = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "positive_interactions.csv"))
user_features_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "user_features.csv"))
movie_features_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "movie_features.csv"))

print(interactions.shape)
print(user_features_df.shape)
print(movie_features_df.shape)

NUM_USERS = user_features_df['user_id'].max() + 1
NUM_MOVIES = movie_features_df['movie_id'].max() + 1

print(f'Number of users: {NUM_USERS}')
print(f'Number of movies: {NUM_MOVIES}')

user_features_df = user_features_df.sort_values('user_id')
movie_features_df = movie_features_df.sort_values('movie_id')

user_features_matrix = user_features_df.drop('user_id', axis=1).values.astype(np.float32)
movie_features_matrix = movie_features_df.drop('movie_id', axis=1).values.astype(np.float32)

print(f'User features matrix shape: {user_features_matrix.shape}')
print(f'Movie features matrix shape: {movie_features_matrix.shape}')

NUM_USER_FEATURES = user_features_matrix.shape[1]
NUM_MOVIE_FEATURES = movie_features_matrix.shape[1]

interactions = interactions.sort_values('timestamp')

n_total = len(interactions)
train_size = int(n_total * 0.80)
val_size = int(n_total * 0.15)

train_data = interactions.iloc[:train_size]
val_data = interactions.iloc[train_size:train_size + val_size]
test_data = interactions.iloc[train_size + val_size:]

BATCH_SIZE = 4096

# In-batch negative sampling
def build_dataset(df, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices({
        'user_input': df['user_id'].values,
        'movie_input': df['movie_id'].values
    })

    if is_train:
        dataset = dataset.shuffle(buffer_size=100000)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
    return dataset.prefetch(tf.data.AUTOTUNE)

train_dataset = build_dataset(train_data, is_train=True)
val_dataset = build_dataset(val_data, is_train=False)
test_dataset = build_dataset(test_data, is_train=False)

# Lấy ra danh sách toàn bộ các ID phim duy nhất có trong hệ thống
unique_movie_ids = np.unique(movie_features_df['movie_id'].values)
movie_dataset = tf.data.Dataset.from_tensor_slices(unique_movie_ids)