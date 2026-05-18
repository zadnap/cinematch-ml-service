import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Flatten,
    Input,
    Embedding,
    Dense,
    Dropout,
    Concatenate,
    UnitNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler

# --- LẤY BIẾN TỪ FILE DATASET_LOADER ---
from ml.training.dataset_loader import (
    NUM_USERS, NUM_MOVIES, 
    NUM_USER_FEATURES, NUM_MOVIE_FEATURES,
    user_features_matrix, movie_features_matrix, 
    movie_features_df, movie_dataset
)

EMBEDDING_DIM = 64
user_input = Input(shape=(1,), name='user_input', dtype=tf.int32)
movie_input = Input(shape=(1,), name='movie_input', dtype=tf.int32)
scaler = StandardScaler()

# Tạo embedding cho user_id
user_id_emb = Embedding(
    input_dim=NUM_USERS,
    output_dim=EMBEDDING_DIM,
    embeddings_initializer='he_normal',
    embeddings_regularizer=l2(1e-4),
    name='user_id_emd'
)(user_input)

# Tạo embedding cho user_features
user_features_matrix_scaled = scaler.fit_transform(user_features_matrix)
user_feat_layer = Embedding(
    input_dim=NUM_USERS,
    output_dim=NUM_USER_FEATURES,
    weights=[user_features_matrix_scaled],
    trainable=False,
    name='user_feat_emd'
)(user_input)

user_id_emb = Flatten()(user_id_emb)
user_feat_layer = Flatten()(user_feat_layer)

user_concat = Concatenate()([user_id_emb, user_feat_layer])
user_tower = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(user_concat)
user_tower = Dropout(0.4)(user_tower)
user_tower = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(user_tower)
user_tower = Dropout(0.3)(user_tower)
user_tower = Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(user_tower)
user_tower = Dropout(0.2)(user_tower)
user_vector = Dense(64, activation=None, name='user_vector')(user_tower)
user_vector = UnitNormalization(axis=1)(user_vector)

# Tạo embedding cho movie_id
movie_id_emb = Embedding(
    input_dim=NUM_MOVIES,
    output_dim=EMBEDDING_DIM,
    embeddings_initializer='he_normal',
    embeddings_regularizer=l2(1e-4),
    name='movie_id_emd'
)(movie_input)

# Tạo embedding cho movie_features
movie_features_matrix_scaled = scaler.fit_transform(movie_features_matrix)
movie_feat_layer = Embedding(
    input_dim=NUM_MOVIES,
    output_dim=NUM_MOVIE_FEATURES,
    weights=[movie_features_matrix_scaled],
    trainable=False,
    name='movie_feat_emd'
)(movie_input)

movie_id_emb = Flatten()(movie_id_emb)
movie_feat_layer = Flatten()(movie_feat_layer)

movie_concat = Concatenate()([movie_id_emb, movie_feat_layer])
movie_tower = Dense(256, activation='relu',kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(movie_concat)
movie_tower = Dropout(0.4)(movie_tower)
movie_tower = Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(movie_tower)
movie_tower = Dropout(0.3)(movie_tower)
movie_tower = Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(movie_tower)
movie_tower = Dropout(0.2)(movie_tower)
movie_vector = Dense(64, activation=None, name='movie_vector')(movie_tower)
movie_vector = UnitNormalization(axis=1)(movie_vector)

# Lấy ra danh sách toàn bộ các ID phim duy nhất có trong hệ thống
unique_movie_ids = np.unique(movie_features_df['movie_id'].values)

movie_dataset = tf.data.Dataset.from_tensor_slices(unique_movie_ids)

user_model = Model(inputs=user_input, outputs=user_vector)
movie_model = Model(inputs=movie_input, outputs=movie_vector)

class TwoTowerModel(tfrs.Model):
    def __init__(self, user_model, movie_model, movie_dataset):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        
        # Thêm đo lường Top-K Accuracy
        metrics = tfrs.metrics.FactorizedTopK(
            candidates=movie_dataset.batch(1024).map(self.movie_model),
            ks=[10, 100]
        )
        
        self.task = tfrs.tasks.Retrieval(
            # metrics=metrics,
            temperature=0.05
        )

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_input"])
        movie_embeddings = self.movie_model(features["movie_input"])
        
        # Đưa vào Task để tính Loss
        return self.task(user_embeddings, movie_embeddings)

model = TwoTowerModel(user_model, movie_model, movie_dataset)

model.compile(optimizer=Adam(learning_rate=0.001))