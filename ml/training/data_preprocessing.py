import os
import pandas as pd
import re

PROCESSED_DATA_PATH = "ml/data/processed_data"
RAW_DATA_PATH = "ml/data/raw_data"

def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)

    if match:
        return int(match.group(1))

    return None


def export_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Đã xuất file: {filename}")


def prepare_global_movie_scores(ratings_df, movies_df, movie2movie_encoded):
    rating_counts = (
        ratings_df
        .groupby('movieId')
        .size()
        .reset_index(name='num_ratings')
    )

    ratings_avg = (
        ratings_df
        .groupby('movieId')['rating']
        .mean()
        .reset_index(name='avg_rating')
    )

    movie_scores_df = pd.merge(
        rating_counts,
        movies_df,
        on='movieId'
    )

    movie_scores_df = pd.merge(
        ratings_avg,
        movie_scores_df,
        on='movieId'
    )

    movie_scores_df['score'] = (
        movie_scores_df['num_ratings']
        * movie_scores_df['avg_rating']
    )

    movie_scores_df['encoded_id'] = (
        movie_scores_df['movieId']
        .map(movie2movie_encoded)
    )

    movie_scores_df = movie_scores_df.sort_values(
        by='score',
        ascending=False
    )

    return movie_scores_df


movies_df = pd.read_csv(
    os.path.join(RAW_DATA_PATH, "movies.csv")
)

ratings_df = pd.read_csv(
    os.path.join(RAW_DATA_PATH, "ratings.csv")
)

movie2movie_encoded = {
    x: i for i, x in enumerate(movies_df['movieId'].unique())
}


if __name__ == "__main__":

    # =========================
    # Encode User IDs
    # =========================

    user_ids = ratings_df['userId'].unique()

    user2user_encoded = {
        x: i for i, x in enumerate(user_ids)
    }

    ratings_df['user_id'] = (
        ratings_df['userId']
        .map(user2user_encoded)
    )

    # =========================
    # Encode Movie IDs
    # =========================

    movie_ids = movies_df['movieId'].unique()

    movie2movie_encoded = {
        x: i for i, x in enumerate(movie_ids)
    }

    movies_df['movie_id'] = (
        movies_df['movieId']
        .map(movie2movie_encoded)
    )

    ratings_df['movie_id'] = (
        ratings_df['movieId']
        .map(movie2movie_encoded)
    )

    ratings_df = ratings_df.dropna(subset=['movie_id'])

    ratings_df['movie_id'] = (
        ratings_df['movie_id']
        .astype(int)
    )

    print(user_ids.shape)
    print(movie_ids.shape)
    print(ratings_df.shape)

    # =========================
    # Genre One-hot Encoding
    # =========================

    genres_dummies = (
        movies_df['genres']
        .str.get_dummies(sep='|')
    )

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

    # =========================
    # Extract Year
    # =========================

    movies_df['year'] = (
        movies_df['title']
        .apply(extract_year)
    )

    movies_df['year'] = (
        movies_df['year']
        .fillna(
            movies_df['year']
            .rolling(
                window=5,
                center=True,
                min_periods=1
            )
            .mean()
        )
        .astype(int)
    )

    # =========================
    # Movie Features
    # =========================

    movie_features = pd.concat(
        [
            movies_df[['movie_id']],
            movies_df[['year']],
            genres_dummies
        ],
        axis=1
    )

    export_to_csv(
        movie_features,
        os.path.join(
            PROCESSED_DATA_PATH,
            "movie_features.csv"
        )
    )

    # =========================
    # User Features
    # =========================

    merged_df = pd.merge(
        ratings_df[['user_id', 'movie_id', 'rating']],
        movie_features,
        on='movie_id'
    )

    user_features = pd.DataFrame({
        'user_id': ratings_df['user_id'].unique()
    })

    total_ratings_per_user = (
        merged_df
        .groupby('user_id')['rating']
        .count()
    )

    for genre in genres_dummies.columns:

        genre_ratings = merged_df['rating'].where(
            merged_df[genre] == 1
        )

        sum_genre_ratings = (
            genre_ratings
            .groupby(merged_df['user_id'])
            .sum()
        )

        custom_avg_series = (
            sum_genre_ratings
            / total_ratings_per_user
        )

        avg_col_name = genre + '_avg'

        user_features[avg_col_name] = (
            user_features['user_id']
            .map(custom_avg_series)
            .fillna(0.0)
        )

    export_to_csv(
        user_features,
        os.path.join(
            PROCESSED_DATA_PATH,
            "user_features.csv"
        )
    )

    # =========================
    # Positive Interactions
    # =========================

    positive_interactions = ratings_df[
        ratings_df['rating'] > 4
    ]

    positive_interactions = positive_interactions[
        ['user_id', 'movie_id', 'timestamp']
    ]

    export_to_csv(
        positive_interactions,
        os.path.join(
            PROCESSED_DATA_PATH,
            "positive_interactions.csv"
        )
    )

    # =========================
    # Global Movie Scores
    # =========================

    global_movie_scores = prepare_global_movie_scores(
        ratings_df,
        movies_df,
        movie2movie_encoded
    )

    export_to_csv(
        global_movie_scores,
        os.path.join(
            PROCESSED_DATA_PATH,
            "global_movie_scores.csv"
        )
    )

    print("\nPreprocessing hoàn tất!")