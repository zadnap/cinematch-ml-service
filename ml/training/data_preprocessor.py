import os
import pickle
import pandas as pd
from ml.utils.extract_year_from_title import extract_year_from_title
from ml.utils.export_to_csv import fetch_and_export_to_csv
from ml.services.web_api_service import WebAPIService
from ml.constants import RAW_DATA_PATH, PROCESSED_DATA_PATH, DB_DATA_PATH

class DataPreprocessor:
    def __init__(self):
        self.movie2movie_encoded = {}
        self.user2user_encoded = {}


    def save_mappings(self):
        """Lưu bộ mã hóa ID (Encoding) vào ổ cứng"""
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        movie_mapping_path = os.path.join(PROCESSED_DATA_PATH, "movie2movie_encoded.pkl")
        user_mapping_path = os.path.join(PROCESSED_DATA_PATH, "user2user_encoded.pkl")
        
        with open(movie_mapping_path, 'wb') as f:
            pickle.dump(self.movie2movie_encoded, f)
        with open(user_mapping_path, 'wb') as f:
            pickle.dump(self.user2user_encoded, f)
        print(f"Đã lưu các bộ mapping vào: {PROCESSED_DATA_PATH}")


    def load_mappings(self):
        """Tải các bộ mapping từ file đã lưu lên bộ nhớ nếu có"""
        movie_mapping_path = os.path.join(PROCESSED_DATA_PATH, "movie2movie_encoded.pkl")
        user_mapping_path = os.path.join(PROCESSED_DATA_PATH, "user2user_encoded.pkl")
        
        if os.path.exists(movie_mapping_path) and os.path.exists(user_mapping_path):
            with open(movie_mapping_path, 'rb') as f:
                self.movie2movie_encoded = pickle.load(f)
            with open(user_mapping_path, 'rb') as f:
                self.user2user_encoded = pickle.load(f)
            print("Đã load thành công mapping cũ từ file pickle!")
            return True
        print("Không tìm thấy file mapping cũ. Sẽ khởi tạo mới.")
        return False


    def _process_genres(self, movies_df):
        """Xử lý định dạng thể loại phim sang dạng One-Hot Encoding"""
        genres_dummies = movies_df['genres'].str.get_dummies(sep='|')
        if '(no genres listed)' in genres_dummies.columns:
            genres_dummies.drop(columns=['(no genres listed)'], inplace=True)

        genres_dummies.columns = [
            col.lower().replace('-', '_').replace(' ', '_')
            for col in genres_dummies.columns
        ]
        return genres_dummies.T.groupby(level=0).max().T


    def _process_years(self, movies_df):
        """Trích xuất và làm sạch năm phát hành phim"""
        movies_df['year'] = movies_df['title'].apply(extract_year_from_title)
        movies_df['year'] = (
            movies_df['year']
            .fillna(movies_df['year'].rolling(window=5, center=True, min_periods=1).mean())
            .fillna(2000)
            .astype(int)
        )
        return movies_df[['year']]


    def _build_movie_features(self, movies_df, output_dir=PROCESSED_DATA_PATH):
        """Xử lý Genres & Years -> Xuất file movie_features.csv"""
        genres_dummies = self._process_genres(movies_df)
        movies_df['year'] = self._process_years(movies_df)

        movie_features = pd.concat([movies_df[['movie_id', 'year']], genres_dummies], axis=1)
        movie_features.to_csv(os.path.join(output_dir, "movie_features.csv"), index=False)
        return movie_features, genres_dummies


    @staticmethod
    def _fetch_db_user_features():
        def _process(df):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df.columns = [
                col if col == 'user_id' else f"{str(col).lower().replace('-', '_').replace(' ', '_')}_avg"
                for col in df.columns
            ]
            df = df.loc[:, ~df.columns.duplicated()]
            
            target_columns = [
                'user_id', 'action_avg', 'adventure_avg', 'animation_avg', 'children_avg', 
                'comedy_avg', 'crime_avg', 'documentary_avg', 'drama_avg', 'fantasy_avg', 
                'film_noir_avg', 'horror_avg', 'imax_avg', 'musical_avg', 'mystery_avg', 
                'romance_avg', 'sci_fi_avg', 'thriller_avg', 'war_avg', 'western_avg'
            ]
            return df.reindex(columns=target_columns, fill_value=0.0)

        fetch_and_export_to_csv(
            api_method=WebAPIService.get_all_user_features,
            columns=None, filename='db_user_features.csv', post_processing_fn=_process
        )


    @staticmethod
    def _fetch_db_ratings():
        fetch_and_export_to_csv(
            api_method=WebAPIService.get_ratings,
            columns=['userId', 'movieId', 'timestamp'],
            filename='db_ratings.csv',
            post_processing_fn=lambda df: df.assign(rating=5)[['userId', 'movieId', 'rating', 'timestamp']]
        )


    @staticmethod
    def _fetch_db_movies():
        fetch_and_export_to_csv(
            api_method=WebAPIService.get_favourite_movies,
            columns=['movieId', 'title', 'genres'],
            filename='db_movies.csv'
        )


    def _fetch_all_api_data(self):
        """Bước 1: Tải tất cả dữ liệu từ API Database"""
        print("1. Đang lấy dữ liệu mới nhất từ Database...")
        self._fetch_db_user_features()
        self._fetch_db_ratings()
        self._fetch_db_movies()


    def _merge_movie_sources(self, raw_movies, db_movies, links_df, db_ratings):
        """Bước 2: Gộp dữ liệu Phim từ Raw và DB (qua liên kết TMDB ID)"""
        print("3. Tiến hành gộp dữ liệu Phim (Raw + DB via TMDB link)...")
        db_tmdb_ids = set(db_ratings['movieId'].unique())
        existing_tmdb_ids = set(links_df['tmdbId'].dropna())
        new_tmdb_ids = [tid for tid in db_tmdb_ids if tid not in existing_tmdb_ids]

        if new_tmdb_ids:
            print(f"-> Phát hiện {len(new_tmdb_ids)} phim mới từ database.")
            new_links, new_movies = [], []
            next_movie_id = int(raw_movies['movieId'].max()) + 1

            for tmdb_id in new_tmdb_ids:
                movie_row = db_movies[db_movies['movieId'] == tmdb_id]
                if movie_row.empty:
                    continue
                new_links.append({'movieId': next_movie_id, 'imdbId': '', 'tmdbId': tmdb_id})
                new_movies.append({'movieId': next_movie_id, 'title': movie_row.iloc[0]['title'], 'genres': movie_row.iloc[0]['genres']})
                next_movie_id += 1

            if new_links:
                links_df = pd.concat([links_df, pd.DataFrame(new_links)], ignore_index=True).drop_duplicates(subset=['tmdbId']).reset_index(drop=True)
                raw_movies = pd.concat([raw_movies, pd.DataFrame(new_movies)], ignore_index=True).drop_duplicates(subset=['movieId']).reset_index(drop=True)

        links_df.to_csv(os.path.join(PROCESSED_DATA_PATH, 'links.csv'), index=False)
        raw_movies.to_csv(os.path.join(PROCESSED_DATA_PATH, 'movies.csv'), index=False)
        return raw_movies, links_df


    def _encode_and_merge_interactions(self, raw_movies, raw_ratings, db_ratings, links_df):
        """Bước 3: Encode ID đồng bộ và gộp lịch sử tương tác (Ratings) từ hai nguồn"""
        print("4. Cập nhật ánh xạ ID Encode cho Movie & User...")
        
        # 1. Encode Movies
        for mid in raw_movies['movieId'].unique():
            if mid not in self.movie2movie_encoded:
                self.movie2movie_encoded[mid] = len(self.movie2movie_encoded)
        raw_movies['movie_id'] = raw_movies['movieId'].map(self.movie2movie_encoded)

        # 2. Map DB ratings về ID hệ thống chung thông qua bảng links
        tmdb2real = dict(zip(links_df['tmdbId'], links_df['movieId']))
        db_ratings['real_movieId'] = db_ratings['movieId'].map(tmdb2real)
        db_ratings['movie_id'] = db_ratings['real_movieId'].map(self.movie2movie_encoded)
        db_ratings = db_ratings.dropna(subset=['movie_id'])
        db_ratings['movie_id'] = db_ratings['movie_id'].astype(int)

        # 3. Map Raw ratings
        raw_ratings['movie_id'] = raw_ratings['movieId'].map(self.movie2movie_encoded)
        raw_ratings = raw_ratings.dropna(subset=['movie_id'])
        
        # 4. Encode Users từ nguồn dữ liệu Raw
        for uid in raw_ratings['userId'].unique():
            if uid not in self.user2user_encoded:
                self.user2user_encoded[uid] = len(self.user2user_encoded)
        raw_ratings['user_id'] = raw_ratings['userId'].map(self.user2user_encoded)
        
        # 5. Gộp chung toàn bộ lịch sử tương tác
        db_interactions = db_ratings[['userId', 'movie_id', 'rating', 'timestamp']].rename(columns={'userId': 'user_id'})
        total_ratings = pd.concat([
            raw_ratings[['user_id', 'movie_id', 'rating', 'timestamp']],
            db_interactions
        ], ignore_index=True).drop_duplicates(subset=['user_id', 'movie_id'], keep='last')

        self.save_mappings()
        return raw_movies, total_ratings


    def _generate_user_features(self, total_ratings, movie_features, genres_dummies):
        """Bước 4: Xây dựng hồ sơ sở thích người dùng (User Features)"""
        print("7. Xây dựng hồ sơ sở thích người dùng (User Features)...")
        merged_df = pd.merge(total_ratings[['user_id', 'movie_id', 'rating']], movie_features, on='movie_id')
        total_ratings_per_user = merged_df.groupby('user_id')['rating'].count()

        weighted_genres = merged_df[list(genres_dummies.columns)].multiply(merged_df['rating'], axis=0)
        weighted_genres['user_id'] = merged_df['user_id']
        
        user_genre_avgs = weighted_genres.groupby('user_id').sum().divide(total_ratings_per_user, axis=0).fillna(0.0)
        user_genre_avgs.columns = [f"{col}_avg" for col in user_genre_avgs.columns]
        
        user_features = pd.DataFrame({'user_id': total_ratings['user_id'].unique()})
        user_features = pd.merge(user_features, user_genre_avgs, on='user_id', how='left').fillna(0.0)
        
        # Gộp bổ sung dữ liệu đặc trưng từ DB bổ trợ nếu có
        db_user_feature = pd.read_csv(os.path.join(DB_DATA_PATH, "db_user_features.csv"))
        if not db_user_feature.empty:
            user_features = pd.concat([user_features, db_user_feature], ignore_index=True).drop_duplicates(subset=['user_id'], keep='last').reset_index(drop=True)
        user_features['user_id'] = user_features['user_id'].astype(int)

        user_features.to_csv(os.path.join(PROCESSED_DATA_PATH, "user_features.csv"), index=False)


    def _generate_global_scores(self, total_ratings, raw_movies):
        """Bước 5: Tính toán điểm xếp hạng thịnh hành toàn cục (Global Movie Scores)"""
        print("8. Tính toán điểm xếp hạng phim tổng quan (Global Movie Scores)...")
        rating_counts = total_ratings.groupby('movie_id').size().reset_index(name='num_ratings')
        ratings_avg = total_ratings.groupby('movie_id')['rating'].mean().reset_index(name='avg_rating')

        movie_scores_df = pd.merge(rating_counts, raw_movies, on='movie_id')
        movie_scores_df = pd.merge(ratings_avg, movie_scores_df, on='movie_id')
        movie_scores_df['score'] = movie_scores_df['num_ratings'] * movie_scores_df['avg_rating']

        global_movie_scores = movie_scores_df.sort_values(by='score', ascending=False)[
            ['movieId', 'movie_id', 'title', 'genres', 'num_ratings', 'avg_rating', 'score']
        ].rename(columns={'movie_id': 'encoded_id'})
        
        global_movie_scores.to_csv(os.path.join(PROCESSED_DATA_PATH, 'global_movie_scores.csv'), index=False)


    def process_and_sync_pipeline(self):
        """Luồng duy nhất điều phối toàn bộ quá trình xử lý dữ liệu"""
        # Bước 1: Thu thập dữ liệu ngoại vi
        self._fetch_all_api_data()

        print("2. Đọc dữ liệu thô (Raw) và dữ liệu DB...")
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        self.load_mappings()

        raw_movies = pd.read_csv(os.path.join(RAW_DATA_PATH, "movies.csv")).drop_duplicates(subset=['movieId']).reset_index(drop=True)
        raw_ratings = pd.read_csv(os.path.join(RAW_DATA_PATH, "ratings.csv"))
        links_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'links.csv')).drop_duplicates(subset=['tmdbId']).reset_index(drop=True)
        db_movies = pd.read_csv(os.path.join(DB_DATA_PATH, 'db_movies.csv'))
        db_ratings = pd.read_csv(os.path.join(DB_DATA_PATH, 'db_ratings.csv'))

        # Bước 2: Đồng bộ danh sách phim nguồn
        raw_movies, links_df = self._merge_movie_sources(raw_movies, db_movies, links_df, db_ratings)

        # Bước 3: Đánh mã hóa ID và gộp dữ liệu tương tác
        raw_movies, total_ratings = self._encode_and_merge_interactions(raw_movies, raw_ratings, db_ratings, links_df)

        # Bước 4: Xử lý Movie Features độc lập
        print("5. Tạo Movie Features...")
        movie_features, genres_dummies = self._build_movie_features(raw_movies, output_dir=PROCESSED_DATA_PATH)

        # Bước 5: Trích xuất tương tác tích cực
        print("6. Trích xuất Tương tác tích cực (Positive Interactions)...")
        positive_interactions = total_ratings[total_ratings['rating'] >= 4][['user_id', 'movie_id', 'timestamp']]
        positive_interactions.to_csv(os.path.join(PROCESSED_DATA_PATH, "positive_interactions.csv"), index=False)

        # Bước 6: Xây dựng đặc trưng người dùng và bảng điểm toàn cục
        self._generate_user_features(total_ratings, movie_features, genres_dummies)
        self._generate_global_scores(total_ratings, raw_movies)

        print(f"--- Hoàn thành! Toàn bộ pipeline kết thúc hoàn hảo tại: {PROCESSED_DATA_PATH} ---")

preprocessor = DataPreprocessor()