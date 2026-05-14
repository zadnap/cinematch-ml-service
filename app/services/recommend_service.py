from app.utils.response import success, error
from ml.inference.recommender import recommend_movies

class RecommendService:
    @staticmethod
    def get_rec_ids(user_id, top_k):
        top_tmdb_ids = recommend_movies(user_id, top_k)
        return success(top_tmdb_ids, "Getting recommended IDs successfully")