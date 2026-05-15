from app.utils.response import success, error
from ml.inference.recommender import recommend_movies

class RecommendService:
    @staticmethod
    def get_rec_ids(user_id, user_features, top_k):
        try:
            top_tmdb_ids = recommend_movies(
                user_id=int(user_id),
                user_features=user_features,
                top_k=top_k
            )

            return success(
                top_tmdb_ids,
                "Getting recommended IDs successfully"
            )

        except Exception as e:
            print(f"[RECOMMEND ERROR] {e}")

            return error(
                "RECOMMEND_ERROR",
                "Failed to get recommendations",
                500
            )