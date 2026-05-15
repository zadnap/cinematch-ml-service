from flask import Blueprint, jsonify, request
from app.services.recommend_service import RecommendService

recommend_bp = Blueprint("recommend", __name__)

@recommend_bp.route("", methods=["POST"])
def get_rec_ids():
    data = request.get_json()
    user_id = data.get("user_id")
    top_k = data.get("top_k", 20)
    user_features = data.get("user_features")

    if user_id is None:
        return jsonify({
            "error": "Missing user_id"
        }), 400

    if user_features is None:
        return jsonify({
            "error": "Missing user_features"
        }), 400

    result, status_code = RecommendService.get_rec_ids(
        user_id=user_id,
        user_features=user_features,
        top_k=top_k
    )

    return jsonify(result), status_code