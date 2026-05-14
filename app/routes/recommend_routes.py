from flask import Blueprint, jsonify, request
from app.services.recommend_service import RecommendService

recommend_bp = Blueprint("recommend", __name__)

@recommend_bp.route("/", methods=["GET"])
def get_rec_ids():
    user_id = request.args.get('user_id', type=int)
    top_k = request.args.get('top_k', default=20, type=int)
    result, status_code = RecommendService.get_rec_ids(user_id, top_k)
    return jsonify(result), status_code