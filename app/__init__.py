import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

def create_app():
    load_dotenv()
    app = Flask(__name__)

    cors_origins = os.getenv("CORS_ORIGINS", "")
    origins_list = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

    CORS(
        app,
        origins=origins_list,
        supports_credentials=True
    )

    with app.app_context():
        from app.routes.recommend_routes import recommend_bp

        app.register_blueprint(recommend_bp, url_prefix="/recommend")

    return app