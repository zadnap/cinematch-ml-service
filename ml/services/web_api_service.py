import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEB_API_URL = os.getenv("WEB_API_URL")

if not WEB_API_URL:
    raise ValueError("Missing WEB_API_URL environment variable")


class WebAPIService:
    TIMEOUT = 10

    @staticmethod
    def get_user_features(user_id):
        response = requests.get(
            f"{WEB_API_URL}/training-data/user-features/{user_id}",
            timeout=WebAPIService.TIMEOUT
        )

        response.raise_for_status()

        return response.json()


    @staticmethod
    def get_all_user_features():
        response = requests.get(
            f"{WEB_API_URL}/training-data/user-features/all",
            timeout=WebAPIService.TIMEOUT
        )

        response.raise_for_status()

        return response.json()


    @staticmethod
    def get_ratings():
        response = requests.get(
            f"{WEB_API_URL}/training-data/ratings",
            timeout=WebAPIService.TIMEOUT
        )

        response.raise_for_status()

        return response.json()


    @staticmethod
    def get_favourite_movies():
        response = requests.get(
            f"{WEB_API_URL}/training-data/favourites",
            timeout=WebAPIService.TIMEOUT
        )

        response.raise_for_status()

        return response.json()