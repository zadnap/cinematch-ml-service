import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEB_API_URL = os.getenv("WEB_API_URL")

if not WEB_API_URL:
    raise ValueError("Missing WEB_API_URL environment variable")


class WebAPIService:
    TIMEOUT = 30

    @staticmethod
    def _get(endpoint):
        try:
            response = requests.get(
                f"{WEB_API_URL}{endpoint}",
                timeout=WebAPIService.TIMEOUT
            )

            response.raise_for_status()

            json_data = response.json()

            if "data" not in json_data:
                print(f"Missing data field: {endpoint}")
                return None

            return json_data["data"]

        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] {endpoint}")
            return None

        except requests.exceptions.ConnectionError:
            print(f"[CONNECTION ERROR] {endpoint}")
            return None

        except requests.exceptions.HTTPError as e:
            print(f"[HTTP ERROR] {endpoint}: {e}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"[REQUEST ERROR] {endpoint}: {e}")
            return None

        except Exception as e:
            print(f"[UNKNOWN ERROR] {endpoint}: {e}")
            return None


    @staticmethod
    def get_user_features(user_id):
        return WebAPIService._get(
            f"/training-data/user-features/{user_id}"
        )


    @staticmethod
    def get_all_user_features():
        return WebAPIService._get(
            "/training-data/user-features/all"
        )


    @staticmethod
    def get_ratings():
        return WebAPIService._get(
            "/training-data/ratings"
        )


    @staticmethod
    def get_favourite_movies():
        return WebAPIService._get(
            "/training-data/favourites"
        )