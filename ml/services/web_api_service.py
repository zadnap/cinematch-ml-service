import os
import requests
from dotenv import load_dotenv

load_dotenv()

WEB_API_URL = os.getenv("WEB_API_URL")

if not WEB_API_URL:
    raise ValueError("Missing WEB_API_URL")


class WebAPIService:
    TIMEOUT = (5, 60)
    SESSION = requests.Session()
    ID_OFFSET = 200947

    @staticmethod
    def _build_url(endpoint):
        return (
            f"{WEB_API_URL.rstrip('/')}/"
            f"{endpoint.lstrip('/')}"
        )

    @staticmethod
    def _get(endpoint):
        url = WebAPIService._build_url(endpoint)

        try:
            response = WebAPIService.SESSION.get(
                url,
                timeout=WebAPIService.TIMEOUT
            )

            response.raise_for_status()

            json_data = response.json()

            if "data" not in json_data:
                print(f"[INVALID RESPONSE] {endpoint}")
                return None

            return json_data["data"]

        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] {endpoint}")

        except requests.exceptions.ConnectionError:
            print(f"[CONNECTION ERROR] {endpoint}")

        except requests.exceptions.HTTPError as e:
            print(f"[HTTP ERROR] {endpoint}: {e}")

        except ValueError:
            print(f"[INVALID JSON] {endpoint}")

        except requests.exceptions.RequestException as e:
            print(f"[REQUEST ERROR] {endpoint}: {e}")

        except Exception as e:
            print(f"[UNKNOWN ERROR] {endpoint}: {e}")

        return None


    @staticmethod
    def get_all_user_features():
        data = WebAPIService._get(
            "/training-data/user-features/all"
        )
    
        if data is None:
                return None

        return [[int(row[0]) + WebAPIService.ID_OFFSET] + row[1:] for row in data[1:]] 

    @staticmethod
    def get_ratings():
        data = WebAPIService._get("/training-data/ratings")
        if data is None:
            return None
        
        return [[int(row[0]) + WebAPIService.ID_OFFSET] + row[1:] for row in data[1:]] 


    @staticmethod
    def get_favourite_movies():
        return WebAPIService._get(
            "/training-data/favourites"
        )