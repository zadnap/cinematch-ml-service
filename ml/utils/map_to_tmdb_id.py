import os
import pandas as pd
from ml.constants import RAW_DATA_PATH

class IdMapper:
    _links_df = None

    @classmethod
    def get_links_df(cls) -> pd.DataFrame:
        if cls._links_df is None:
            cls._links_df = pd.read_csv(os.path.join(RAW_DATA_PATH, "links.csv"))
        return cls._links_df


def map_to_tmdb_id(movie_ids):
    if not movie_ids:
        return []
        
    df = IdMapper.get_links_df()
    
    return (
        df[df["movieId"].isin(movie_ids)]
        .set_index("movieId")
        .loc[movie_ids, "tmdbId"]
        .dropna()
        .astype(int)
        .tolist()
    )