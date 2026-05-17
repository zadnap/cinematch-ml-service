import os
import pandas as pd

class IdMapper:
    _links_df = None
    _raw_data_path = "ml/data/raw_data"

    @classmethod
    def get_links_df(cls) -> pd.DataFrame:
        if cls._links_df is None:
            cls._links_df = pd.read_csv(os.path.join(cls._raw_data_path, "links.csv"))
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