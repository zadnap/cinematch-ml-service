import os
import pandas as pd
from ml.constants import PROCESSED_DATA_PATH

class IdMapper:
    _links_df = None

    @classmethod
    def get_links_df(cls) -> pd.DataFrame:
        if cls._links_df is None:
            cls._links_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "links.csv"))
        return cls._links_df


def map_to_tmdb_id(movie_ids):
    if not movie_ids:
        return []
        
    df = IdMapper.get_links_df()
    
    mapping_series = df.set_index("movieId")["tmdbId"]
    
    return [
        int(mapping_series[mid]) 
        for mid in movie_ids 
        if mid in mapping_series.index and pd.notna(mapping_series[mid])
    ]