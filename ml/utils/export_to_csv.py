import os
import pandas as pd
from ml.constants import DB_DATA_PATH

def export_to_csv(df, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Đã xuất file: {filename}")


def fetch_and_export_to_csv(api_method, columns, filename, post_processing_fn=None):
    """Fetch -> Biến đổi -> Xuất file."""
    data = api_method()
    df = pd.DataFrame(data, columns=columns)
    
    if post_processing_fn:
        df = post_processing_fn(df)
        
    export_to_csv(df, os.path.join(DB_DATA_PATH, filename))