import sys
import csv 
import os 
import pandas as pd 
from src.data_loading import interactions


def load_aggregated_data():
    """
    load separate csv files produced by aggregate text function 
    """
    csv.field_size_limit(sys.maxsize)

    # get combined aggregated text stored in 10 different files 
    metadata_aggregated = pd.concat(
        [pd.read_csv(os.path.join(
            interactions.ROOT, 
            f'data/metadata-aggregated-20221112-part{i}.csv'), 
            engine='python')
        for i in range(10)]
    ).dropna()

    # get aggregated text from new batch of data points
    #  and combine with metadata_aggregated
    metadata_aggregated_hc = pd.read_csv(os.path.join(
            interactions.ROOT, 
            'data/metadata-aggregated-hc.csv')
    )
    metadata_aggregated_balanced = pd.concat([
        metadata_aggregated,
        metadata_aggregated_hc
    ])
    return metadata_aggregated_balanced 
