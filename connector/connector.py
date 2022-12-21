import pandas as pd 
from conf.conf import logging

def get_data(link: str) -> pd.DataFrame:
    """
    This function extracts data from github
    """
    logging.info('Extracting data')
    df = pd.read_csv(link)
    logging.info('Data is extracted and loaded into a df')

    return df