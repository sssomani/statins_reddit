import pandas as pd

CONFIG = {
    'data' : '/Users/ssomani/research/heartlab/statins_reddit/data/processed/posts_comms_20220712.csv',
    'output' : '/Users/ssomani/research/heartlab/statins_reddit/reports/tables/table_two.xlsx'
}

def create_table_two():
    """
    
    The purpose of this table is to identify how different authors are interacting on Reddit, and in what way.

    Things we want to represent:
        1. How many different posts each author may be generating. 
        2. How many different subreddits each author interacts with by posting content to.   
    """