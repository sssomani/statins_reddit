import pandas as pd
import tableone

CONFIG = {
    'data' : '/Users/ssomani/research/heartlab/statins_reddit/data/raw/posts_comms_20220712.xlsx',
    't1_features' : ['type', 'length', 'query', 'subreddit', 'upvotes'],
    't1_cat_feats' : ['type', 'query', 'subreddit'],
    'output' : '/Users/ssomani/research/heartlab/statins_reddit/reports/tables/table_one.xlsx'
}

def add_to_tableone(table, feature, values):
    """
    Helper function to add features with values to tableone's aggregated dataframes.
    """

    for column, value, total in zip(table.columns, values, table.loc['n'].values[0]):
        percentage = value / total * 100
        table.loc[feature, column] = "{0} ({1:.1f})".format(value, percentage)

    return table

def create_table_one():
    """
    Create our table 1 based on the specifications below:

        [X] Subreddit count
        [X] Query term count
        [X] Number of unique authors
        [X] Upvotes
        [X] Length of Text 

    """

    df = pd.read_excel(CONFIG['data'])
    
    # Find length of each text
    df['length'] = df['content'].apply(lambda x: len(str(x)))

    # Create our table one
    t1 = tableone.TableOne(df, columns=CONFIG['t1_features'], categorical=CONFIG['t1_cat_feats'], groupby='type', missing=False)
    
    # Next, populate the author field.
    n_auth_c, n_auth_p = df.groupby('type')['author'].nunique()
    n_auth = df['author'].nunique()
    t1.tableone = add_to_tableone(t1.tableone, 'unique authors, n (%)', [n_auth, n_auth_c, n_auth_p])

    # Export
    t1.to_excel(CONFIG['output'])

if __name__ == '__main__':
    create_table_one()