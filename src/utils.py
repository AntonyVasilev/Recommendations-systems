import pandas as pd


def prefilter_items(data, take_n_popular=1000):
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    top_n = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data.loc[~data['item_id'].isin(top_n), 'item_id'] = 999999
    return data


def postfilter_items(user_id, recommednations):
    pass
