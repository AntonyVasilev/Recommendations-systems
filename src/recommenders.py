import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from src.utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        self.data = prefilter_items(data, take_n_popular=5000)
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        """Создает матрицу взаимодействия user_item"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_rec_similar_items(self, x):
        """Находит товар, похоий на x"""
        recs = self.model.similar_items(self.itemid_to_id[x], N=2)
        top_rec = recs[1][0]
        return self.id_to_itemid[top_rec]

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        popular_items = self.data.loc[self.data['user_id'] == user, :].groupby('item_id')['quantity'].sum().reset_index()
        popular_items.rename(columns={'quantity': 'n_sold'}, inplace=True)
        popular_items = popular_items.loc[popular_items['item_id'] != 999999]
        top_n_items = popular_items.sort_values('n_sold', ascending=False).head(N).item_id.tolist()
        similar_items = [self.get_rec_similar_items(item) for item in top_n_items]

        assert len(similar_items) == N, 'Количество рекомендаций != {}'.format(N)
        return similar_items

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        similar_users = self.model.similar_users(self.userid_to_id[user], N)
        res = []
        for similar_user in similar_users:
            popular = self.data.loc[self.data['user_id'] == similar_user[0], :].groupby('item_id')[
                'quantity'].sum().reset_index()
            popular.rename(columns={'quantity': 'n_sold'}, inplace=True)
            popular = popular.loc[popular['item_id'] != 999999]
            top_5 = popular.sort_values('n_sold', ascending=False).head(5).item_id.tolist()
            res.append(np.random.choice(top_5))

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
