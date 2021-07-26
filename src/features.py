import pandas as pd


def get_last_5_purchases_as_bit_array(data_train_ranker, item_features):
    """Заказ товара в последних 5 транзакциях в виде последовательности бит (категориальная)"""
    purchases_by_time = data_train_ranker.merge(item_features, on='item_id',
                                                how='left').groupby(by=['day', 'trans_time', 'basket_id', 'item_id']
                                                                    )['quantity'].count().reset_index().sort_values(
        by=['day', 'trans_time'],
        ascending=False)

    # Словарь с последними 5 покупками
    trans_dict = {}
    trans_no = 0
    for day, trans_time, basket_id, item_id, quantity in purchases_by_time.values:
        if basket_id not in trans_dict.keys():
            trans_dict[basket_id] = {'trans_no': trans_no, 'item_id': []}
            trans_no += 1
        trans_dict[basket_id]['item_id'].append(item_id)
        if len(trans_dict) >= 5:
            break

    # Списки товаров в последних 5 покупках
    trans_list = []
    for value in trans_dict.values():
        trans_list.insert(value['trans_no'], value['item_id'])

    # Список словарей с битовым представлением последних 5 транзакций
    result_list = []
    for item in data_train_ranker.item_id.unique():
        item_trans = ''
        for trans in trans_list:
            item_trans += '1' if item in trans else '0'
        result_list.append({'item_id': item, 'item_in_last_5_transactions': item_trans})

    return pd.DataFrame(result_list)


def get_mean_purchase_per_item_by_department(data_train_ranker, item_features):
    """Средняя сумма покупки 1 товара в каждой категории"""
    sales_value_by_department = data_train_ranker.merge(item_features, on='item_id',
                                                        how='left').groupby(by=['user_id', 'department'])[
        'sales_value'].sum().reset_index()
    quantity_by_department = data_train_ranker.merge(item_features, on='item_id',
                                                     how='left').groupby(by=['user_id', 'department'])[
        'quantity'].sum().reset_index()
    mean_purchase_by_department = sales_value_by_department.merge(quantity_by_department,
                                                                  on=['user_id', 'department'], how='left')
    mean_purchase_by_department.drop(0, axis=0, inplace=True)
    mean_purchase_by_department.reset_index(inplace=True)
    mean_purchase_by_department.drop('index', axis=1, inplace=True)
    mean_purchase_by_department['mean_purchase'] = \
        mean_purchase_by_department['sales_value'] / mean_purchase_by_department['quantity']
    return mean_purchase_by_department


def get_num_purchases_per_department(data_train_ranker, item_features):
    """Кол-во покупок в каждой категории"""
    num_purchases_by_department = data_train_ranker.merge(item_features, on='item_id',
                                                          how='left').groupby(by=['user_id', 'department'])[
        'basket_id'].nunique().reset_index()
    num_purchases_by_department.rename(columns={'basket_id': 'num_purchases'}, inplace=True)
    num_purchases_by_department.drop(0, axis=0, inplace=True)
    num_purchases_by_department.reset_index(inplace=True)
    num_purchases_by_department.drop('index', axis=1, inplace=True)
    return num_purchases_by_department


def get_proportion_of_purchases_by_times_of_day(data_train_ranker):
    """Доля покупок утром/днем/вечером"""
    users_transactions = data_train_ranker[['user_id', 'trans_time']].drop_duplicates(
        subset=['trans_time']).reset_index().drop('index', axis=1)
    users_list = users_transactions.user_id.unique().tolist()

    user_trans_dict = {
        'user_id': [],
        'morning_trans': [],
        'day_trans': [],
        'evening_trans': []
    }
    for user in users_list:
        num_trans = users_transactions.loc[users_transactions.user_id == user,
                                           'trans_time'].count()
        morning_trans = users_transactions[(users_transactions.user_id == user) &
                                           (users_transactions.trans_time <= 900)].trans_time.count()
        day_trans = users_transactions[(users_transactions.user_id == user) &
                                       (users_transactions.trans_time > 900) &
                                       (users_transactions.trans_time < 1800)].trans_time.count()
        evening_trans = users_transactions[(users_transactions.user_id == user) &
                                           (users_transactions.trans_time >= 1800)].trans_time.count()
        user_trans_dict['user_id'].append(user)
        user_trans_dict['morning_trans'].append(morning_trans / num_trans)
        user_trans_dict['day_trans'].append(day_trans / num_trans)
        user_trans_dict['evening_trans'].append(evening_trans / num_trans)

    user_trans_df = pd.DataFrame(user_trans_dict)
    return user_trans_df


def get_num_purchases_per_week(data_train_ranker):
    """Кол-во покупок в неделю"""
    num_purchases_by_week = data_train_ranker.groupby(by=['item_id',
                                                          'week_no'])['basket_id'].nunique().reset_index()
    num_purchases_by_week.rename(columns={'basket_id': 'week_num_purchases'}, inplace=True)
    week_purchases_df = num_purchases_by_week.groupby(by='item_id').agg({'week_no': 'count',
                                                                         'week_num_purchases': 'sum'}).reset_index()
    week_purchases_df['n_purchases_per_week'] = \
        week_purchases_df.week_num_purchases / week_purchases_df.week_no
    return week_purchases_df


def get_mean_num_purchases_per_item_dept_week(data_train_ranker, item_features):
    """Среднее кол-во покупок 1 товара в категории в неделю"""
    n_purchases_by_dept = data_train_ranker.merge(item_features, on='item_id', how='left').groupby(
        by=['department', 'week_no', 'item_id'])['basket_id'].nunique().reset_index()
    n_purchases_by_dept.rename(columns={'basket_id': 'n_purchases'}, inplace=True)
    n_items_per_week = n_purchases_by_dept.groupby(by=['department', 'week_no'])[
        'item_id'].count().reset_index()
    n_items_per_week.rename(columns={'item_id': 'n_items'}, inplace=True)
    n_purchases_per_week = n_purchases_by_dept.groupby(by=['department', 'week_no'])[
        'n_purchases'].sum().reset_index()
    mean_purchases_per_week = n_items_per_week.merge(n_purchases_per_week,
                                                     on=['department', 'week_no'], how='left')
    mean_purchases_per_week['mean_purchases_per_week'] = round(
        mean_purchases_per_week.n_purchases / mean_purchases_per_week.n_items, 4)
    mean_purchases_per_week.drop([0, 1, 2, 3, 4, 5], axis=0, inplace=True)
    mean_purchases_per_week.reset_index(inplace=True)
    mean_purchases_per_week.drop('index', axis=1, inplace=True)

    mean_n_purchases_per_week = mean_purchases_per_week.groupby(by='department').agg({
        'week_no': 'count', 'mean_purchases_per_week': 'sum'}).reset_index()
    mean_n_purchases_per_week['mean_n_purchases_per_week'] = \
        mean_n_purchases_per_week.mean_purchases_per_week / mean_n_purchases_per_week.week_no
    return mean_n_purchases_per_week


def get_price(data_train_ranker):
    """Цена"""
    item_price_df = data_train_ranker[['item_id', 'quantity', 'sales_value', 'retail_disc']].copy()
    item_price_df['price'] = (item_price_df.sales_value -
                              item_price_df.retail_disc) / item_price_df.quantity
    item_price_df = item_price_df.groupby(by=['item_id'])['price'].mean().reset_index()
    return item_price_df


def get_mean_price_by_department(df_ranker_train):
    """Средняя цена товара в категории"""
    mean_price_by_department = df_ranker_train[['department',
                                                'price']].groupby('department')['price'].mean().reset_index()
    mean_price_by_department.rename(columns={'price': 'mean_price'}, inplace=True)
    return mean_price_by_department


def get_user_nun_purchases_per_week(data_train_ranker, item_features):
    """Кол-во покупок юзером конкретной категории в неделю"""
    purchases_by_usr = data_train_ranker.merge(item_features, on='item_id', how='left'
                                               ).groupby(by=['user_id', 'department', 'week_no'])[
        'basket_id'].nunique().reset_index()
    usr_purchases_by_dept = purchases_by_usr.groupby(by=['user_id', 'department']).agg({
        'week_no': 'count', 'basket_id': 'sum'}).reset_index()
    usr_purchases_by_dept.rename(columns={'week_no': 'n_weeks',
                                          'basket_id': 'n_purchases'}, inplace=True)
    usr_purchases_by_dept.drop(0, axis=0, inplace=True)
    usr_purchases_by_dept.reset_index(inplace=True)
    usr_purchases_by_dept.drop('index', axis=1, inplace=True)
    usr_purchases_by_dept['usr_purchases_by_dept_per_week'] = \
        usr_purchases_by_dept.n_purchases / usr_purchases_by_dept.n_weeks
    return usr_purchases_by_dept


def get_mean_purchases_all_users_by_department_per_week(data_train_ranker, item_features):
    """Среднее кол-во покупок всеми юзерами конкретной категории в неделю"""
    all_users_purchases = data_train_ranker.merge(item_features, on='item_id', how='left'
                                                  ).groupby(by=['department', 'week_no', 'user_id'])[
        'basket_id'].nunique().reset_index()
    all_users_purchases_by_dept = all_users_purchases.groupby(by='department').agg({
        'week_no': 'nunique', 'user_id': 'count', 'basket_id': 'sum'}).reset_index()
    all_users_purchases_by_dept.rename(columns={'user_id': 'n_users', 'week_no': 'n_weeks',
                                                'basket_id': 'n_purchases'}, inplace=True)
    all_users_purchases_by_dept.drop(0, axis=0, inplace=True)
    all_users_purchases_by_dept.reset_index(inplace=True)
    all_users_purchases_by_dept.drop('index', axis=1, inplace=True)
    all_users_purchases_by_dept['mean_purchases_all_users_per_week'] = \
        all_users_purchases_by_dept.n_purchases / all_users_purchases_by_dept.n_users \
        / all_users_purchases_by_dept.n_weeks
    return all_users_purchases_by_dept


def get_num_purchases_sub_by_mean(data_train_ranker, item_features):
    """(Кол-во покупок юзером конкретной категории в неделю) - (Среднее кол-во покупок всеми юзерами конкретной категории в неделю)"""
    usr_purchases_by_dept = get_user_nun_purchases_per_week(data_train_ranker, item_features)
    all_users_purchases_by_dept = get_mean_purchases_all_users_by_department_per_week(
        data_train_ranker, item_features)

    n_purchases_sub_by_mean = usr_purchases_by_dept[['user_id', 'department',
                                                     'usr_purchases_by_dept_per_week']].merge(
        all_users_purchases_by_dept[['department',
                                     'mean_purchases_all_users_per_week']], on='department', how='left')
    n_purchases_sub_by_mean['n_purchases_sub_by_mean'] = \
        n_purchases_sub_by_mean.usr_purchases_by_dept_per_week - \
        n_purchases_sub_by_mean.mean_purchases_all_users_per_week
    return n_purchases_sub_by_mean


def get_num_purchases_div_by_mean_all_users(data_train_ranker, item_features):
    """(Кол-во покупок юзером конкретной категории в неделю) / (Среднее кол-во покупок всеми юзерами конкретной категории в неделю)"""
    usr_purchases_by_dept = get_user_nun_purchases_per_week(data_train_ranker, item_features)
    all_users_purchases_by_dept = get_mean_purchases_all_users_by_department_per_week(
        data_train_ranker, item_features)

    n_purchases_div_by_mean = usr_purchases_by_dept[['user_id', 'department',
                                                     'usr_purchases_by_dept_per_week']].merge(
        all_users_purchases_by_dept[['department',
                                     'mean_purchases_all_users_per_week']], on='department', how='left')
    n_purchases_div_by_mean['n_purchases_div_by_mean_all_users'] = \
        n_purchases_div_by_mean.usr_purchases_by_dept_per_week / \
        n_purchases_div_by_mean.mean_purchases_all_users_per_week
    return n_purchases_div_by_mean


def get_mean_sales_value_per_item_by_department(data_train_ranker, item_features):
    """(Средняя сумма покупки 1 товара в каждой категории (берем категорию item_id)) - (Цена item_id)"""
    sales_values_by_dept = data_train_ranker.merge(item_features, on='item_id',
                                                   how='left').groupby(by=['department']
                                                                       ).agg({'item_id': 'count',
                                                                              'sales_value': 'sum'}).reset_index()
    sales_values_by_dept.drop(0, axis=0, inplace=True)
    sales_values_by_dept.reset_index(inplace=True)
    sales_values_by_dept.drop('index', axis=1, inplace=True)
    sales_values_by_dept.rename(columns={'item_id': 'n_items', 'sales_value': 'sale_sum'},
                                inplace=True)
    sales_values_by_dept['mean_sale_sum_per_item'] = \
        sales_values_by_dept.sale_sum / sales_values_by_dept.n_items
    return sales_values_by_dept


def get_total_item_sales_value(df_join_train_matcher):
    """Общая сумма покупок каждого товара"""
    return df_join_train_matcher.groupby(by='item_id').agg('sales_value').sum().rename('total_item_sales_value')


def get_total_quantity_value(df_join_train_matcher):
    """Общее количество по каждому товару"""
    return df_join_train_matcher.groupby(by='item_id').agg('quantity').sum().rename('total_quantity_value')


def get_item_freq(df_join_train_matcher):
    """Количество покупателей по каждому товару"""
    return df_join_train_matcher.groupby(by='item_id').agg('user_id').count().rename('item_freq')


def get_user_freq(df_join_train_matcher):
    """Частота пользователей"""
    return df_join_train_matcher.groupby(by='user_id').agg('user_id').count().rename('user_freq')


def get_total_user_sales_value(df_join_train_matcher):
    """Общее количество покупок по каждому пользователю"""
    return df_join_train_matcher.groupby(by='user_id').agg('sales_value').sum().rename('total_user_sales_value')


def get_item_quantity_per_week(df_join_train_matcher):
    """Среднее количество покупок товара в неделю"""
    return df_join_train_matcher.groupby(by='item_id').agg('quantity').sum(). \
               rename('item_quantity_per_week') / df_join_train_matcher.week_no.nunique()


def get_user_quantity_per_week(df_join_train_matcher):
    """Среднее количество купленного товара пользователем в неделю"""
    return df_join_train_matcher.groupby(by='user_id').agg('quantity').sum(). \
               rename('user_quantity_per_week') / df_join_train_matcher.week_no.nunique()


def get_item_quantity_per_basket(df_join_train_matcher):
    """Среднее количество товара за 1 покупку"""
    return df_join_train_matcher.groupby(by='item_id').agg('quantity').sum(). \
               rename('item_quantity_per_basket') / df_join_train_matcher.basket_id.nunique()


def get_user_quantity_per_basket(df_join_train_matcher):
    """Среднее количество товара у польователя за 1 покупку"""
    return df_join_train_matcher.groupby(by='user_id').agg('quantity').sum(). \
               rename('user_quantity_per_basket') / df_join_train_matcher.basket_id.nunique()


def get_item_freq_per_basket(df_join_train_matcher):
    """Средняя частота товара в карзине"""
    return df_join_train_matcher.groupby(by='item_id').agg('user_id').count(). \
               rename('item_freq_per_basket') / df_join_train_matcher.basket_id.nunique()

def get_user_freq_per_basket(df_join_train_matcher):
    """Средняя частота пользователей купивших товар"""
    return df_join_train_matcher.groupby(by='user_id').agg('user_id').count(). \
               rename('user_freq_per_basket') / df_join_train_matcher.basket_id.nunique()


def get_n_transactions(data_train_ranker):
    """Кол-во транзакций клиента"""
    return data_train_ranker.groupby(by=['user_id'])['basket_id'].nunique().reset_index().rename(
        columns={'basket_id': 'n_transactions'})


def get_n_stores(data_train_ranker):
    """Кол-во магазинов, в которых продавался товар"""
    return data_train_ranker.groupby(by=['item_id'])['store_id'].nunique().reset_index().rename(
        columns={'store_id': 'n_stores'})


def get_unique_items_in_basket(data_train_ranker):
    """mean / max / std кол-ва уникальных товаров в корзине клиента"""
    nunique_items_per_purchse_by_user = data_train_ranker.groupby(by=['user_id', 'basket_id']
                                                                  )['item_id'].nunique().reset_index()
    nunique_items = nunique_items_per_purchse_by_user.groupby(by=['user_id'])['item_id'].mean(). \
        reset_index().rename(columns={'item_id': 'mean_unique_items'})
    nunique_items = nunique_items.merge(nunique_items_per_purchse_by_user.groupby(by=['user_id'])[
                                            'item_id'].max().reset_index().rename(
        columns={'item_id': 'max_unique_items'}), on='user_id', how='left')
    nunique_items = nunique_items.merge(nunique_items_per_purchse_by_user.groupby(by=['user_id'])[
                                            'item_id'].std().reset_index().rename(
        columns={'item_id': 'std_unique_items'}), on='user_id', how='left')
    return nunique_items


def get_unique_departments_in_basket(data_train_ranker, item_features):
    """mean / max / std кол-ва уникальных категорий в корзине клиента"""
    nunique_departments_per_purchse_by_user = \
        data_train_ranker.merge(item_features, on='item_id', how='left').groupby(by=['user_id', 'basket_id'])[
            'department'].nunique().reset_index()
    nunique_departments = nunique_departments_per_purchse_by_user.groupby(by=['user_id'])['department'].mean(). \
        reset_index().rename(columns={'department': 'mean_unique_departments'})
    nunique_departments = nunique_departments.merge(nunique_departments_per_purchse_by_user.groupby(by=['user_id'])[
        'department'].max().reset_index().rename(
        columns={'department': 'max_unique_departments'}), on='user_id', how='left')
    nunique_departments = nunique_departments.merge(nunique_departments_per_purchse_by_user.groupby(by=['user_id'])[
        'department'].std().reset_index().rename(
        columns={'department': 'std_unique_departments'}), on='user_id', how='left')
    return nunique_departments
