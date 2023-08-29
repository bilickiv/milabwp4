import pandas as pd
import numpy as np
from pathlib import Path


pd.set_option('display.width', 300)
np.set_printoptions(linewidth=300)
pd.set_option('display.max_columns', 300)

# READ ALL THE DATAFRAMES FROM THE VARIOUS PLACES
df = pd.read_csv("pyactivalues.csv")

df['Category'] = df['id'].apply(lambda e: e[4])

df2 = pd.read_csv("savgol/sgf_daily_avarages/5min/csvs/processed_e_v_values.csv")
df2.drop(["id", "category"], axis=1, inplace=True)


df = pd.concat([df, df2], axis=1)

columns = df.columns
columns = columns.drop("Category")
columns = columns.drop("id")

df_lists = []

for _col in columns:
    _df = df[['id', _col]]
    _dict = _df.to_dict('tight', into=dict)

    _data_list = _dict["data"]
    _data_list.sort(key=lambda e: e[1], reverse=True)

    _index = 0
    epsilon = _data_list[0][1] * 0.005 # 0.5% epsilon - change this if needed
    print(f"{_col=} {epsilon=}")

    i = 0
    while i < len(_data_list):
        _data_list[i].append(_index)

        j = i + 1
        _index_to_add = 1
        try:
            while _data_list[j][1] >= _data_list[i][1] - epsilon:
                _data_list[j].append(_index)
                j += 1
                _index_to_add += 1
        except IndexError:
            pass
        i = j
        _index += _index_to_add

    max_value = _data_list[0][1]
    _len = len(_data_list)

    for idx, _list in enumerate(_data_list):
        _list.append(((_len - _list[2])/_len) * max_value)

    for idx, _list in enumerate(_data_list):
        _list.append(((_len - _list[2])/_len))


    #calculating linearly normnalized values
    for i in range(len(_data_list)):
        _data_list[i].append(i) #rank
        _data_list[i].append((_data_list[i][1])/max_value) #value

    #calculating statistically normalized values
    value_list = []
    for i in range(len(_data_list)):
        value_list.append(_data_list[i][1])
    avg = np.mean(value_list)
    std = np.std(value_list)

    normalized_value_list = []
    for i in range(len(_data_list)):
        _var = [((_data_list[i][1] - avg) / std), i] # value - index pair
        normalized_value_list.append(_var)

    normalized_value_list.sort(key=lambda e: e[0], reverse=True)
    # [value, index, rank] - sorted by value
    for i in range(len(normalized_value_list)):
        normalized_value_list[i].append(i)

    normalized_value_list.sort(key=lambda e: e[1])
    for i, var in enumerate(normalized_value_list):
        _data_list[i].append(var[2]) # rank - might be redundant
        _data_list[i].append(var[0])

    df_dict = {"name": _col, "data": _data_list}
    df_lists.append(df_dict)


# data: [id, original_value, rank_empiric, empiric_value ,empiric_value_normalized, rank_linear, linear_value, rank_statistical ,statistical_value,]
for _d in df_lists:
    print(_d)

df_s = pd.DataFrame()
ids = []
for idx, _dict in enumerate(df_lists):
    _ids = []
    _original_values = []
    _rank_empiric = []
    _empiric_values = []
    _empiric_values_normalized = []
    _rank_linear = []
    _linear_values = []
    _rank_statistical = []
    _statistical_values = []
    for _v in _dict["data"]:
        if idx == 0:
            ids.append(_v[0][-1])
        _ids.append(_v[0])
        _original_values.append(_v[1])
        _rank_empiric.append(_v[2])
        _empiric_values.append(_v[3])
        _empiric_values_normalized.append(_v[4])
        _rank_linear.append(_v[5])
        _linear_values.append(_v[6])
        _rank_statistical.append(_v[7])
        _statistical_values.append(_v[8])

    df_dict = {f"{_dict['name']}_original": _original_values,
               f"{_dict['name']}_rank_empiric": _rank_empiric,
               f"{_dict['name']}_emp_val": _empiric_values,
               f"{_dict['name']}_emp_val_norm": _empiric_values_normalized,
               f"{_dict['name']}_rank_linear": _rank_linear,
               f"{_dict['name']}_linear_val": _linear_values,
              # f"{_dict['name']}_rank_statistical": _rank_statistical, # probably redundant
               f"{_dict['name']}_statistical_val": _statistical_values,
               }

    _df = pd.DataFrame(index=_ids, data=df_dict)
    df_s = pd.concat([df_s, _df], axis=1)

df_s["Category"] = ids

Path(f"values_normalized/pyacti_values_tied").mkdir(parents=True, exist_ok=True)
df_s.to_csv(f"values_normalized/FULL_pyactiValues_plus_normalized.csv")