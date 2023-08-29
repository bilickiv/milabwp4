from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, accuracy_score
import csv
from pathlib import Path
import warnings
import random
import copy
import tensorflow as tf
import sys

from sklearn.preprocessing import MinMaxScaler
color = sns.color_palette()
pd.set_option('display.width', 1920)
np.set_printoptions(linewidth=1920)
pd.set_option('display.max_columns',300)

warnings.filterwarnings("ignore")
xgb.set_config(verbosity=0)
warnings.filterwarnings('ignore', message='No future splits with positive gain')


glob_cols = None

def make_shap_csv(title_name, shap_array, X_train, _class, SHAP_LIST_TO_WRITE2):
    try:
        dict_to_write2 = {}
        dict_to_write2["title"] = title_name
        dict_to_write2["columns"] = X_train.columns.tolist()
        dict_to_write2["SHAP_values"] = shap_array
        dict_to_write2["comp_class"] = _class
        SHAP_LIST_TO_WRITE2.append(dict_to_write2)
    except:
        pass


def calculate_SHAP(model, X_train, X_test, calculated_indexes, shap_array, title=None):

    try:
        explainer = shap.Explainer(model.predict, X_test)
        sv = explainer(X_test)

        if title == "keras":
            """
            for example the prediction of the 10th row's
            (prediction of the 10th sample) 
            shap_values regards to 1 is (value1)

            it looks like this:
            sv.values =  [row1[feature0[value0, value1]], 
                            feature1[value0, value1], 
                            feature2[value0, value1],...]
                          row2[feature0[value0, value1],...],...] 
            """
            for idx, ci in enumerate(calculated_indexes):
                nums_to_append = []
                nta = 1
                for arr in sv.values[idx]:
                    nums_to_append.append(arr[nta])
                shap_array[ci] = nums_to_append
        else:
            for idx, ci in enumerate(calculated_indexes):
                shap_array[ci] = sv.values[idx]
    except:
        pass


def prepare_dataframe(dataframe):
    """
    Prepares the dataframe if needed
    """

    classes_to_drop = ["Unnamed: 0"]

    columns = dataframe.columns
    for cla in classes_to_drop:
        if cla in columns:
            dataframe.drop([cla], inplace=True, axis=1)

    if "group" in columns:
        dataframe = dataframe.rename(columns={"group": "class"})

    return dataframe


def get_cols_list(cols_list):

    _cols_list_filtered = []
    for c in cols_list:
        c2 = c.replace("[", "")
        c2 = c2.replace("]", "")
        c2 = c2.replace("'", "")
        c2 = c2.replace(" ", "")
        if not "_thrd" in c:
            _cols_list_filtered.append(c2.split(","))

    return _cols_list_filtered


def get_only_unique(cols_list_filtered):

    temp_list = [tuple(c) for c in cols_list_filtered]
    temp_list = list(set(temp_list))
    temp_list = [list(c) for c in temp_list]
    return temp_list


def make_features(cols_list_df, subset_to_run=0, group_by=None, group_list=None, sort_by="accuracy", ascending=True, slice_range=(0,50)):

    cols_list_filtered = []

    columns = cols_list_df.columns.values

    if "features" in columns:
        cols_list_df = cols_list_df.rename(columns={"features": "feature"})
    columns = cols_list_df.columns.values
    columns = [c for c in columns if not c.startswith("col_")]

    cols_list_df = cols_list_df[columns]


    print("Getting the best rows per clique length!")
    lengths = cols_list_df.clique_length.unique()
    #####
    cldf_temp = pd.DataFrame()

    for le in lengths:
        df_temp = cols_list_df[cols_list_df.clique_length == le]
        top_n_row = df_temp.nsmallest(500, "max_weight")
        if cldf_temp.shape[0] == 0:
            cldf_temp = top_n_row
        else:
            cldf_temp = pd.concat([cldf_temp, top_n_row], axis=0)

    ####




    if "limit" in columns:
        limits = cols_list_df.limit.unique().tolist()
    else:
        print("DataFrame has no 'limits' column!")
        limits = None
    
    #collects data for each limit
    if limits:
        for lim in limits:
            _cols_list_df = cldf_temp[cldf_temp["limit"] == lim]
            cols_list = _cols_list_df.feature.values.tolist()
            #do something here - if needed
            cols_list_to_add = get_cols_list(cols_list)
            if cols_list_to_add:
                cols_list_filtered = cols_list_filtered + cols_list_to_add
    else:
        cols_list = cldf_temp.feature.values.tolist()
        cols_list_to_add = get_cols_list(cols_list)
        if cols_list_to_add:
            cols_list_filtered = cols_list_filtered + cols_list_to_add

    cols_list_filtered = get_only_unique(cols_list_filtered)

    return cols_list_filtered


def run_ML(features, cols_list_filtered, out_dir, models_to_run, calculate_shap=False, acc_lim=[0.0, 0.0, 0.0, 0.0], rec_lim=[0.0, 0.0, 0.0, 0.0], prec_lim=[0.0, 0.0, 0.0, 0.0], additional_str="", classes_to_compare=[2,3], sort_by=None):

    """
    :param features: feature values
    :param cols_list_filtered: cliques
    :param out_dir: out directory
    :param models_to_run: which models to use to predict
    :param calculate_shap: calculate shap? y/n
    :param acc_lim: if the accuracy of the prediction is equal or higher than the limit, it gets written in the .csv otherwise not
    :param rec_lim: if the recall of the prediction is equal or higher than the limit, it gets written in the .csv otherwise not
    :param prec_lim: if the precision of the prediction is equal or higher than the limit, it gets written in the .csv otherwise not
    :param additional_str: additional string to mark the csvs
    :param classes_to_compare: compares the 0 class data to these classes. One at a time.
    :param sort_by: sorting the result csv by this
    :return: None
    """

    accepted_models = ["logReg", "RFC", "xGB", "lGB", "keras"]
    unaccepted_models = []

    for mtr in models_to_run:
        if mtr not in accepted_models:
            unaccepted_models.append(mtr)

    if unaccepted_models:
        raise ValueError(f"Attribue - 'models_to_run' contains non viable models:"
                         f"\t{unaccepted_models}\n"
                         f"\tThe viable models are: {accepted_models}")

    SHAP_LIST_TO_WRITE = []
    SHAP_LIST_TO_WRITE2 = []
    headers = ["idx", "title", "columns", "mean_SHAP_values", "comp_class"]
    headers2 = ["idx", "title", "columns", "comp_class", "SHAP_values"]

    feature_row = []
    comp_class = []
    logreg_row = []
    RFC_row = []
    xGB_row = []
    lGB_row = []
    keras_row = []



    for n in range(len(cols_list_filtered)):
        print(f"{n + 1}/{len(cols_list_filtered)}")
        print(f"{n + 1}/{len(cols_list_filtered)}")
        print(f"{n + 1}/{len(cols_list_filtered)}")
        ftr = cols_list_filtered[n]

        print(ftr)

        if "category" in features.columns:
            features = features.rename(columns={"category": "class"})

        for i in classes_to_compare:
            data = features[ftr + ["class"]].copy()
            data = data.loc[(features["class"] == i) | (features["class"] == 1)]
            data["class"][data["class"] == 1] = 0
            data["class"][data["class"] == i] = 1

            data.reset_index(drop=True, inplace=True)
            n_splits = data["class"].value_counts().min()
            print(f"N_splits are: {n_splits}. Length of data is: {data.shape[0]}")

            X_train = data.drop(['class'], axis=1)
            y_train = data["class"]
            k_fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=2018)

            if "logReg" in models_to_run:
                print("logReg is running!")
                penalty = 'l2'
                C = 1.0
                class_weight = 'balanced'
                random_state = 2018
                solver = 'liblinear'
                n_jobs = 1
                logReg = LogisticRegression(penalty=penalty, C=C,
                                            class_weight=class_weight, random_state=random_state,
                                            solver=solver, n_jobs=n_jobs, verbose=False)
                predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                                        index=y_train.index, columns=[0])

                shap_array = [[]] * len(X_train)
                for train_index, cv_index in k_fold.split(np.zeros(len(X_train))
                        , y_train.ravel()):
                    model = copy.deepcopy(logReg)

                    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
                    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
                    model.fit(X_train_fold, y_train_fold)

                    predictionsBasedOnKFolds.loc[X_cv_fold.index, :] = [[x] for x in model.predict(X_cv_fold)]
                    if calculate_shap:
                        calculate_SHAP(model, X_train_fold, X_cv_fold,  cv_index, shap_array)

                preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 0]], axis=1)
                preds.columns = ['trueLabel', 'prediction']
                predictionsBasedOnKFoldsRandomForests = preds.copy()
                logreg_prec = precision_score(list(preds['trueLabel']), list(preds['prediction']))
                logreg_recall = recall_score(list(preds['trueLabel']), list(preds['prediction']))
                logreg_acc = accuracy_score(list(preds['trueLabel']), list(preds['prediction']))

                logreg_row.append([ftr, '1' + str(i), logreg_prec, logreg_recall, logreg_acc, preds['trueLabel'].values, preds['prediction'].values, None])
                if calculate_shap:
                    make_shap_csv("logReg", shap_array, X_train, '1' + str(i), SHAP_LIST_TO_WRITE2)
                print("logReg is done!")
            ####################################################################################################################################
            if "RFC" in models_to_run:
                print("RFC is running!")
                n_estimators = 50
                max_features = 'auto'
                max_depth = None
                min_samples_split = 2
                min_samples_leaf = 1
                min_weight_fraction_leaf = 0.0
                max_leaf_nodes = None
                bootstrap = True
                oob_score = False
                n_jobs = -1
                random_state = 2018
                class_weight = 'balanced'
                RFC = RandomForestClassifier(n_estimators=n_estimators,
                                             max_features=max_features, max_depth=max_depth,
                                             min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                             min_weight_fraction_leaf=min_weight_fraction_leaf,
                                             max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap,
                                             oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                             class_weight=class_weight, verbose=False)
                trainingScores = []
                predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                                        index=y_train.index, columns=[0])
                shap_array = [[]] * len(X_train)
                for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                                          y_train.ravel()):
                    model = copy.deepcopy(RFC)
                    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
                    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
                    model.fit(X_train_fold, y_train_fold)
                    predictionsBasedOnKFolds.loc[X_cv_fold.index, :] = [[x] for x in model.predict(X_cv_fold)]
                    if calculate_shap:
                        calculate_SHAP(model, X_train_fold, X_cv_fold, cv_index, shap_array)

                preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 0]], axis=1)
                preds.columns = ['trueLabel', 'prediction']
                predictionsBasedOnKFoldsRandomForests = preds.copy()
                rfc_prec = precision_score(list(preds['trueLabel']), list(preds['prediction']))
                rfc_recall = recall_score(list(preds['trueLabel']), list(preds['prediction']))
                rfc_acc = accuracy_score(list(preds['trueLabel']), list(preds['prediction']))
                RFC_row.append([ftr, '1' + str(i), rfc_prec, rfc_recall, rfc_acc, preds['trueLabel'].values, preds['prediction'].values,None])
                if calculate_shap:
                    make_shap_csv("RFC", shap_array, X_train, '1' + str(i), SHAP_LIST_TO_WRITE2)
                print("RFC is done!")
            ####################################################################################################################################
            if "xGB" in models_to_run:
                params_xGB = {
                    'nthread': 16,
                    'learning_rate': 0.3,
                    'gamma': 0,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'max_delta_step': 0,
                    'subsample': 1.0,
                    'colsample_bytree': 1.0,
                    'objective': 'binary:logistic',
                    'num_class': 1,
                    'eval_metric': 'logloss',
                    'seed': 2018,
                    'silent': 1
                }
                trainingScores = []
                cvScores = []
                predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                                        index=y_train.index, columns=['prediction'])
                shap_array = [[]] * len(X_train)
                for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                                          y_train.ravel()):
                    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
                    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
                    dtrain = xgb.DMatrix(data=X_train_fold, label=y_train_fold)
                    dCV = xgb.DMatrix(data=X_cv_fold)
                    bst = xgb.cv(params_xGB, dtrain, num_boost_round=2000,
                                 nfold=5, early_stopping_rounds=200, verbose_eval=50)
                    best_rounds = np.argmin(bst['test-logloss-mean'])
                    bst = xgb.train(params_xGB, dtrain, best_rounds)
                    loglossTraining = log_loss(y_train_fold, bst.predict(dtrain))
                    trainingScores.append(loglossTraining)
                    predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'] = bst.predict(dCV)
                    loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'])
                    cvScores.append(loglossCV)
                    if calculate_shap:
                        calculate_SHAP(bst, X_train_fold, X_cv_fold, cv_index, shap_array)

                preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 'prediction']], axis=1)
                preds.columns = ['trueLabel', 'prediction']
                predictionsBasedOnKFoldsXGBoostGradientBoosting = preds.copy()
                preds["prediction_label"] = np.nan
                preds["prediction_label"].loc[preds["prediction"] >= 0.5] = 1
                preds["prediction_label"].loc[preds["prediction"] < 0.5] = 0
                preds["prediction_label"] = preds["prediction_label"].astype("int64")
                xgb_prec = precision_score(list(preds['trueLabel']), list(preds['prediction_label']))
                xgb_recall = recall_score(list(preds['trueLabel']), list(preds['prediction_label']))
                xgb_acc = accuracy_score(list(preds['trueLabel']), list(preds['prediction_label']))
                xGB_row.append([ftr, '1' + str(i), xgb_prec, xgb_recall, xgb_acc, preds['trueLabel'].values, preds['prediction_label'].values, preds['prediction']])
                if calculate_shap:
                    make_shap_csv("xGB", shap_array, X_train, '1' + str(i), SHAP_LIST_TO_WRITE2)
            ####################################################################################################################################
            if "lGB" in models_to_run:
                params_lightGB = {
                    'task': 'train',
                    'application': 'binary',
                    'num_class': 1,
                    'boosting': 'gbdt',
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'metric_freq': 50,
                    'is_training_metric': False,
                    'max_depth': 4,
                    'num_leaves': 31,
                    'learning_rate': 0.01,
                    'feature_fraction': 1.0,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 0,
                    'bagging_seed': 2018,
                    'verbose': 0,
                    'num_threads': 16
                }
                trainingScores = []
                cvScores = []
                predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                                        index=y_train.index, columns=['prediction'])
                shap_array = [[]] * len(X_train)
                for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                                          y_train.ravel()):
                    X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
                    y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
                    lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
                    lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
                    gbm = lgb.train(params_lightGB, lgb_train, num_boost_round=2000,
                                    valid_sets=lgb_eval, early_stopping_rounds=200)
                    loglossTraining = log_loss(y_train_fold, gbm.predict(X_train_fold, num_iteration=gbm.best_iteration))
                    trainingScores.append(loglossTraining)
                    predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'] = gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration)
                    loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'])
                    cvScores.append(loglossCV)
                    if calculate_shap:
                        calculate_SHAP(gbm, X_train_fold, X_cv_fold, cv_index, shap_array)

                preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 'prediction']], axis=1)
                preds.columns = ['trueLabel', 'prediction']
                predictionsBasedOnKFoldsLightGBMGradientBoosting = preds.copy()
                preds["prediction_label"] = np.nan
                preds["prediction_label"].loc[preds["prediction"] >= 0.5] = 1
                preds["prediction_label"].loc[preds["prediction"] < 0.5] = 0
                preds["prediction_label"] = preds["prediction_label"].astype("int64")
                lgb_prec = precision_score(list(preds['trueLabel']), list(preds['prediction_label']))
                lgb_recall = recall_score(list(preds['trueLabel']), list(preds['prediction_label']))
                lgb_acc = accuracy_score(list(preds['trueLabel']), list(preds['prediction_label']))
                lGB_row.append([ftr, '1' + str(i), lgb_prec, lgb_recall,
                                lgb_acc, preds['trueLabel'].values, preds['prediction_label'].values, list(preds["prediction"])])  ### és akkor itt egy sor a csv-ben is egy sor lenne

                if calculate_shap:
                    make_shap_csv("lGB", shap_array, X_train, '1' + str(i), SHAP_LIST_TO_WRITE2)
            ##############################################################################################
            if "keras" in models_to_run:
                def root_mean_squared_error(y_true, y_pred):
                    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))

                scaler = MinMaxScaler(feature_range=(0.1, 0.9))
                X = scaler.fit_transform(X_train)
                X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

                y_ = tf.keras.utils.to_categorical(y_train)

                predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                                        index=y_train.index, columns=[0])

                shap_array = [[]] * len(X_train)
                for train_index, cv_index in k_fold.split(np.zeros(len(X_train))
                        , y_train.ravel()):


                    print(f"cv index: {cv_index}")

                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(8, activation='sigmoid', input_shape=(X.shape[1],)),
                        tf.keras.layers.Lambda(
                            lambda x: 0.8 * (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)) + 0.1),
                        tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax),

                        ])
                    print(model.summary())
                    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
                    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
                    model.compile(loss='categorical_crossentropy', optimizer=opt,
                                  metrics=["mse", root_mean_squared_error, "accuracy"])


                    X_train_fold, X_cv_fold = X_train_scaled.iloc[train_index, :], X_train_scaled.iloc[cv_index, :]
                    y_train_fold, y_cv_fold = y_[train_index], y_[cv_index]

                    model.fit(X_train_fold, y_train_fold, epochs=100, verbose=0, callbacks=[callback])
                    #loss = model.evaluate(X_cv_fold, y_cv_fold, verbose=0)
                    predictionsBasedOnKFolds.loc[X_cv_fold.index, :] = [[x] for x in model.predict(X_cv_fold)]
                    if calculate_shap:
                        calculate_SHAP(model, X_train_fold, X_cv_fold, cv_index, shap_array, title="keras", )

                preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 0]], axis=1)
                preds.columns = ['trueLabel', 'prediction']

                predslist = [0 if pr[0] > pr[1] else 1 for pr in list(preds['prediction'])]
                keras_prec = precision_score(list(preds['trueLabel']), predslist)
                keras_recall = recall_score(list(preds['trueLabel']), predslist)
                keras_acc = accuracy_score(list(preds['trueLabel']), predslist)

                keras_row.append([ftr, '1' + str(i), keras_prec, keras_recall, keras_acc, preds['trueLabel'].values,
                                   predslist, [pr[1] for pr in list(preds['prediction'])]])
                if calculate_shap:
                    make_shap_csv("keras", shap_array, X_train_scaled, '1' + str(i), SHAP_LIST_TO_WRITE2)

            ################################################################################################


            feature_row.append(ftr)
            comp_class.append('1' + str(i))

    if calculate_shap:
        idx = 0
        """
        for _d in SHAP_LIST_TO_WRITE:
            _d["idx"] = idx
            idx += 1

        Path(f"{out_dir}").mkdir(parents=True, exist_ok=True)
        with open(f"{out_dir}/mean_shap_values{additional_str}.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(SHAP_LIST_TO_WRITE)
        """
        for _d in SHAP_LIST_TO_WRITE2:
            _d["idx"] = idx
            idx += 1

        with open(f"{out_dir}/shap_values{additional_str}.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers2)
            writer.writeheader()
            writer.writerows(SHAP_LIST_TO_WRITE2)


    FEATURES_TO_WRITE_LIST = []
    headers3 = ["model","features", "compare_to",
    "precision", "recall", "accuracy", "trueLabel", "prediction", "prediction_1_chance"]

    if sort_by in headers3:
        index_to_sortby = headers3.index(sort_by) - 1
    else:
        index_to_sortby = None

    if "logReg" in models_to_run:
        if index_to_sortby:
            logreg_row.sort(key=lambda e:e[index_to_sortby], reverse=True)
        for i in range(len(logreg_row)):
            if logreg_row[i][4] > acc_lim[0] and logreg_row[i][3] > rec_lim[0] and logreg_row[i][2] > prec_lim[0]:
                dict_to_write = {}
                dict_to_write["model"] = "logReg"
                dict_to_write["features"] = logreg_row[i][0]
                dict_to_write["compare_to"] = logreg_row[i][1]
                dict_to_write["precision"] = logreg_row[i][2]
                dict_to_write["recall"] = logreg_row[i][3]
                dict_to_write["accuracy"] = logreg_row[i][4]
                dict_to_write["trueLabel"] = logreg_row[i][5]
                dict_to_write["prediction"] = logreg_row[i][6]
                dict_to_write["prediction_1_chance"] = logreg_row[i][7]
                FEATURES_TO_WRITE_LIST.append(dict_to_write)

    if "RFC" in models_to_run:
        if index_to_sortby:
            RFC_row.sort(key=lambda e: e[index_to_sortby], reverse=True)
        for i in range(len(RFC_row)):
            if RFC_row[i][4] > acc_lim[1] and RFC_row[i][3] > rec_lim[1] and RFC_row[i][2] > prec_lim[1]:
                dict_to_write = {}
                dict_to_write["model"] = "RFC"
                dict_to_write["features"] = RFC_row[i][0]
                dict_to_write["compare_to"] = RFC_row[i][1]
                dict_to_write["precision"] = RFC_row[i][2]
                dict_to_write["recall"] = RFC_row[i][3]
                dict_to_write["accuracy"] = RFC_row[i][4]
                dict_to_write["trueLabel"] = RFC_row[i][5]
                dict_to_write["prediction"] = RFC_row[i][6]
                dict_to_write["prediction_1_chance"] = RFC_row[i][7]
                FEATURES_TO_WRITE_LIST.append(dict_to_write)
    if "xGB" in models_to_run:
        if index_to_sortby:
            xGB_row.sort(key=lambda e: e[index_to_sortby], reverse=True)
        for i in range(len(xGB_row)):
            if xGB_row[i][4] >= acc_lim[2] and xGB_row[i][3] >= rec_lim[2] and xGB_row[i][2] >= prec_lim[2]:
                dict_to_write = {}
                dict_to_write["model"] = "xGB"
                dict_to_write["features"] = xGB_row[i][0]
                dict_to_write["compare_to"] = xGB_row[i][1]
                dict_to_write["precision"] = xGB_row[i][2]
                dict_to_write["recall"] = xGB_row[i][3]
                dict_to_write["accuracy"] = xGB_row[i][4]
                dict_to_write["trueLabel"] = xGB_row[i][5]
                dict_to_write["prediction"] = xGB_row[i][6]
                dict_to_write["prediction_1_chance"] = xGB_row[i][7]
                FEATURES_TO_WRITE_LIST.append(dict_to_write)
    if "lGB" in models_to_run:
        if index_to_sortby:
            lGB_row.sort(key=lambda e: e[index_to_sortby], reverse=True)
        for i in range(len(lGB_row)):
            if lGB_row[i][4] >= acc_lim[3] and lGB_row[i][3] >= rec_lim[3] and lGB_row[i][2] >= prec_lim[3]:
                dict_to_write = {}
                dict_to_write["model"] = "lGB"
                dict_to_write["features"] = lGB_row[i][0]
                dict_to_write["compare_to"] = lGB_row[i][1]
                dict_to_write["precision"] = lGB_row[i][2]
                dict_to_write["recall"] = lGB_row[i][3]
                dict_to_write["accuracy"] = lGB_row[i][4]
                dict_to_write["trueLabel"] = lGB_row[i][5]
                dict_to_write["prediction"] = lGB_row[i][6]
                dict_to_write["prediction_1_chance"] = lGB_row[i][7]
                FEATURES_TO_WRITE_LIST.append(dict_to_write)

    if "keras" in models_to_run:
        if index_to_sortby:
            keras_row.sort(key=lambda e: e[index_to_sortby], reverse=True)
        for i in range(len(keras_row)):
            if keras_row[i][4] > acc_lim[4] and keras_row[i][3] > rec_lim[4] and keras_row[i][2] > prec_lim[4]:
                dict_to_write = {}
                dict_to_write["model"] = "keras"
                dict_to_write["features"] = keras_row[i][0]
                dict_to_write["compare_to"] = keras_row[i][1]
                dict_to_write["precision"] = keras_row[i][2]
                dict_to_write["recall"] = keras_row[i][3]
                dict_to_write["accuracy"] = keras_row[i][4]
                dict_to_write["trueLabel"] = keras_row[i][5]
                dict_to_write["prediction"] = keras_row[i][6]
                dict_to_write["prediction_1_chance"] = keras_row[i][7]
                FEATURES_TO_WRITE_LIST.append(dict_to_write)

    Path(f"{out_dir}").mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/ML_model_values{additional_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers3)
        writer.writeheader()
        writer.writerows(FEATURES_TO_WRITE_LIST)

if __name__ == "__main__":

    sysargv_min = None
    sysargv_max = None
    sysargv_text = None
    sysargv_cc = None

    sysargv_features = None
    sysargv_cliques = None

    sysargv_shannon = None
    sysargv_shancol = None

    # command line arguments if ran by another program else you need to specify them

    sysargv_min = int(sys.argv[1])
    sysargv_max = int(sys.argv[2])
    sysargv_text = sys.argv[3]
    sysargv_cc = int(sys.argv[4])

    sargv_features = sys.argv[5]
    sysargv_cliques = sys.argv[6]



    models_to_run = ["keras", "logReg"]

    ################################################################################################################################

    #dataframe for the feature values
    # c_features_dir = "features.csv"
    features_df = pd.read_csv(sargv_features)


    print(features_df.shape[0])


    print(features_df.shape[0])

    #dataframe for the cliques
    cols_list_df = pd.read_csv(
        sysargv_cliques)
    # cols_list_df = cols_list_df.sort_values(by="max_weight", ascending=False)[:2]
    print(cols_list_df)
    # cols_list_dir = "colslist.csv"
    _out_dir = "SHAPML/NEW"
    _subset_to_run = 0
    #calculte shap values y/n
    calc_shap = True
    #which models to run
    #models_to_run = ["keras"]
    # [logreg_limit, rfc_limit, xgb_limit, lgb_limit, keras_limit]
    accuracy_limit = [0.0, 0.0, 0.0, 0.0, 0.0]
    recall_limit = [0.0, 0.0, 0.0, 0.0, 0.0]
    precision_limit = [0.0, 0.0, 0.0, 0.0, 0.0]

    features_df = prepare_dataframe(features_df)

    cols_list = make_features(cols_list_df, subset_to_run=0, group_by=None,
                              group_list=None, sort_by="accuracy",
                              ascending=False, slice_range=(0, 50))

    random.seed(42)
    cols_list = random.sample(cols_list, min(2000,len(cols_list)))

    if sysargv_min > len(cols_list):
        sys.exit()

    outfile_name_str = "outfile_"

    max_lim = sysargv_max if sysargv_max < len(cols_list) else len(cols_list)

    run_ML(features_df, cols_list[sysargv_min: max_lim], _out_dir, models_to_run, calc_shap,
           accuracy_limit, recall_limit, precision_limit, sysargv_text, [sysargv_cc],
           "accuracy", )