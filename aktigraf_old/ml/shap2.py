from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
color = sns.color_palette()
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, precision_score
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score

import csv
from pathlib import Path
import warnings
import random

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
xgb.set_config(verbosity=0)

def calculate_SHAP(title_name, model, X_train, _class, SHAP_LIST_TO_WRITE, SHAP_LIST_TO_WRITE2):
    try:
        if title_name == "logReg":
            explainer = shap.Explainer(model, X_train)
        else:
            explainer = shap.TreeExplainer(model)
        sv = explainer(X_train)
        means = []
        for i in range(len(X_train.columns)):  # hány legjobb featuret néz
            means.append(np.mean([abs(v[i]) for v in sv.values]))
        dict_to_write = {}
        dict_to_write["title"] = title_name
        dict_to_write["columns"] = X_train.iloc[:len(sv.values), :].columns.tolist()
        dict_to_write["mean_SHAP_values"] = means
        dict_to_write["comp_class"] = _class
        SHAP_LIST_TO_WRITE.append(dict_to_write)

        dict_to_write2 = {}
        dict_to_write2["title"] = title_name
        dict_to_write2["columns"] = X_train.iloc[:len(sv.values), :].columns.tolist()
        dict_to_write2["SHAP_values"] = sv.values
        dict_to_write2["comp_class"] = _class
        SHAP_LIST_TO_WRITE2.append(dict_to_write2)
    except:
        print("FAILED")
        print("FAILED")
        print("FAILED")


def run_ML(features_path, cols_list_path, out_dir, subset_to_run=800, calculate_shap=False, acc_lim=[0.0, 0.0, 0.0, 0.0], rec_lim=[0.0, 0.0, 0.0, 0.0], prec_lim=[0.0, 0.0, 0.0, 0.0], additional_str=""):

    SHAP_LIST_TO_WRITE = []
    SHAP_LIST_TO_WRITE2 = []
    headers = ["idx", "title", "columns", "mean_SHAP_values", "comp_class"]
    headers2 = ["idx", "title", "columns", "comp_class", "SHAP_values"]

    features = pd.read_csv(features_path)
    features.drop(["Unnamed: 0"], axis=1, inplace=True)


    cols_list_df = pd.read_csv(cols_list_path)

    columns = cols_list_df.columns.values
    columns = [c for c in columns if not c.startswith("col_")]
    cols_list_df = cols_list_df[columns]
    cols_list = cols_list_df.feature.values.tolist()
    if subset_to_run > 0:
        cols_list = random.sample(cols_list, k=subset_to_run)
    cols_list_filtered = []
    for c in cols_list:
        c2 = c.replace("[", "")
        c2 = c2.replace("]", "")
        c2 = c2.replace("'", "")
        c2 = c2.replace(" ", "")
        cols_list_filtered.append(c2.split(","))
    print(cols_list_filtered[0])



    feature_row = []
    comp_class = []
    logreg_row = []
    RFC_row = []
    xGB_row = []
    lGB_row = []


    for n in range(len(cols_list_filtered)):
        print(f"{n + 1}/{len(cols_list_filtered)}")
        print(f"{n + 1}/{len(cols_list_filtered)}")
        print(f"{n + 1}/{len(cols_list_filtered)}")
        ftr = cols_list_filtered[n]

        for i in [2, 3]:
            data = features[ftr + ["class"]].copy()
            data = data.loc[(features["class"] == i) | (features["class"] == 1)]
            data["class"][data["class"] == 1] = 0
            data["class"][data["class"] == i] = 1

            X_train = data.drop(['class'], axis=1)
            y_train = data["class"]
            k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)
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
            model = logReg
            for train_index, cv_index in k_fold.split(np.zeros(len(X_train))
                    , y_train.ravel()):
                X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
                y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
                model.fit(X_train_fold, y_train_fold)
                predictionsBasedOnKFolds.loc[X_cv_fold.index, :] = [[x] for x in model.predict(X_cv_fold)]
            if calculate_shap:
                calculate_SHAP("logReg", model, X_train, '1' + str(i), SHAP_LIST_TO_WRITE, SHAP_LIST_TO_WRITE2)
            preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 0]], axis=1)
            preds.columns = ['trueLabel', 'prediction']
            predictionsBasedOnKFoldsRandomForests = preds.copy()
            logreg_prec = precision_score(list(preds['trueLabel']), list(preds['prediction']))
            logreg_recall = recall_score(list(preds['trueLabel']), list(preds['prediction']))
            logreg_acc = accuracy_score(list(preds['trueLabel']), list(preds['prediction']))


            ####################################################################################################################################
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
            model = RFC
            for train_index, cv_index in k_fold.split(np.zeros(len(X_train)),
                                                      y_train.ravel()):
                X_train_fold, X_cv_fold = X_train.iloc[train_index, :], X_train.iloc[cv_index, :]
                y_train_fold, y_cv_fold = y_train.iloc[train_index], y_train.iloc[cv_index]
                model.fit(X_train_fold, y_train_fold)
                predictionsBasedOnKFolds.loc[X_cv_fold.index, :] = [[x] for x in model.predict(X_cv_fold)]
            if calculate_shap:
                calculate_SHAP("RFC", model, X_train, '1' + str(i), SHAP_LIST_TO_WRITE, SHAP_LIST_TO_WRITE2)
            preds = pd.concat([y_train, predictionsBasedOnKFolds.loc[:, 0]], axis=1)
            preds.columns = ['trueLabel', 'prediction']
            predictionsBasedOnKFoldsRandomForests = preds.copy()
            rfc_prec = precision_score(list(preds['trueLabel']), list(preds['prediction']))
            rfc_recall = recall_score(list(preds['trueLabel']), list(preds['prediction']))
            rfc_acc = accuracy_score(list(preds['trueLabel']), list(preds['prediction']))
            ####################################################################################################################################
            """
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
            g_bst = None
            predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                                    index=y_train.index, columns=['prediction'])
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
                g_bst = bst
            if calculate_shap:
                calculate_SHAP("xGB", g_bst, X_train, '1' + str(i), SHAP_LIST_TO_WRITE, SHAP_LIST_TO_WRITE2)
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
            """
            ####################################################################################################################################
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
            g_gbm = None
            predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                                    index=y_train.index, columns=['prediction'])
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
                predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'] = gbm.predict(X_cv_fold,
                                                                                          num_iteration=gbm.best_iteration)
                loglossCV = log_loss(y_cv_fold, predictionsBasedOnKFolds.loc[X_cv_fold.index, 'prediction'])
                cvScores.append(loglossCV)
                g_gbm = gbm
            if calculate_shap:
                calculate_SHAP("lGB", g_gbm, X_train, '1' + str(i), SHAP_LIST_TO_WRITE, SHAP_LIST_TO_WRITE2)
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
            ##############################################################################################

            feature_row.append(ftr)
            comp_class.append('1' + str(i))
            logreg_row.append([ftr,'1' + str(i),logreg_prec, logreg_recall, logreg_acc])
            RFC_row.append([ftr,'1' + str(i),rfc_prec, rfc_recall, rfc_acc])
            #xGB_row.append([ftr,'1' + str(i),xgb_prec, xgb_recall, xgb_acc])
            lGB_row.append([ftr,'1' + str(i),lgb_prec, lgb_recall, lgb_acc]) ### és akkor itt egy sor a csv-ben is egy sor lenne

    if calculate_shap:
        idx = 0
        for _d in SHAP_LIST_TO_WRITE:
            _d["idx"] = idx
            idx += 1

        Path(f"{out_dir}").mkdir(parents=True, exist_ok=True)
        with open(f"{out_dir}/mean_shap_values{additional_str}.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(SHAP_LIST_TO_WRITE)

        for _d in SHAP_LIST_TO_WRITE2:
            _d["idx"] = idx
            idx += 1

        with open(f"{out_dir}/shap_values{additional_str}.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers2)
            writer.writeheader()
            writer.writerows(SHAP_LIST_TO_WRITE2)


    FEATURES_TO_WRITE_LIST = []
    headers3 = ["model","features", "compare_to",
    "precision", "accuracy", "recall"
                ]

    for i in range(len(logreg_row)):
        if logreg_row[i][4] > acc_lim[0] and logreg_row[i][3] > rec_lim[0] and logreg_row[i][2] > prec_lim[0]:
            dict_to_write = {}
            dict_to_write["model"] = "logReg"
            dict_to_write["features"] = logreg_row[i][0]
            dict_to_write["compare_to"] = logreg_row[i][1]
            dict_to_write["precision"] = logreg_row[i][2]
            dict_to_write["recall"] = logreg_row[i][3]
            dict_to_write["accuracy"] = logreg_row[i][4]
            FEATURES_TO_WRITE_LIST.append(dict_to_write)

    for i in range(len(RFC_row)):
        if RFC_row[i][4] > acc_lim[1] and RFC_row[i][3] > rec_lim[1] and RFC_row[i][2] > prec_lim[1]:
            dict_to_write = {}
            dict_to_write["model"] = "RFC"
            dict_to_write["features"] = RFC_row[i][0]
            dict_to_write["compare_to"] = RFC_row[i][1]
            dict_to_write["precision"] = RFC_row[i][2]
            dict_to_write["recall"] = RFC_row[i][3]
            dict_to_write["accuracy"] = RFC_row[i][4]
            FEATURES_TO_WRITE_LIST.append(dict_to_write)
    """
    for i in range(len(xGB_row)):
        if xGB_row[i][4] > acc_lim[2] and xGB_row[i][3] > rec_lim[2] and xGB_row[i][2] > prec_lim[2]:
            dict_to_write = {}
            dict_to_write["model"] = "xGB"
            dict_to_write["features"] = xGB_row[i][0]
            dict_to_write["compare_to"] = xGB_row[i][1]
            dict_to_write["precision"] = xGB_row[i][2]
            dict_to_write["recall"] = xGB_row[i][3]
            dict_to_write["accuracy"] = xGB_row[i][4]
            FEATURES_TO_WRITE_LIST.append(dict_to_write)
"""
    for i in range(len(lGB_row)):
        if lGB_row[i][4] > acc_lim[3] and lGB_row[i][3] > rec_lim[3] and lGB_row[i][2] > prec_lim[3]:
            dict_to_write = {}
            dict_to_write["model"] = "lGB"
            dict_to_write["features"] = lGB_row[i][0]
            dict_to_write["compare_to"] = lGB_row[i][1]
            dict_to_write["precision"] = lGB_row[i][2]
            dict_to_write["recall"] = lGB_row[i][3]
            dict_to_write["accuracy"] = lGB_row[i][4]
            FEATURES_TO_WRITE_LIST.append(dict_to_write)

    Path(f"{out_dir}").mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/ML_model_values{additional_str}.csv", 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers3)
        writer.writeheader()
        writer.writerows(FEATURES_TO_WRITE_LIST)


if __name__ == "__main__":

    c_features_dir = "features.csv"
    cols_list_dir = "colslist.csv"
    _out_dir = "SHAPML"
    _subset_to_run = 0
    calc_shap = False

    # [logreg_limit, rfc_limit, xgb_limit, lgb_limit ]
    accuracy_limit = [0.0, 0.0, 0.0, 0.0]
    recall_limit = [0.0, 0.0, 0.0, 0.0]
    precision_limit = [0.0, 0.0, 0.0, 0.0]

    i=1
    run_ML(c_features_dir, cols_list_dir, _out_dir, _subset_to_run, calc_shap,
                                         accuracy_limit, recall_limit, precision_limit, f"_20221021")

