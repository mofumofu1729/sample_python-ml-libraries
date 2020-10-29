#!/usr/bin/env python3
# coding: utf-8
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn import tree

import lightgbm as lgb

from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids


def imbalanced_data_split(X, y, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test


def lgbm_train(X_train_df, X_valid_df, y_train_df, y_valid_df, lgbm_params):
    lgb_train = lgb.Dataset(X_train_df, y_train_df)
    lgb_eval = lgb.Dataset(X_valid_df, y_valid_df)

    model = lgb.train(lgbm_params, lgb_train,
                      valid_sets=lgb_eval,
                      num_boost_round=1000,
                      early_stopping_rounds=10)

    return model


def eval_model(model, X_test, y_test):
    y_pred = \
        model.predict(X_test,
                      num_iteration=model.best_iteration)

    accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
    micro_f1 = f1_score(y_test, np.argmax(y_pred, axis=1),
                        average='micro')
    macro_f1 = f1_score(y_test, np.argmax(y_pred, axis=1),
                        average='macro')

    print(accuracy)
    print(micro_f1)
    print(macro_f1)


NUM_CLASSES = 5


def main():
    args = {
        'n_samples': 100000,
        'n_features': 20,
        'n_informative': 3,
        'n_redundant': 0,
        'n_repeated': 0,
        'n_classes': NUM_CLASSES,
        'n_clusters_per_class': 1,
        'weights': [100, 40, 10, 5, 3],
        'random_state': 42,
    }

    X, y = make_classification(**args)

    X_train, X_test, y_train, y_test = imbalanced_data_split(X, y,
                                                             test_size=0.2)
    # for validation
    X_train2, X_valid, y_train2, y_valid = imbalanced_data_split(X_train,
                                                                 y_train,
                                                                 test_size=0.2)
    lgbm_params = {
        'learning_rate': 0.1,
        'num_leaves': 8,
        'boosting_type': 'gbdt',
        'reg_alpha': 1,
        'reg_lambda': 1,
        'objective': 'multiclass',
        'metrics': 'multi_logloss',
        'num_class': NUM_CLASSES,

    }
    model_normal = lgbm_train(X_train2, X_valid,
                              y_train2, y_valid,
                              lgbm_params)

    eval_model(model_normal, X_test, y_test)

    # downsampling
    sampler = RandomUnderSampler(random_state=42)

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    X_train2, X_valid, y_train2, y_valid = imbalanced_data_split(X_resampled,
                                                                 y_resampled,
                                                                 test_size=0.2)

    model_under_sample = lgbm_train(X_train2, X_valid,
                                    y_train2, y_valid,
                                    lgbm_params)

    eval_model(model_under_sample, X_test, y_test)

    # 決定木
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train2, y_train2)

    predict_test = clf.predict(X_test)
    print(accuracy_score(y_test, predict_test))


if __name__ == "__main__":
    main()
