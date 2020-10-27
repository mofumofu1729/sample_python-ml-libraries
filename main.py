#!/usr/bin/env python3
# coding: utf-8
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

import lightgbm as lgb

from sklearn import tree


def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    # 訓練データとテストデータに分割する
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # データセットを生成する
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # LightGBM のハイパーパラメータ
    lgbm_params = {
        # 多値分類問題
        'objective': 'multiclass',
        # クラス数は 3
        'num_class': 3,
    }

    # 上記のパラメータでモデルを学習する
    model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

    # テストデータを予測する
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする

    # 精度 (Accuracy) を計算する
    accuracy = sum(y_test == y_pred_max) / len(y_test)
    print(accuracy)

    #
    # 決定木
    #
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_train, y_train)
    import pdb; pdb.set_trace()
    
    predict_test = clf.predict(X_test)
    print(metrics.accuracy_score(y_test, predict_test))


if __name__ == "__main__":
    main()
