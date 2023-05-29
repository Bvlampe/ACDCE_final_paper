import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

def query_yn(question):
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(question)
    return True if answer == "y" else False

def models():
    main_data = pd.read_csv("merged_data.csv", index_col=[0, 1])

    # Create DV as change of democracy over time
    time_lag = 5
    for (ctry, year) in main_data.index:
        demchg = main_data.loc[(ctry, year + time_lag), "Democracy"] -\
                 main_data.loc[(ctry, year), "Democracy"] if year + time_lag \
                 in main_data.index.get_level_values(1) else np.NaN
        main_data.loc[(ctry, year), "Democracy change"] = demchg

    main_data.drop(["Democracy"], axis=1, inplace=True)

    for col in ["Political Contestation", "Eligible population"]:
        main_data[col] = main_data.groupby(level=0)[col].apply(
            lambda group: group.fillna(method="ffill", limit=10))

    main_data.dropna(inplace=True)

    main_data["Democracy decrease"] = np.where(main_data["Democracy change"] < -1, 1, 0)
    main_data.drop(["Democracy change"], axis=1, inplace=True)

    main_data['Year'] = main_data.index.get_level_values(1)
    n_splits = 5

    tss = TimeSeriesSplit(n_splits=n_splits)
    model_logreg = LogisticRegression(max_iter=1000)
    model_rf = RandomForestClassifier()
    model_gbm = GradientBoostingClassifier()

    indep_vars = list(main_data.columns.values)
    indep_vars.remove("Democracy decrease")
    log = pd.DataFrame(index=indep_vars)

    i = 1
    for train_index, test_index in tss.split(main_data['Year'].unique()):
        print("Fold number", i, ":")
        # Get the training and testing data for this split
        # x_train, x_test = main_data.iloc[train_index][indep_vars], main_data.iloc[test_index][indep_vars]
        # y_train, y_test = main_data.iloc[train_index]["Terrorist attack"], main_data.iloc[test_index]["Terrorist attack"]

        x_train, x_test = main_data.loc[main_data['Year'].isin(main_data['Year'].unique()[train_index])], main_data.loc[
            main_data['Year'].isin(main_data['Year'].unique()[test_index])]
        x_train.columns = x_train.columns.tolist()
        x_test.columns = x_test.columns.tolist()

        y_train, y_test = x_train["Democracy decrease"], x_test["Democracy decrease"]
        x_train = x_train.drop(["Democracy decrease", "Year"], axis=1)
        x_test = x_test.drop(["Democracy decrease", "Year"], axis=1)

        model_logreg.fit(x_train, y_train)
        y_pred = model_logreg.predict(x_test)
        print("Logistic regression:")
        # print(classification_report(y_test, y_pred))
        # print(confusion_matrix(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred), "Recall:", recall_score(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_logreg.predict_proba(x_test)[:, 1]))
        print()
        # log.loc[:len(model_logreg.feature_importances_),f"LR Fold {i}"] = model_logreg.feature_importances_
        log.loc["Accuracy", f"LR Fold {i}"] = accuracy_score(y_test, y_pred)
        log.loc["Precision", f"LR Fold {i}"] = precision_score(y_test, y_pred)
        log.loc["Recall", f"LR Fold {i}"] = recall_score(y_test, y_pred)
        log.loc["ROC-AUC", f"LR Fold {i}"] = roc_auc_score(y_test, model_logreg.predict_proba(x_test)[:, 1])

        model_rf.fit(x_train, y_train)
        y_pred = model_rf.predict(x_test)
        print("Random forest:")
        print("Precision:", precision_score(y_test, y_pred), "Recall:", recall_score(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_rf.predict_proba(x_test)[:, 1]))
        print()
        log.loc[:len(model_rf.feature_importances_), f"RF Fold {i}"] = model_rf.feature_importances_
        log.loc["Accuracy", f"RF Fold {i}"] = accuracy_score(y_test, y_pred)
        log.loc["Precision", f"RF Fold {i}"] = precision_score(y_test, y_pred)
        log.loc["Recall", f"RF Fold {i}"] = recall_score(y_test, y_pred)
        log.loc["ROC-AUC", f"RF Fold {i}"] = roc_auc_score(y_test, model_rf.predict_proba(x_test)[:, 1])

        model_gbm.fit(x_train, y_train)
        y_pred = model_gbm.predict(x_test)
        print("Gradient boosting machine:")
        print("Precision:", precision_score(y_test, y_pred), "Recall:", recall_score(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("ROC-AUC-score: ", roc_auc_score(y_test, model_gbm.predict_proba(x_test)[:, 1]))

        # The feature_importances_ attribute measures the "mean and standard deviation of accumulation
        # of the impurity decrease within each tree" - scikit-learn doc
        for feature, v in zip(x_train.columns, model_gbm.feature_importances_):
            print(f"Feature: {feature}, Score: %.5f" % (v))

        log.loc[:len(model_gbm.feature_importances_), f"GBM Fold {i}"] = model_gbm.feature_importances_
        log.loc["Accuracy", f"GBM Fold {i}"] = accuracy_score(y_test, y_pred)
        log.loc["Precision", f"GBM Fold {i}"] = precision_score(y_test, y_pred)
        log.loc["Recall", f"GBM Fold {i}"] = recall_score(y_test, y_pred)
        log.loc["ROC-AUC", f"GBM Fold {i}"] = roc_auc_score(y_test, model_gbm.predict_proba(x_test)[:, 1])
        print("--------------------------------------------------------")
        i += 1
    print("Log file:\n", log)
    if query_yn("Write model output to log file? y/n: "):
        log.to_csv("output_files/output.csv")