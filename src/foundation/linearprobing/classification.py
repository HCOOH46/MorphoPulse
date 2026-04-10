
import numpy as np
import pandas as pd
import torch
import json
from utils import bootstrap_metric_confidence_interval
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

def classification_model(estimator, param_grid, X_train, y_train, X_test, y_test, bootstrap=True, custom_threshold=0.5, save_probe_path=None):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=estimator, 
                        param_grid=param_grid, 
                        cv=4, 
                        scoring='accuracy', 
                        verbose=2, 
                        n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    try:
        # 优先尝试获取概率 (适用于 LR, RF, MLP)
        y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    except AttributeError:
        # 如果模型没有 predict_proba (如 RidgeClassifier)，则使用 decision_function 获取置信度分数
        y_pred_proba = grid_search.decision_function(X_test)

    y_pred_adjusted = (y_pred_proba >= custom_threshold).astype(int)

    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        ret = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        return ret
    

    best_model = grid_search.best_estimator_

    if save_probe_path is not None:
        if not hasattr(best_model, "coef_"):
            raise ValueError(
                f"Saving CAM-compatible probe requires a linear classifier with coef_. Got: {type(best_model)}"
            )

        probe_bundle = {
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "coef": best_model.coef_,
            "intercept": best_model.intercept_,
            "classes": best_model.classes_,
            "best_params": grid_search.best_params_,
        }
        joblib.dump(probe_bundle, save_probe_path)
        print(f"[INFO] Saved probe bundle to {save_probe_path}")


    lower_bound_ci_auc, upper_bound_ci_auc, lower_bound_ci_f1, upper_bound_ci_f1 = -999, -999, -999, -999 
    if bootstrap:

        lower_bound_ci_auc, upper_bound_ci_auc, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                                y_pred=y_pred_proba,
                                                                                                metric_func=roc_auc_score)

        lower_bound_ci_f1, upper_bound_ci_f1, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                        y_pred=y_pred_adjusted,
                                                                                        metric_func=f1_score) 

        lower_bound_ci_sen, upper_bound_ci_sen, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                        y_pred=y_pred_adjusted,
                                                                                        metric_func=recall_score)       

        lower_bound_ci_spe, upper_bound_ci_spe, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                        y_pred=y_pred_adjusted,
                                                                                        metric_func=specificity_score)                                                                                

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_adjusted).ravel()
    sensitivity = recall_score(y_test, y_pred_adjusted)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    results = {'parameters': grid_search.best_params_,
            'auc': roc_auc_score(y_test, y_pred_proba),
            'auc_lower_ci': lower_bound_ci_auc,
            'auc_upper_ci': upper_bound_ci_auc,
            'f1': f1_score(y_test, y_pred_adjusted),
            'f1_lower_ci': lower_bound_ci_f1,
            'f1_upper_ci': upper_bound_ci_f1,
            'sensitivity': sensitivity,
            'sensitivity_upper_ci': upper_bound_ci_sen,
            'sensitivity_lower_ci': lower_bound_ci_sen,
            'specificity': specificity,
            'specificity_upper_ci': upper_bound_ci_spe,
            'specificity_lower_ci': lower_bound_ci_spe,
            'y_test': json.dumps(np.asarray(y_test).tolist()),
            'y_pred_proba':json.dumps(y_pred_proba.tolist()),
            'y_pred': json.dumps(y_pred.tolist()),
            'y_pred_adjusted': json.dumps(y_pred_adjusted.tolist()),}

    return results
