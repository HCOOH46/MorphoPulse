
import pandas as pd 
import numpy as np
import sys 
import json
import os
import argparse
from classification import classification_model
from regression import regression_model
from utils import load_linear_probe_dataset_objs, bootstrap_metric_confidence_interval, get_data_for_ml, get_data_for_ml_from_df
from utilities import get_data_info
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from time import time
from datetime import datetime
import joblib
from project_paths import PROBE_ROOT, ensure_dir


def multiclass_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    return float(np.nanmean(specificities))


def to_jsonable_string(obj):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    return json.dumps(obj)

def binary_classification(dataset_name, model_name, linear_model, label, func, content, concat, level="patient", string_convert=True, percent=None, custom_threshold=0.5, save_probe_path=None):

    if concat:
        X_train, y_train, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                                         model_name=model_name,
                                                                         label=label,
                                                                         func=func,
                                                                         level=level,
                                                                         string_convert=string_convert,
                                                                         content=content)   
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                             model_name=model_name,
                                                             label=label,
                                                             func=func,
                                                             level=level,
                                                             concat=False,
                                                             string_convert=string_convert,
                                                             content=content)
        X_test = np.concatenate((X_test, X_val))
        y_test = np.concatenate((y_test, y_val))
        
    if percent is not None:
        size = int(len(X_train))
        idx = np.random.choice(np.arange(size), size=int(percent * size), replace=False)  # Add replace=False if necessary
        print(f"Selected indices: {idx}")
        
        # Ensure X_train and y_train are numpy arrays
        X_train = np.array(X_train)[idx.astype(int)]
        y_train = np.array(y_train)[idx.astype(int)]
        print(f"Using {len(X_train)} of {size}")
        
        
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"[DEBUG] {dataset_name}-{label} class dist:", dict(zip(unique, counts)))
    if unique.size < 2:
        raise ValueError(
            f"{dataset_name}-{label} y_train 只有 {unique[0]}，请检查 load_linear_probe_dataset_objs 的筛选逻辑或底层特征文件。"
        )
        
        
    # Define the parameter grid
    if linear_model == "lr":
        # estimator = LogisticRegression()
        estimator = LogisticRegression(class_weight='balanced')

        param_grid = [{
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs'],
            'max_iter': [2000, 500, 1000]
        },
        {
            'penalty': [None],
            'solver': ['lbfgs'],
            'max_iter': [2000, 500, 1000]
        }]


    if linear_model == "rf":
        estimator = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200],    
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30],     
            'min_samples_split': [2, 5],     
            'min_samples_leaf': [1, 2],      
            }
    elif linear_model == "rf_n":
        estimator = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 200, 100, 160],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 7, 10],
            'min_samples_leaf': [1, 2, 4, 7]
        }

    elif linear_model == "ridge":
        estimator = RidgeClassifier()
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'sag']
        }
    elif linear_model == "mlp":
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier())
        ])
        param_grid = {
            'clf__hidden_layer_sizes': [(100,), (100, 50)],
            'clf__activation': ['logistic', 'relu', 'tanh'],
            'clf__alpha': [0.0001, 0.001, 0.0005, 0.002],
            'clf__max_iter': [200, 500]
        }

    results = classification_model(estimator=estimator,
                                param_grid=param_grid,
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                bootstrap=True,
                                custom_threshold=custom_threshold,
                                save_probe_path=save_probe_path,)

    results['test_keys'] = to_jsonable_string(list(test_keys))
    results['model'] = model_name
    results['dataset'] = dataset_name
    results['label'] = label
    
    return results

def multilabel_classification(dataset_name, model_name, label, func, content, concat, level="patient", string_convert=True, linear=True, custom_threshold=0.5, save_probe_path=None):

    if concat:
        X_train, y_train, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                                         model_name=model_name,
                                                                         label=label,
                                                                         func=func,
                                                                         level=level,
                                                                         string_convert=string_convert,
                                                                         content=content)   
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, test_keys = load_linear_probe_dataset_objs(dataset_name=dataset_name,
                                                             model_name=model_name,
                                                             label=label,
                                                             func=func,
                                                             level=level,
                                                             concat=False,
                                                             string_convert=string_convert,
                                                             content=content)
        X_test = np.concatenate((X_test, X_val))
        y_test = np.concatenate((y_test, y_val))
    
    if linear:
        estimator = LogisticRegression(class_weight='balanced')

        param_grid = [{
            'penalty': ['l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs'],
            'max_iter': [2000, 500, 1000]
        },
        {
            'penalty': [None],
            'solver': ['lbfgs'],
            'max_iter': [2000, 500, 1000]
        }]

    else:
        estimator = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200],    
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30],     
            'min_samples_split': [2, 5],     
            'min_samples_leaf': [1, 2],      
            }
                                           
                                                                                     
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    grid_search = GridSearchCV(estimator=estimator, 
                        param_grid=param_grid, 
                        cv=StratifiedKFold(n_splits=3), 
                        scoring='accuracy', 
                        verbose=2, 
                        n_jobs=-1)

    grid_search.fit(X_train, y_train)
    y_pred_proba = grid_search.predict_proba(X_test)

    classes, counts = np.unique(y_train, return_counts=True)
    priors = counts / len(y_train) 
    
    print(f"[DEBUG] 训练集各类别先验比例: {dict(zip(classes, priors))}")
    adjusted_proba = y_pred_proba / priors
    
    y_pred_adjusted = grid_search.classes_[np.argmax(adjusted_proba, axis=1)]

    
    y_pred = grid_search.predict(X_test)
    y_pred_proba = grid_search.predict_proba(X_test)

    # --- 计算 Accuracy ---
    acc = accuracy_score(y_test, y_pred)
    lower_bound_ci_acc, upper_bound_ci_acc, _ = bootstrap_metric_confidence_interval(y_test=np.array(y_test),
                                                                                    y_pred=np.array(y_pred),
                                                                                    metric_func=accuracy_score)     
    
    # --- 计算 F1 Score (Macro & Weighted) ---
    f1_macro = f1_score(y_test, y_pred_adjusted, average='macro')
    f1_weighted = f1_score(y_test, y_pred_adjusted, average='weighted')
    sensitivity = recall_score(y_test, y_pred_adjusted, average='macro')
    specificity = multiclass_specificity(y_test, y_pred_adjusted)
    
    # 计算 F1 Macro 的置信区间
    lower_bound_ci_f1, upper_bound_ci_f1, _ = bootstrap_metric_confidence_interval(
        y_test=np.array(y_test),
        y_pred=np.array(y_pred_adjusted),
        metric_func=lambda y_t, y_p: f1_score(y_t, y_p, average='macro')
    )

    # --- 计算 AUROC (One-vs-Rest) ---
    try:
        classes = np.unique(y_train)
        n_classes = len(classes)
        
        if n_classes == 2:
            # 二分类情况
            auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
            # 计算 AUROC 的置信区间
            lower_bound_ci_auroc, upper_bound_ci_auroc, _ = bootstrap_metric_confidence_interval(
                y_test=np.array(y_test),
                y_pred=y_pred_proba[:, 1],
                metric_func=roc_auc_score
            )
        else:
            # 多分类情况：需要将标签二值化,或者使用 multi_class='ovr'
            # 方法 A: 直接使用 sklearn 的 ovr 支持 (推荐)
            auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            # 计算 AUROC 的置信区间
            lower_bound_ci_auroc, upper_bound_ci_auroc, _ = bootstrap_metric_confidence_interval(
                y_test=np.array(y_test),
                y_pred=y_pred_proba,
                metric_func=lambda y_t, y_p: roc_auc_score(y_t, y_p, multi_class='ovr', average='macro')
            )


    except Exception as e:
        print(f"[WARN] AUROC calculation failed: {e}")
        auroc = None

    recall_mul = lambda y_true, y_pred : recall_score(y_true, y_pred, average='macro')

    lower_bound_ci_sen, upper_bound_ci_sen, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                    y_pred=y_pred_adjusted,
                                                                                    metric_func=recall_mul)       

    lower_bound_ci_spe, upper_bound_ci_spe, _ = bootstrap_metric_confidence_interval(y_test=y_test,
                                                                                    y_pred=y_pred_adjusted,
                                                                                    metric_func=multiclass_specificity)  

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
    results = {
        'parameters': grid_search.best_params_,
        'auc': auroc,
        'auc_lower_ci': lower_bound_ci_auroc,
        'auc_upper_ci': upper_bound_ci_auroc,
        'f1': f1_macro,  # 二分类叫f1，多分类也叫f1对接上
        'f1_lower_ci': lower_bound_ci_f1,
        'f1_upper_ci': upper_bound_ci_f1,
        'sensitivity': sensitivity,
        'sensitivity_upper_ci': upper_bound_ci_sen,
        'sensitivity_lower_ci': lower_bound_ci_sen,
        'specificity': specificity,
        'specificity_upper_ci': upper_bound_ci_spe,
        'specificity_lower_ci': lower_bound_ci_spe,
        'y_test': to_jsonable_string(np.asarray(y_test)),
        'y_pred_proba': to_jsonable_string(np.asarray(y_pred_proba)),
        'y_pred': to_jsonable_string(np.asarray(y_pred)),
        'y_pred_adjusted': to_jsonable_string(np.asarray(y_pred_adjusted)),
        
        # 二分类没有但多分类特有的指标放后面
        # 'acc': acc,
        # 'acc_lower_ci': lower_bound_ci_acc,
        # 'acc_upper_ci': upper_bound_ci_acc,
        # 'f1_weighted': f1_weighted,
        
        'test_keys': to_jsonable_string(list(test_keys)),
        'model': model_name,
        'dataset': dataset_name,
        'label': label
    }

    return results

def get_results(model, config):
    all_results = []
    func = get_data_for_ml
    for key in config.keys():
        configuration = config[key]
        print(f"{configuration['dataset']} | {configuration['label']}")
        print("######################")

        probe_dir = ensure_dir(PROBE_ROOT / configuration['dataset'])

        probe_path = os.path.join(
            probe_dir,
            f"{model}_{configuration['label']}_{configuration['linear_model']}.joblib"
        )

        if configuration['classification_type'] == "binary":
            results = binary_classification(dataset_name=configuration['dataset'],
                                           model_name=model,
                                           linear_model=configuration['linear_model'],
                                           label=configuration['label'],
                                           func=func,
                                           content=configuration['content'],
                                           level=configuration['level'],
                                           string_convert=configuration['string_convert'],
                                           concat=configuration['concat'],
                                           percent=configuration['percent'],
                                           custom_threshold=configuration['y_pred_adjusted'],
                                           save_probe_path=probe_path)
            
        if configuration['classification_type'] == "multi":
            results = multilabel_classification(dataset_name=configuration['dataset'],
                                           model_name=model,
                                           label=configuration['label'],
                                           func=func,
                                           content=configuration['content'],
                                           level=configuration['level'],
                                           string_convert=configuration['string_convert'],
                                           concat=configuration['concat'],
                                           save_probe_path=probe_path)
        all_results.append(results)
        
    return all_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="model directory")
    parser.add_argument('classification_type', type=str)
    args = parser.parse_args()
    percent = None

    if args.classification_type == "binary":
        config = {
            0: {"dataset": "sdb", "label": "label", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent, 'y_pred_adjusted':0.5},
            1: {"dataset": "ppg-bp", "label": "Hypertension", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent, 'y_pred_adjusted':0.6},
            2: {"dataset": "ppg-bp", "label": "Diabetes", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent, 'y_pred_adjusted':0.25},
            3: {"dataset": "mimic_af", "label": "label", "classification_type": "binary", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False, 'percent': percent, 'y_pred_adjusted':0.55},
            4: {"dataset": "lexin_af", "label": "label", "classification_type": "binary", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False, 'percent': percent, 'y_pred_adjusted':0.5},
            5: {"dataset": "dalia", "label": "label", "classification_type": "multi", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False, 'y_pred_adjusted':0.36},
            6: {"dataset": "wesad", "label": "label", "classification_type": "multi", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": True, 'concat': False, 'y_pred_adjusted':0.36},   
            7: {"dataset": "vital", "label": "icu", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent, 'y_pred_adjusted':0.36},
            8: {"dataset": "vital", "label": "pft", "classification_type": "binary", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False, 'percent': percent, 'y_pred_adjusted':0.25},
            }
        
    # # added to binary, they actually has no difference in the entry point, the "binary" type runs all of the tasks.
    # if args.classification_type == "multi":   
    #     config = {
    #         0: {"dataset": "vital", "label": "optype", "classification_type": "multi", "linear_model": "lr", "level": "patient", "content": "_patient", "string_convert": False, 'concat': False},
    #         # 0: {"dataset": "dalia", "label": "label", "classification_type": "multi", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": False, 'concat': False},
    #         # 1: {"dataset": "wesad", "label": "label", "classification_type": "multi", "linear_model": "lr", "level": "segment", "content": "_segment", "string_convert": True, 'concat': False},   
    #     }

    all_results = get_results(args.model, config)
    df = pd.DataFrame(all_results)
    date_tag = datetime.now().strftime("%Y%m%d")
    os.makedirs(f"../results_{date_tag}", exist_ok=True)
    df.to_csv(f"../results_{date_tag}/{args.model}_{args.classification_type}.csv", index=False)