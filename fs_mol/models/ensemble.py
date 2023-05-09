import numpy as np
from sklearn.model_selection import GridSearchCV
from fs_mol.utils.metrics import (compute_binary_task_metrics, 
                                  compute_regression_task_metrics)



def train_ensemble(models_dict, X_train, y_train):
        trained_model_dict = {}
        for name, model in models_dict.items():
            model.fit(X_train, y_train)
            if isinstance(models_dict[name], GridSearchCV):
                trained_model_dict[name] = model.best_estimator_
            else:
                trained_model_dict[name] = model
        
        return trained_model_dict
        
def predict_ensemble(models_dict, X_test, regression_task):
    predictions_dict = {}
    
    for name, model in models_dict.items():
        if regression_task:
            y_predicted = model.predict(X_test)
            # TODO: inverse scaling y_predicted
            predictions_dict[name] = y_predicted
        else:
            y_predicted_true_probs = model.predict_proba(X_test)[:, 1]
            predictions_dict[name] = y_predicted_true_probs
    
    return predictions_dict
    
def metric_ensemble(predictions_dict, y_test, regression_task, reduce = np.mean, **reduce_kwargs):
    y_predicted = np.stack([y_predicted for y_predicted in predictions_dict.values()], axis=-1)
    y_predicted = reduce(y_predicted, axis=-1, **reduce_kwargs) 
    if regression_task:
        # TODO: inverse scaling y_predicted
        test_metrics = compute_regression_task_metrics(y_predicted, y_test)
    else:
        test_metrics = compute_binary_task_metrics(y_predicted, y_test)
    return test_metrics

def uncertainty_ensemble(predictions_dict, reduce = np.std, **reduce_kwargs):
    y_predicted = np.stack([y_predicted for y_predicted in predictions_dict.values()], axis=-1)
    uncertainty_ensemble = 1/reduce(y_predicted, axis=-1, **reduce_kwargs)
    return uncertainty_ensemble
