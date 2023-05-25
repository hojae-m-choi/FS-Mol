#!/usr/bin/env python3
import json
import logging
import sys
from typing import Dict, Optional, List, Any, Union
from functools import partial

import numpy as np
import sklearn.ensemble
import sklearn.neighbors
import xgboost as xgb
from dpu_utils.utils import run_and_debug
from pyprojroot import here as project_root
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_task import FSMolTaskSample
from fs_mol.data.active_label import ActiveLearningLabel
from fs_mol.models.ensemble import (train_ensemble,
                                    predict_ensemble,
                                    metric_ensemble,
                                    uncertainty_ensemble)
from fs_mol.utils.logging import prefix_log_msgs
from fs_mol.utils.cli_utils import str2bool
from fs_mol.utils.metrics import (
                                  compute_binary_task_metrics, 
                                  compute_regression_task_metrics,
                                  EvalMetrics,
                                  get_uncertain_indices)
from fs_mol.utils.test_utils import (
                                    eval_model,
                                    add_eval_cli_args,
                                    set_up_test_run,)
from fs_mol.utils.heuristics import (
                                    RandomSampling)
logger = logging.getLogger(__name__)

# TODO: extend to whichever models seem useful.
# hyperparam search params
DEFAULT_GRID_SEARCH: Dict[str, Dict[str, List[Any]]] = {
    "randomForest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 20],
        "max_features": [None, "sqrt", "log2"],
        "min_samples_leaf": [2, 5],
    },
    "kNN": {"n_neighbors": [4, 8, 16, 32, 64, 128], "metric": ["minkowski"]},
}

NAME_TO_MODEL_CLS: Dict[str, Any] = {
    "XGboost": {
        'classification': xgb.XGBClassifier,
        'regression': xgb.XGBRegressor,
    },
    "randomForest": {
        'classification': sklearn.ensemble.RandomForestClassifier,
        'regression': sklearn.ensemble.RandomForestRegressor,
    },
    
    "kNN": {
        'classification': sklearn.neighbors.KNeighborsClassifier,
        'regression': sklearn.neighbors.KNeighborsRegressor,
    },
}
DEFAULT_PARALLEL_NJOBS = -1

def test_with_active_learning(
    model_name: str,
    task_sample: FSMolTaskSample,
    use_grid_search: bool = True,
    grid_search_parameters: Optional[Dict[str, Any]] = None,
    model_params: Dict[str, Any] = {},
    regression_task: bool = False,
    query_sizes: List[int] = None,
    disclosing_sizes: List[int] = None,
    heuristics: List[str] = [],
) -> Dict[int, EvalMetrics]:
    train_data = task_sample.train_samples
    test_data = task_sample.test_samples
    
    # get data in to form for sklearn
    X_train = np.array([x.get_fingerprint() for x in train_data])
    X_test = np.array([x.get_fingerprint() for x in test_data])
    logger.info(f" Training {model_name} with {X_train.shape[0]} datapoints.")
    
    if regression_task:
        y_train = [float(x.numeric_label) for x in train_data]  # regression label
        # TODO: scaling y_train
        y_test = [float(x.numeric_label) for x in test_data]  # regression label  
        task_type = 'regression'
    else:
        y_train = [float(x.bool_label) for x in train_data]  # binary label
        y_test = [float(x.bool_label) for x in test_data]  # binary label
        task_type = 'classification'
        
    # use the train data to train a baseline model with CV grid search
    # reinstantiate model for each seed.
    # TODO: split train_data with 3 folds
    # TODO: ensemble of models (best model with each data folds)
    
    models_dict: Dict[str, object] = {}
    data_dict = {}
    model_param_dict = {}
    for i in range(5):
        if use_grid_search:
            if grid_search_parameters is None:
                grid_search_parameters = DEFAULT_GRID_SEARCH[model_name]
                # in the case of kNNs the grid search has to be modified -- one cannot have
                # more nearest neighbours than datapoints.
                if model_name == "kNN":
                    permitted_n_neighbors = [
                        x for x in grid_search_parameters["n_neighbors"] if x < int(len(train_data) / 2)
                    ]
                    grid_search_parameters.update({"n_neighbors": permitted_n_neighbors})
            grid_search = GridSearchCV(NAME_TO_MODEL_CLS[model_name][task_type](), grid_search_parameters,
                                    n_jobs=DEFAULT_PARALLEL_NJOBS)
            models_dict[f'{model_name}_{i}'] = grid_search
                
        else:
            model = NAME_TO_MODEL_CLS[model_name][task_type](n_jobs=DEFAULT_PARALLEL_NJOBS)
            params = model.get_params()
            params.update(model_params)
            model.set_params(**params) # maybe errota. no **params in parameters    
            models_dict[f'{model_name}_{i}'] = model
    
    # Init active-learning label
    ac_label = ActiveLearningLabel(oracle_size=len(X_train))  ## AL
    
    cycle_metrics_dict = {key:[] for key in range(1, len(query_sizes))}  ## AL
    # Start active-learning cycle
    ac_label.disclose_randomly(n = disclosing_sizes[0])  ## AL
    ac_label.label(pool_indices_list = list(range(query_sizes[0])))  ## AL
    for curr_cycle in range(1, len(query_sizes)):  ## AL
        with prefix_log_msgs( f"- Cycle {curr_cycle}" ):
            labelled_indices = ac_label.get_indices_for_active_cycle()  ## AL
            X_labelled = X_train[labelled_indices, :]  ## AL
            y_labelled = [y_train[idx] for idx in labelled_indices]  ## AL
            
            # Train models:
            logger.info(f" Training with {X_labelled.shape[0]} datapoints.")
            trained_models_dict = train_ensemble(models_dict, X_labelled, y_labelled)
            
            # Compute test results:
            test_predictions_dict = predict_ensemble(trained_models_dict, X_test, regression_task)
            test_metrics = metric_ensemble(test_predictions_dict, y_test, regression_task)
            logger.info(f" Test metrics: {test_metrics}")
            cycle_metrics_dict[curr_cycle] = test_metrics
            
            # Import disclosed samples into pool
            ac_label.disclose_randomly(n = disclosing_sizes[curr_cycle])  ## AL
            unlabelled_indices = ac_label.unlabelled_indices  ## AL
            unlabelled_indices = ac_label.get_unlabelled_indices_for_active_cycle(curr_cycle)  ## AL
            X_pool = X_train[unlabelled_indices, :]  ## AL
            
            # Extract uncertain samples
            logger.info(f" Computing uncertainty with {X_pool.shape[0]} datapoints.")  ## AL
            
            if "random" in heuristics:
                querymethod = RandomSampling()
                sampled_pool_indices = querymethod.query(samples=X_pool, 
                                                         amount=query_sizes[curr_cycle])  ## AL
            elif "var_ensemble" in heuristics:
                pool_predictions_dict = predict_ensemble(trained_models_dict, X_pool, regression_task)  ## AL
                uncertainty = uncertainty_ensemble(pool_predictions_dict)  ## AL
                sampled_pool_indices = get_uncertain_indices(uncertainty, k = query_sizes[curr_cycle])  ## AL
            else:
                raise NotImplementedError()
            
            # Label selected samples
            logger.info(f" Label {len(sampled_pool_indices)} datapoints.")  ## AL
            ac_label.label(sampled_pool_indices)  ## AL
        
    logger.info(
        f" Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data."
    )

    return cycle_metrics_dict


def test(
    model_name: str,
    task_sample: FSMolTaskSample,
    use_grid_search: bool = True,
    grid_search_parameters: Optional[Dict[str, Any]] = None,
    model_params: Dict[str, Any] = {},
    regression_task: bool = False
) -> EvalMetrics:
    train_data = task_sample.train_samples
    test_data = task_sample.test_samples

    # get data in to form for sklearn
    X_train = np.array([x.get_fingerprint() for x in train_data])
    X_test = np.array([x.get_fingerprint() for x in test_data])
    logger.info(f" Training {model_name} with {X_train.shape[0]} datapoints.")
    if regression_task:
        y_train = [float(x.numeric_label) for x in train_data]  # regression label
        # TODO: scaling y_train
        y_test = [float(x.numeric_label) for x in test_data]  # regression label  
        task_type = 'regression'
    else:
        y_train = [float(x.bool_label) for x in train_data]  # binary label
        y_test = [float(x.bool_label) for x in test_data]  # binary label
        task_type = 'classification'

    # use the train data to train a baseline model with CV grid search
    # reinstantiate model for each seed.
    if use_grid_search:
        if grid_search_parameters is None:
            grid_search_parameters = DEFAULT_GRID_SEARCH[model_name]
            # in the case of kNNs the grid search has to be modified -- one cannot have
            # more nearest neighbours than datapoints.
            if model_name == "kNN":
                permitted_n_neighbors = [
                    x for x in grid_search_parameters["n_neighbors"] if x < int(len(train_data) / 2)
                ]
                grid_search_parameters.update({"n_neighbors": permitted_n_neighbors})
            grid_search = GridSearchCV(NAME_TO_MODEL_CLS[model_name][task_type](), grid_search_parameters,
                                       n_jobs=DEFAULT_PARALLEL_NJOBS)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        model = NAME_TO_MODEL_CLS[model_name][task_type](n_jobs=DEFAULT_PARALLEL_NJOBS)
        params = model.get_params()
        params.update(model_params)
        model.set_params(**params) # maybe errota. no **params in parameters
        model.fit(X_train, y_train)

    # Compute test results:
    if regression_task:
        y_predicted = model.predict(X_test)
        # TODO: inverse scaling y_predicted
        test_metrics = compute_regression_task_metrics(y_predicted, y_test)
    else:
        y_predicted_true_probs = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_binary_task_metrics(y_predicted_true_probs, y_test)

    logger.info(f" Test metrics: {test_metrics}")
    logger.info(
        f" Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data."
    )

    return test_metrics


def run_from_args(args) -> None:
    out_dir, dataset = set_up_test_run(args.model, args)
    if args.active_learning:
        test_fn = partial(test_with_active_learning,
                          query_sizes = args.query_sizes,
                          disclosing_sizes = args.disclosing_sizes
        )
    else:
        test_fn = test
        
    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> EvalMetrics:
        return test_fn(
            model_name=args.model,
            task_sample=task_sample,
            use_grid_search=args.grid_search,
            model_params=args.model_params,
            regression_task=args.regression_task,
            heuristics=args.heuristics,
        )

    eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=args.train_sizes,
        out_dir=out_dir,
        num_samples=args.num_runs,
        seed=args.seed,
        al = args.active_learning,
    )


def run():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test sklearn models on tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--model",
        type=str,
        default="randomForest",
        choices=["randomForest", "kNN"],
        help="The model to use.",
    )
    parser.add_argument(
        "--grid-search",
        type=str2bool,
        default=True,
        help="Perform grid search over hyperparameter space rather than use defaults/passed parameters.",
    )
    parser.add_argument(
        "--model-params",
        type=lambda s: json.loads(s),
        default={},
        help=(
            "JSON dictionary containing model hyperparameters, if not using grid search these will"
            " be used."
        ),
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")
    parser.add_argument("--regression-task", dest="regression_task", action="store_true", help="Enable train/test for regression task")
    args = parser.parse_args()

    run_and_debug(lambda: run_from_args(args), args.debug)


if __name__ == "__main__":
    run()
