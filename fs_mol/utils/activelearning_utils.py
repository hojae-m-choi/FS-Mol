from fs_mol.data.active_label import ActiveLearningLabel
from fs_mol.utils.metrics import get_uncertain_indices
from fs_mol.models.ensemble import (train_ensemble,
                                    predict_ensemble,
                                    metric_ensemble,
                                    uncertainty_ensemble)
from fs_mol.utils.heuristics import RandomSampling



AL_HEURISTIC_IMPLEMENTATIONS={
    'random': RandomSampling,
    
}


def activelearning_loop(models_dict, X_train, y_train, X_test, y_test, regression_task, logger):
    # Init active-learning label
    ac_label = ActiveLearningLabel(oracle_size=len(X_train))  ## AL
    
    # Start active-learning cycle
    ac_label.disclose_randomly(n = 32)  ## AL
    ac_label.label(pool_indices_list = list(range(32)))  ## AL
    for curr_cycle in range(3):  ## AL
        labelled_indices = ac_label.get_indices_for_active_cycle()  ## AL
        X_labelled = X_train[labelled_indices, :]  ## AL
        y_labelled = [y_train[idx] for idx in labelled_indices]  ## AL
        
        # Train models:
        logger.info(f" AL cycle-{curr_cycle} with {X_labelled.shape[0]} datapoints.")
        trained_models_dict = train_ensemble(models_dict, X_labelled, y_labelled)
        
        # Compute test results:
        test_predictions_dict = predict_ensemble(trained_models_dict, X_test, regression_task)
        test_metrics = metric_ensemble(test_predictions_dict, y_test, regression_task)
        logger.info(f" Test metrics: {test_metrics}")
        
        # Import disclosed samples into pool
        ac_label.disclose_randomly(n = 28)  ## AL
        unlabelled_indices = ac_label.unlabelled_indices  ## AL
        X_pool = X_train[unlabelled_indices, :]  ## AL
        
        # Extract uncertain samples
        logger.info(f" Computing uncertainty with {X_pool.shape[0]} datapoints.")  ## AL
        pool_predictions_dict = predict_ensemble(trained_models_dict, X_pool, regression_task)  ## AL
        uncertainty = uncertainty_ensemble(pool_predictions_dict)  ## AL
        sampled_pool_indices = get_uncertain_indices(uncertainty, k = 10)  ## AL
        
        # Label selected samples
        logger.info(f" Label {len(sampled_pool_indices)} datapoints.")  ## AL
        ac_label.label(sampled_pool_indices)  ## AL
    return test_metrics