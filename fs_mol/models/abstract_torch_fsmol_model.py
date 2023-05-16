from __future__ import annotations
from dataclasses import dataclass

import logging
import os
import sys
import time
from abc import abstractclassmethod, abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Tuple,
    Dict,
    List,
    Optional,
    DefaultDict,
    Callable,
    Iterable,
    Union,
    Type,
    Any,
    Generic,
    TypeVar,
)
from typing_extensions import Literal

import numpy as np
import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import (
    FSMolBatcher,
    FSMolBatchIterable,
    FSMolTaskSample,
)
from fs_mol.utils.logging import PROGRESS_LOG_LEVEL
from fs_mol.utils.metric_logger import MetricLogger
from fs_mol.utils.metrics import (
    avg_task_metrics_list,
    compute_metrics,
    BinaryEvalMetrics,
    RegressionEvalMetrics,
    BinaryMetricType,
    RegressionMetricType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TorchFSMolModelOutput:
    # Predictions for each input molecule, as a [NUM_MOLECULES, 1] float tensor
    molecule_label: torch.Tensor
    label_type: str = 'classification'

    @property
    def is_regression_task(self):
        return bool( self.label_type == 'regression' )

@dataclass
class TorchFSMolModelLoss:
    label_loss: torch.Tensor

    @property
    def total_loss(self) -> torch.Tensor:
        return self.label_loss

    @property
    def metrics_to_log(self) -> Dict[str, Any]:
        return {"total_loss": self.total_loss, "label_loss": self.label_loss}


BatchFeaturesType = TypeVar("BatchFeaturesType")
BatchOutputType = TypeVar("BatchOutputType", bound=TorchFSMolModelOutput)
BatchLossType = TypeVar("BatchLossType", bound=TorchFSMolModelLoss)
MetricType = Union[BinaryMetricType, RegressionMetricType, Literal["loss"]]
ModelStateType = Dict[str, Any]


class AbstractTorchFSMolModel(
    Generic[BatchFeaturesType, BatchOutputType, BatchLossType], torch.nn.Module
):
    def __init__(self, label_type='classification'):
        super().__init__()
        self.label_type = label_type
        if label_type == 'classification':
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif label_type == 'regression':
            self.criterion = torch.nn.MSELoss(reduction="none")
        else:
            raise KeyError

    @abstractmethod
    def forward(self, batch: BatchFeaturesType) -> BatchOutputType:
        """
        Given the features of a batch of molecules, compute proability of each of these molecules
        having an "active" label in the currently learned assay.

        Args:
            batch: representation of the features of NUM_MOLECULES, as chosen by the implementor.

        Returns:
            Model output, a subtype of TorchFSMolModelOutput, ensuring that at least molecule_label
            is present.
        """
        raise NotImplementedError()

    def compute_loss(
        self, batch: BatchFeaturesType, model_output: BatchOutputType, labels: torch.Tensor
    ) -> BatchLossType:
        """
        Compute loss; can be overwritten by implementor to implement extra objectives.

        Args:
            batch: representation of the features of NUM_MOLECULES, as chosen by the implementor.
            labels: float Tensor of shape [NUM_MOLECULES], indicating the target label of the each molecule.
            model_output: output of the model, as chosen by the implementor.

        Returns:
            Dictionary mapping partial loss names to the loss. Optimization will be performed over the sum of values.
        """
        predictions = model_output.molecule_label.squeeze(dim=-1)
        if self.label_type != model_output.label_type:
            raise TypeError
        label_loss = torch.mean(self.criterion(predictions, labels.float()))
        return TorchFSMolModelLoss(label_loss=label_loss)

    @abstractmethod
    def get_model_state(self) -> ModelStateType:
        """
        Return the state of the model such as configuration and learnable parameters.

        Returns:
            Dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def load_model_state(
        self,
        model_state: ModelStateType,
        load_task_specific_weights: bool,
        quiet: bool = False,
    ) -> None:
        """Load model weights from a model state as generated by get_model_state.

        Args:
            model_state: a dictionary representing model state, as returned by model.get_model_state().
            load_task_specific_weights: a flag specifying whether, if applicable, task-specific weights
                should be loaded or not. This would be False for the case of loading the weights when
                transferring the model to a new task.
            quiet: flag indicating if the loading should report additional details (e.g., which weights
                have been loaded / re-initialized).
        """
        raise NotImplementedError()

    @abstractmethod
    def is_param_task_specific(self, param_name: str) -> bool:
        raise NotImplementedError()

    @abstractclassmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType]:
        """Build the model architecture based on a saved checkpoint."""
        raise NotImplementedError()


def linear_warmup(cur_step: int, warmup_steps: int = 0) -> float:
    if cur_step >= warmup_steps:
        return 1.0
    return cur_step / warmup_steps


def create_optimizer(
    model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
    lr: float = 0.001,
    task_specific_lr: float = 0.005,
    warmup_steps: int = 1000,
    task_specific_warmup_steps: int = 100,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    # Split parameters into shared and task-specific ones:
    shared_parameters, task_spec_parameters = [], []
    for param_name, param in model.named_parameters():
        if model.is_param_task_specific(param_name):
            task_spec_parameters.append(param)
        else:
            shared_parameters.append(param)

    opt = torch.optim.Adam(
        [
            {"params": task_spec_parameters, "lr": task_specific_lr},
            {"params": shared_parameters, "lr": lr},
        ],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt,
        lr_lambda=[
            partial(
                linear_warmup, warmup_steps=task_specific_warmup_steps
            ),  # for task specific paramters
            partial(linear_warmup, warmup_steps=warmup_steps),  # for shared paramters
        ],
    )

    return opt, scheduler


def save_model(
    path: str,
    model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
) -> None:
    data = model.get_model_state()

    if optimizer is not None:
        data["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        data["epoch"] = epoch

    torch.save(data, path)


def load_model_weights(
    model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
    path: str,
    load_task_specific_weights: bool,
    quiet: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    checkpoint = torch.load(path, map_location=device)
    model.load_model_state(checkpoint, load_task_specific_weights, quiet)


def resolve_starting_model_file(
    model_file: str,
    model_cls: Type[AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType]],
    out_dir: str,
    use_fresh_param_init: bool,
    config_overrides: Dict[str, Any] = {},
    device: Optional[torch.device] = None,
) -> str:
    # If we start from a fresh init, create a model, do a random init, and store that away somewhere:
    if use_fresh_param_init:
        logger.info("Using fresh model init.")
        model = model_cls.build_from_model_file(
            model_file=model_file, config_overrides=config_overrides, device=device
        )

        resolved_model_file = os.path.join(out_dir, f"fresh_init.pkl")
        save_model(resolved_model_file, model)

        # Hack to give AML some time to actually save.
        time.sleep(1)
    else:
        resolved_model_file = model_file
        logger.info(f"Using model weights loaded from {resolved_model_file}.")

    return resolved_model_file


def run_on_data_iterable(
    model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
    data_iterable: Iterable[Tuple[BatchFeaturesType, torch.Tensor]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    max_num_steps: Optional[int] = None,
    quiet: bool = False,
    metric_name_prefix: str = "",
    aml_run=None,
) -> Tuple[float, Dict[int, Union[BinaryEvalMetrics, RegressionEvalMetrics]]]:
    """Run the given model on the provided data loader.

    Args:
        model: Model to run things on.
        data_iterable: Iterable that provides the data we run on; data has been batched
            by an appropriate batcher.
        optimizer: Optional optimizer. If present, the given model will be trained.
        lr_scheduler: Optional learning rate scheduler around optimizer.
        max_num_steps: Optional number of steps. If not provided, will run until end of data loader.
    """
    if optimizer is None:
        model.eval()
    else:
        model.train()

    per_task_preds: DefaultDict[int, List[float]] = defaultdict(list)
    per_task_labels: DefaultDict[int, List[float]] = defaultdict(list)

    metric_logger = MetricLogger(
        log_fn=lambda msg: logger.log(PROGRESS_LOG_LEVEL, msg),
        aml_run=aml_run,
        quiet=quiet,
        metric_name_prefix=metric_name_prefix,
    )
    for batch_idx, (batch, labels) in enumerate(iter(data_iterable)):
        if max_num_steps is not None and batch_idx >= max_num_steps:
            break

        if optimizer is not None:
            optimizer.zero_grad()

        predictions: BatchOutputType = model(batch)

        model_loss = model.compute_loss(batch, predictions, labels)
        metric_logger.log_metrics(**model_loss.metrics_to_log)

        # Training step:
        if optimizer is not None:
            model_loss.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        # === Finally, collect per-task results to be used for further eval:
        sample_to_task_id: Dict[int, int] = {}
        if hasattr(batch, "sample_to_task_id"):
            sample_to_task_id = batch.sample_to_task_id
        else:
            # If we don't have a sample task information, just use 0 as default task ID:
            sample_to_task_id = defaultdict(lambda: torch.tensor(0))

        # Apply sigmoid to have predictions in appropriate range for computing (scikit) scores.
        num_samples = labels.shape[0]
        if predictions.is_regression_task:
            predicted_labels = predictions.molecule_label.detach().cpu() #TODO: if in regression task, prediction should be inverse transformed.
        else:
            predicted_labels = torch.sigmoid(predictions.molecule_label).detach().cpu()  
        for i in range(num_samples):
            task_id = sample_to_task_id[i].item()
            per_task_preds[task_id].append(predicted_labels[i].item())
            per_task_labels[task_id].append(labels[i].item())

    metrics = compute_metrics(per_task_preds, per_task_labels, label_type=predictions.label_type)

    return metric_logger.get_mean_metric_value("total_loss"), metrics


def validate_on_data_iterable(
    model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
    data_iterable: Iterable[Tuple[BatchFeaturesType, torch.Tensor]],
    metric_to_use: MetricType = "avg_precision",
    quiet: bool = False,
) -> float:
    valid_loss, valid_metrics = run_on_data_iterable(
        model,
        data_iterable=data_iterable,
        quiet=quiet,
    )
    if not quiet:
        logger.info(f"  Validation loss: {valid_loss:.5f}")
    if metric_to_use == "loss":
        return -valid_loss  # We are maximising things elsewhere, so flip the sign on the loss
    else:
        # If our data_iterable had more than one task, we'll have one result per task - average them:
        mean_valid_metrics = avg_task_metrics_list(list(valid_metrics.values()))
        return mean_valid_metrics[metric_to_use][0]


def train_loop(
    model: AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_data: Iterable[Tuple[BatchFeaturesType, torch.Tensor]],
    valid_fn: Callable[
        [AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType]], float
    ],
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 100,
    patience: int = 5,
    aml_run=None,
    quiet: bool = False,
) -> Tuple[float, ModelStateType]:
    if quiet:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    initial_valid_metric = float("-inf")
    best_valid_metric = initial_valid_metric
    logger.log(log_level, f"  Initial validation metric: {best_valid_metric:.5f}")

    best_model_state = model.get_model_state()
    epochs_since_best = 0
    for epoch in range(0, max_num_epochs):
        logger.log(log_level, f"== Epoch {epoch}")
        logger.log(log_level, f"  = Training")
        train_loss, train_metrics = run_on_data_iterable(
            model,
            data_iterable=train_data,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            quiet=quiet,
            metric_name_prefix="train_",
            aml_run=aml_run,
        )
        mean_train_metric = np.mean(
            [getattr(task_metrics, metric_to_use) for task_metrics in train_metrics.values()]
        )
        logger.log(log_level, f"  Mean train loss: {train_loss:.5f}")
        logger.log(log_level, f"  Mean train {metric_to_use}: {mean_train_metric:.5f}")
        logger.log(log_level, f"  = Validation")
        valid_metric = valid_fn(model)
        logger.log(log_level, f"  Validation metric: {valid_metric:.5f}")
        
        # early stopping
        # valid_metric should be the higher, the better performance
        if valid_metric > best_valid_metric:
            logger.log(
                log_level,
                f"   New best validation result {valid_metric:.5f} (increased from {best_valid_metric:.5f}).",
            )
            best_valid_metric = valid_metric
            epochs_since_best = 0

            best_model_state = model.get_model_state()
        else:
            epochs_since_best += 1
            logger.log(log_level, f"   Now had {epochs_since_best} epochs since best result.")
            if epochs_since_best >= patience:
                break

    return best_valid_metric, best_model_state


def eval_model_by_finetuning_on_task(
    model_weights_file: str,
    model_cls: Type[AbstractTorchFSMolModel[BatchFeaturesType, BatchOutputType, BatchLossType]],
    task_sample: FSMolTaskSample,
    batcher: FSMolBatcher[BatchFeaturesType, torch.Tensor],
    learning_rate: float,
    task_specific_learning_rate: float,
    metric_to_use: MetricType = "avg_precision",
    max_num_epochs: int = 50,
    patience: int = 10,
    seed: int = 0,
    quiet: bool = False,
    device: Optional[torch.device] = None,
    config_overrides: Dict[str, Any] = {},
) -> Union[BinaryEvalMetrics,RegressionEvalMetrics]:
    # Build the model afresh and load the shared weights.
    config_overrides.update({"num_tasks": 1})  # this forcing num_taks as 1
    model: AbstractTorchFSMolModel[
        BatchFeaturesType, BatchOutputType, BatchLossType
    ] = model_cls.build_from_model_file(
        model_weights_file, quiet=quiet, device=device, config_overrides=config_overrides
    )
    load_model_weights(model, model_weights_file, load_task_specific_weights=False)

    (optimizer, lr_scheduler) = create_optimizer(
        model,
        lr=learning_rate,
        task_specific_lr=task_specific_learning_rate,
        warmup_steps=2,
        task_specific_warmup_steps=2,
    )

    best_valid_metric, best_model_state = train_loop(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_data=FSMolBatchIterable(task_sample.train_samples, batcher, shuffle=True, seed=seed),
        valid_fn=partial(
            validate_on_data_iterable,
            data_iterable=FSMolBatchIterable(task_sample.valid_samples, batcher),
            metric_to_use="loss",
            quiet=quiet,
        ),
        metric_to_use=metric_to_use,
        max_num_epochs=max_num_epochs,
        patience=patience,
        quiet=True,
    )

    logger.log(PROGRESS_LOG_LEVEL, f" Final validation loss:       {float(best_valid_metric):.5f}")
    # Load best model state and eval on test data:
    model.load_model_state(best_model_state, load_task_specific_weights=True)
    test_loss, _test_metrics = run_on_data_iterable(
        model, data_iterable=FSMolBatchIterable(task_sample.test_samples, batcher), quiet=quiet
    )
    test_metrics = next(iter(_test_metrics.values()))
    logger.log(PROGRESS_LOG_LEVEL, f" Test loss:                   {float(test_loss):.5f}")
    logger.info(f" Test metrics: {test_metrics}")
    logger.info(
        f"Dataset sample has {task_sample.test_pos_label_ratio:.4f} positive label ratio in test data.",
    )
    logger.log(
        PROGRESS_LOG_LEVEL,
        f"Dataset sample test {metric_to_use}: {getattr(test_metrics, metric_to_use):.4f}",
    )

    return test_metrics
