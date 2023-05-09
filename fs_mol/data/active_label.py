from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from sklearn.utils import check_random_state

class ActiveLearningLabel:
    def __init__(
            self,
            oracle_size: int,
            random_state: Union[int, Any] = None,
            last_active_cycles: int = -1,
            train_val_split: float = 0.2,
    ) -> None:
        """_summary_

        Args:
            oracle_size (int): _description_
            random_state (Union[int, Any], optional): _description_. Defaults to None.
            last_active_cycles (int, optional): _description_. Defaults to -1.
            train_val_split (float, optional): _description_. Defaults to 0.2.

        Raises:
            ValueError: _description_
        """
        self.labelled_map = np.zeros(oracle_size)
        self.disclosed_map = np.zeros(oracle_size)
        self.train_labelled_map = np.zeros(oracle_size)
        self.valid_labelled_map = np.zeros(oracle_size)
        self.train_val_split = train_val_split

        self.random_state = check_random_state(random_state)
        
        if last_active_cycles == 0 or last_active_cycles < -1:
            raise ValueError("last_active_cycles must be > 0 or -1 when disabled.")
        self.last_active_cycles = last_active_cycles

    def get_indices_for_active_cycle(self, min_cycle: int = None) -> List[int]:
        """Returns the indices required for the active cycle.
        Returns the indices of the labelled items. Also takes into account self.last_active_cycle.
        Returns:
            List of the selected indices for training.
        """
        if min_cycle is not None:
            min_labelled_cycle = min_cycle
        elif self.last_active_cycles == -1:  ## ??
            min_labelled_cycle = 0
        else:
            min_labelled_cycle = max(0, self.current_al_cycle - self.last_active_cycles)

        # we need to work with lists since arrow dataset is not compatible with np.int types!
        oracle_indices = [indx for indx, val in enumerate(self.labelled_map) if val > min_labelled_cycle]
        return oracle_indices

    def get_train_val_indices_for_active_cycle(
            self, 
            min_cycle: int = None, 
            train_val: str = "train") -> List[int]:
        if min_cycle is not None:
            min_labelled_cycle = min_cycle
        elif self.last_active_cycles == -1:
            min_labelled_cycle = 0
        else:
            min_labelled_cycle = max(0, self.current_al_cycle - self.last_active_cycles)

        # we need to work with lists since arrow dataset is not compatible with np.int types!
        if train_val == "train":
            oracle_indices = [indx for indx, val in enumerate(self.train_labelled_map) if val > min_labelled_cycle]
        elif train_val == "valid":
            oracle_indices = [indx for indx, val in enumerate(self.valid_labelled_map) if val > min_labelled_cycle]
        else:
            raise ValueError("train_val argument must be either train or valid")
        return oracle_indices
    
    @property
    def closed_indices(self) -> List[int]:
        oracle_indices = [indx for indx, bool_val in enumerate(~self.disclosed) if bool_val]
        return oracle_indices
    
    @property
    def disclosed_indices(self) -> List[int]:
        oracle_indices = [indx for indx, bool_val in enumerate(self.disclosed) if bool_val]
        return oracle_indices
    
    @property
    def unlabelled_indices(self) -> List[int]:
        oracle_indices = [idx for idx, bool_val in enumerate(self.unlabelled) if bool_val]
        return oracle_indices

    def is_labelled(self, idx: int) -> bool:
        """Check if a datapoint is labelled."""
        return bool(self.labelled[idx].item() == 1)
    
    def disclose(self, index_list:List[int] = None) -> None:
        active_step = self.current_al_cycle + 1
        for idx in index_list:
            self.disclosed_map[idx] = active_step
    
    def disclose_randomly(self, n:int) -> None:
        active_step = self.current_al_cycle + 1
        oracle_index_list = np.random.choice(
            self.closed_indices,
            size=min(n, self.n_closed), replace=False).tolist()

        for idx in oracle_index_list:
            self.disclosed_map[idx] = active_step
            
    def label(self, pool_indices_list) -> None:
        active_step = self.current_al_cycle + 1
        oracle_indices_list = self._pool_to_oracle_index(pool_indices_list)
        
        disclosed_oracle_indices = self.disclosed_indices
        for idx in oracle_indices_list:
            assert idx in disclosed_oracle_indices
            assert idx in self.unlabelled_indices
            self.labelled_map[idx] = active_step
            if np.random.rand() > self.train_val_split:
                # traininig
                self.train_labelled_map[idx] = active_step
            else:
                self.valid_labelled_map[idx] = active_step

    @property
    def current_al_cycle(self) -> int:
        """Get the current active learning cycle."""
        return int(self.labelled_map.max())

    @property
    def labelled(self) -> np.array:
        """An array that acts as a boolean mask which is True for every
        data point that is labelled, and False for every data point that is not
        labelled."""
        return self.labelled_map.astype(bool)
    
    @property
    def disclosed(self) -> np.array:
        """"""
        return self.disclosed_map.astype(bool)
    
    @property
    def unlabelled(self) -> np.array:
        """"""
        return (~self.labelled * self.disclosed)
    
    @property
    def n_labelled(self):
        """The number of labelled data points."""
        return self.labelled.sum()
    
    @property
    def n_unlabelled(self):
        """The number of unlabelled data points."""
        return self.unlabelled.sum()
    
    @property
    def n_closed(self):
        """The number of unlabelled data points."""
        return (~self.disclosed).sum()
    
    @property
    def n_disclosed(self):
        """The number of unlabelled data points."""
        return (self.disclosed).sum()
    
    def __len__(self) -> int:
        """Return how many actual data / label pairs we have."""
        return len(self.get_indices_for_active_cycle())
    
    def labelled_to_oracle_index(self, index: int) -> int:
        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]
            
        return [int(self.labelled.nonzero()[0][idx].squeeze().item()) for idx in index]

    def _pool_to_oracle_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, np.int64) or isinstance(index, int):
            index = [index]

        unlabelled_nonezero = self.unlabelled.nonzero()[0]
        return [int(unlabelled_nonezero[idx].squeeze().item()) for idx in index]

    def _oracle_to_pool_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, int):
            index = [index]

        # Pool indices are the unlabelled, starts at 0
        unlabelled_cumsum = np.cumsum(self.unlabelled) - 1
        return [int(unlabelled_cumsum[idx].squeeze().item()) for idx in index]

    def oracle_to_labelled_index(self, index: Union[int, List[int]]) -> List[int]:
        if isinstance(index, int):
            index = [index]
        
        return [self.get_indices_for_active_cycle().index(idx) for idx in index]