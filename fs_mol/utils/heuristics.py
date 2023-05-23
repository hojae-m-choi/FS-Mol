import gc
import numpy as np
from scipy.spatial import distance_matrix
import inspect

__all__ = [
    'RandomSampling',
    'InvVarSampling',
    'CoreSetSampling',
]

# Create a context to change the seed
class RandomStateContext:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        np.random.set_state(self.state)
        
def get_cls_by_name(name, module):
    classes = {}

    # Get all classes defined in the module
    for obj_name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            classes[obj_name] = obj

    return classes[name]

class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, model, num_labels=10, gpu=1):
        self.model = model
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, samples, amount):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param samples: the pool dataset
        :param amount: the amount of examples to query
        :return: the sampled pool indices
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class RandomSampling(QueryMethod):
    """
    A random sampling query strategy baseline.
    ref: https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py
    """
    
    __name__ = 'random'
    
    def __init__(self, 
                 seed: int = None,
                 model=None, num_labels=None, gpu=None):
        super().__init__(model, num_labels, gpu)
        
        self.seed = seed if seed is not None else np.random.get_state()[1][0]
        self.local_rnsctx = RandomStateContext(self.seed)
        
    def query(self, samples, amount):
        unlabeled_idx = np.arange(len(samples))
        with self.local_rnsctx:
            return np.random.choice(unlabeled_idx, amount, replace=False)


class InvVarSampling(QueryMethod):
    __name__ = 'inv_var_ensemble'
    
    def __init__(self, model,  num_labels, gpu):
        super().__init__(model,  num_labels, gpu)

    def query(self,  samples, amount):
        unlabeled_idx = np.arange(len(samples))
        return np.random.choice(unlabeled_idx, amount, replace=False)


class CoreSetSampling(QueryMethod):
    """
    An implementation of the greedy core set query strategy.
    ref: https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py
    """
    __name__ = 'coreset'
    def __init__(self, model,  num_labels, gpu):
        super().__init__(model,  num_labels, gpu)
        # init batcher for given model
        self.batcher = None

    def greedy_k_center(self, labeled, unlabeled, amount):
        """_summary_

        Args:
            labeled (_type_): labeled ndarray (X)
            unlabeled (_type_): unlabeled ndarray (X)
            amount (_type_): amount of query samples

        Returns:
            _type_: _description_
        """
        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(
            distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), 
                            unlabeled),
                        axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack(
                (min_dist, 
                 np.min(dist, axis=0).reshape((1, min_dist.shape[1])))
                                  )
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        # newly added sample affect to the pairwise distances
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(
                unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), 
                unlabeled)
            min_dist = np.vstack(
                (min_dist, 
                 dist.reshape((1, min_dist.shape[1])))
                 )
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

    def query(self, labeled_samples, unlabeled_samples, amount):

        labeled_idx = np.arange(len(labeled_samples))
        unlabeled_idx = np.arange(len(unlabeled_samples))
        
        X_train = self.batcher ( [sample for sample in labeled_samples] + [sample for sample in unlabeled_samples] )
        # use the learned representation for the k-greedy-center algorithm:
        
        ## TODO: extract representation model from original model
        ## randomForest: Identical (distance is not updated during training)
        ## gnn-mt: GNNMultitaskModel.graph_feature_extractor(batch)
        ## gnn-maml: tensorflow 기반으로 작성되어서 비교적 쉬울 수 있음..??
        ## mat: GraphTransformer.encode() method
        
        representation_model = Model(inputs=self.model.input, outputs=self.model.get_layer('softmax').input)
        
        representation = representation_model(X_train)
        new_indices = self.greedy_k_center(representation[labeled_idx, :], representation[labeled_idx.max() + unlabeled_idx, :], amount)
        return unlabeled_idx[new_indices]  # new_indices is indices on unlabelled representations
