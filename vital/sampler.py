import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import math
from itertools import cycle
import typing as tp

class DeterministicImbalancedSampler(Sampler[int]): # Inherit from Sampler[int] as it yields individual indices
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int, # Need batch_size here to know how many samples per conceptual batch
        minority_class_label: int,
        minority_samples_per_batch: int,
        label_key: str = "y",
        # drop_last is handled by DataLoader now, but we need it for num_batches_per_epoch calculation
        drop_last: bool = False,
        generator: tp.Optional[torch.Generator] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.minority_class_label = minority_class_label
        self.minority_samples_per_batch = minority_samples_per_batch
        self.label_key = label_key
        self.drop_last_batch_sampler_logic = drop_last # Store for internal calculation
        self.generator = generator if generator is not None else np.random.default_rng()

        if minority_samples_per_batch >= batch_size:
            raise ValueError(f"`minority_samples_per_batch` ({minority_samples_per_batch}) "
                             f"must be less than `batch_size` ({batch_size}).")
        if minority_samples_per_batch < 0:
            raise ValueError("`minority_samples_per_batch` cannot be negative.")
        if minority_samples_per_batch == 0:
            print("Warning: `minority_samples_per_batch` is 0. Batches will not guarantee minority presence.")

        self.majority_samples_per_batch = self.batch_size - self.minority_samples_per_batch

        self.majority_indices, self.minority_indices = self._get_stratified_indices(
            dataset, minority_class_label, label_key
        )
            
        print(f"Found {len(self.majority_indices)} majority samples and {len(self.minority_indices)} minority samples (label {self.minority_class_label}).")

        if not self.minority_indices and self.minority_samples_per_batch > 0:
            raise ValueError(f"No samples found for minority class label {minority_class_label}, but `minority_samples_per_batch` > 0.")
        if not self.majority_indices and self.majority_samples_per_batch > 0:
            raise ValueError(f"No samples found for majority classes, but `majority_samples_per_batch` > 0.")
             
        # Define epoch length based on majority class.
        if self.majority_samples_per_batch > 0:
            total_possible_batches = len(self.majority_indices) / self.majority_samples_per_batch
        elif self.minority_samples_per_batch > 0:
             total_possible_batches = len(self.minority_indices) / self.minority_samples_per_batch
        else:
            total_possible_batches = 0

        # This `num_batches_per_epoch` governs how many *conceptual* batches we compose
        if self.drop_last_batch_sampler_logic: # Use this logic for internal num batches
            self.num_conceptual_batches_per_epoch = math.floor(total_possible_batches)
        else:
            self.num_conceptual_batches_per_epoch = math.ceil(total_possible_batches)

        if self.num_conceptual_batches_per_epoch == 0 and (len(self.majority_indices) > 0 or len(self.minority_indices) > 0):
             raise ValueError("Calculated 0 conceptual batches per epoch, but dataset contains samples. Check `batch_size` and samples per batch settings.")


    @staticmethod
    def _get_stratified_indices(
        dataset: Dataset, 
        minority_class_label: int, 
        label_key: str
    ) -> tp.Tuple[tp.List[int], tp.List[int]]:
        """
        Efficiently collects indices for majority and minority classes by directly
        accessing `dataset.data` or `dataset._data`. This avoids calling `__getitem__`
        which can be slow due to data loading/transforms.
        """
        majority_indices = []
        minority_indices = []

        raw_data_list = None
        if hasattr(dataset, 'data') and isinstance(dataset.data, list):
            raw_data_list = dataset.data
            print(f"Collecting labels from `dataset.data` (raw list) using label_key='{label_key}'.")
        elif hasattr(dataset, '_data') and isinstance(dataset._data, list): # For some MONAI internal structures
            raw_data_list = dataset._data
            print(f"Collecting labels from `dataset._data` (raw list) using label_key='{label_key}'.")
        else:
            raise AttributeError(
                "Dataset must have a `.data` or `._data` attribute that is a list of dictionaries "
                "to allow efficient label collection for DeterministicImbalancedSampler."
            )
        
        for i, item_dict in enumerate(raw_data_list):
            if not isinstance(item_dict, dict) or label_key not in item_dict:
                raise TypeError(f"Item {i} in raw_data_list is not a dictionary or does not contain key '{label_key}'. "
                                f"Ensure your dataset's raw data (e.g., `dataset.data`) is a list of dictionaries with the label.")
            label = item_dict[label_key]
            if isinstance(label, torch.Tensor):
                label = label.item() # Convert tensor label to Python int
            
            if label == minority_class_label:
                minority_indices.append(i)
            else:
                majority_indices.append(i)
        return majority_indices, minority_indices


    def __iter__(self) -> tp.Iterator[int]:
        # Create a torch.Generator for PyTorch operations, seeded from the numpy generator
        # Fix: Explicitly convert seed_val to a Python int
        seed_val = int(self.generator.integers(0, 2**32 - 1)) 
        torch_generator = torch.Generator(device="cpu").manual_seed(seed_val)

        # Shuffle main index lists using numpy's shuffle (operates in-place on lists)
        shuffled_majority_indices = list(self.majority_indices) 
        self.generator.shuffle(shuffled_majority_indices)

        shuffled_minority_indices = list(self.minority_indices)
        self.generator.shuffle(shuffled_minority_indices)
        
        majority_iter = iter(shuffled_majority_indices)
        minority_cycle_iter = cycle(shuffled_minority_indices)

        all_epoch_indices: tp.List[int] = []

        # Step 1: Compose all conceptual batches for the epoch
        for _ in range(self.num_conceptual_batches_per_epoch):
            current_conceptual_batch_indices = []

            # Add minority samples
            for _ in range(self.minority_samples_per_batch):
                current_conceptual_batch_indices.append(next(minority_cycle_iter))

            # Add majority samples
            for _ in range(self.majority_samples_per_batch):
                try:
                    current_conceptual_batch_indices.append(next(majority_iter))
                except StopIteration:
                    break
            
            # This is the "in-batch" shuffle, applied to each conceptual batch
            if current_conceptual_batch_indices:
                # Convert to tensor, shuffle using torch.randperm, then convert back
                batch_tensor = torch.tensor(current_conceptual_batch_indices, dtype=torch.long)
                shuffled_batch_tensor = batch_tensor[torch.randperm(len(batch_tensor), generator=torch_generator)]
                all_epoch_indices.extend(shuffled_batch_tensor.tolist()) # Add to the master list
            else:
                # If a conceptual batch is empty, stop composing.
                break

        # Step 2: Yield individual indices from the pre-composed, flattened list
        yield from all_epoch_indices

    def __len__(self) -> int:
        # This length is the total number of individual indices yielded over an epoch,
        # which is the sum of all samples in all conceptual batches.
        # This is what DataLoader uses to determine total steps.
        return self.num_conceptual_batches_per_epoch * self.batch_size