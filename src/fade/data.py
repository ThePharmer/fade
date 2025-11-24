"""
Synthetic Data Generation for FADE POC.

Creates a key-value memorization task:
- Model must learn to associate keys with values
- We control when information was "seen" to test memory decay
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class KeyValuePair:
    """A single key-value pair for memorization."""
    key: List[int]
    value: List[int]
    creation_time: int = 0
    access_count: int = 0
    last_access_time: int = 0


class KeyValueMemorizationDataset(Dataset):
    """
    Dataset for key-value memorization task.

    Format: [KEY_START] key tokens [KEY_END] [VAL_START] value tokens [VAL_END]

    The model must learn to predict value tokens given key tokens.
    """

    # Special tokens
    PAD = 0
    KEY_START = 1
    KEY_END = 2
    VAL_START = 3
    VAL_END = 4
    QUERY = 5  # Used during evaluation: predict value from key
    VOCAB_START = 10  # Regular tokens start here

    def __init__(
        self,
        num_pairs: int = 100,
        key_length: int = 4,
        value_length: int = 4,
        vocab_size: int = 256,
        seed: int = 42,
    ):
        """
        Initialize dataset.

        Args:
            num_pairs: Number of key-value pairs
            key_length: Length of each key
            value_length: Length of each value
            vocab_size: Total vocabulary size
            seed: Random seed
        """
        self.num_pairs = num_pairs
        self.key_length = key_length
        self.value_length = value_length
        self.vocab_size = vocab_size

        random.seed(seed)
        torch.manual_seed(seed)

        # Generate key-value pairs
        self.pairs = self._generate_pairs()

        # Track access history for decay simulation
        self.current_time = 0

    def _generate_pairs(self) -> List[KeyValuePair]:
        """Generate unique key-value pairs."""
        pairs = []
        used_keys = set()

        available_tokens = list(range(self.VOCAB_START, self.vocab_size))

        for i in range(self.num_pairs):
            # Generate unique key
            while True:
                key = random.choices(available_tokens, k=self.key_length)
                key_tuple = tuple(key)
                if key_tuple not in used_keys:
                    used_keys.add(key_tuple)
                    break

            # Generate value
            value = random.choices(available_tokens, k=self.value_length)

            pairs.append(KeyValuePair(
                key=key,
                value=value,
                creation_time=i,  # Stagger creation times
            ))

        return pairs

    def get_pair(self, idx: int) -> KeyValuePair:
        """Get a specific pair and update access tracking."""
        pair = self.pairs[idx]
        pair.access_count += 1
        pair.last_access_time = self.current_time
        return pair

    def advance_time(self, steps: int = 1):
        """Advance simulation time."""
        self.current_time += steps

    def create_memorization_sequence(self, pair: KeyValuePair) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a sequence for memorization (training).

        Format: [KEY_START] key [KEY_END] [VAL_START] value [VAL_END]
        Target: shifted version for next-token prediction
        """
        sequence = (
            [self.KEY_START] + pair.key + [self.KEY_END] +
            [self.VAL_START] + pair.value + [self.VAL_END]
        )
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        return input_seq, target_seq

    def get_time_since_creation(self, idx: int) -> int:
        """Get time since pair was created."""
        return self.current_time - self.pairs[idx].creation_time

    def get_time_since_access(self, idx: int) -> int:
        """Get time since pair was last accessed."""
        pair = self.pairs[idx]
        if pair.access_count == 0:
            return self.current_time - pair.creation_time
        return self.current_time - pair.last_access_time

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.get_pair(idx)
        input_seq, target_seq = self.create_memorization_sequence(pair)

        return {
            "input_ids": input_seq,
            "targets": target_seq,
            "pair_idx": idx,
            "time_since_creation": self.get_time_since_creation(idx),
            "time_since_access": self.get_time_since_access(idx),
            "access_count": pair.access_count,
        }


class MemorizationCollator:
    """Collate function for batching memorization sequences."""

    def __init__(self, pad_token: int = 0, max_len: int = 64):
        self.pad_token = pad_token
        self.max_len = max_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = min(max(len(item["input_ids"]) for item in batch), self.max_len)

        input_ids = []
        targets = []
        masks = []
        metadata = {
            "pair_idx": [],
            "time_since_creation": [],
            "time_since_access": [],
            "access_count": [],
        }

        for item in batch:
            seq_len = len(item["input_ids"])
            pad_len = max_len - seq_len

            # Pad sequences
            input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad_len,), self.pad_token)])
            )
            targets.append(
                torch.cat([item["targets"], torch.full((pad_len,), self.pad_token)])
            )
            masks.append(
                torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
            )

            # Collect metadata
            metadata["pair_idx"].append(item["pair_idx"])
            metadata["time_since_creation"].append(item["time_since_creation"])
            metadata["time_since_access"].append(item["time_since_access"])
            metadata["access_count"].append(item["access_count"])

        return {
            "input_ids": torch.stack(input_ids),
            "targets": torch.stack(targets),
            "mask": torch.stack(masks),
            **{k: torch.tensor(v) for k, v in metadata.items()},
        }


def create_data_loaders(
    num_pairs: int = 100,
    key_length: int = 4,
    value_length: int = 4,
    vocab_size: int = 256,
    batch_size: int = 32,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, KeyValueMemorizationDataset]:
    """
    Create train and eval data loaders.

    Returns:
        Tuple of (train_loader, eval_loader, dataset)
    """
    dataset = KeyValueMemorizationDataset(
        num_pairs=num_pairs,
        key_length=key_length,
        value_length=value_length,
        vocab_size=vocab_size,
        seed=seed,
    )

    # Split indices
    num_train = int(len(dataset) * train_split)
    train_indices = list(range(num_train))
    eval_indices = list(range(num_train, len(dataset)))

    # Create subset datasets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    eval_subset = torch.utils.data.Subset(dataset, eval_indices)

    collator = MemorizationCollator(pad_token=dataset.PAD)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    return train_loader, eval_loader, dataset
