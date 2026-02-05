import os
import json
from pathlib import Path
import bisect

from torch.utils.data import Dataset


class MultiDatasetWrapper(Dataset):
    def __init__(self, datasets, sampling_weights=None):
        """
        Args:
            datasets: list of LeRobotSingleDataset instances
            sampling_weights: list of float, same length as datasets, should sum to 1
        """
        self.datasets = datasets
        self.dataset_lengths = [len(d) for d in datasets]
        self.cum_lengths = self._compute_cum_lengths(self.dataset_lengths)  
        self.total_length = sum(self.dataset_lengths)

    def _compute_cum_lengths(self, lengths):
        cum = [0]
        for l in lengths:
            cum.append(cum[-1] + l)
        return cum

    def __len__(self):
        # You can set this arbitrarily or to a multiple of epoch size
        return self.total_length

    def __getitem__(self, idx):
        # Randomly choose a dataset based on weights
        dataset_idx = bisect.bisect_right(self.cum_lengths, idx) - 1
        local_idx = idx - self.cum_lengths[dataset_idx]
        return self.datasets[dataset_idx][local_idx]
    
    def save_metadata(self, path: str):
        path = Path(path)
        exp_cfg_dir = path / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        if os.path.exists(exp_cfg_dir / "metadata.json"):
            with open(exp_cfg_dir / "metadata.json", "r") as f:
                metadata_json = json.load(f)
        metadata_json = {}
        for i in range(len(self.datasets)):
            metadata_json.update(
                {self.datasets[i].tag: self.datasets[i].metadata.model_dump(mode="json")}
            )
        with open(exp_cfg_dir / "metadata.json", "w") as f:
            json.dump(metadata_json, f, indent=4)
