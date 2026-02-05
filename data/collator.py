from typing import List, Dict, Any
import numpy as np
import torch

class MOTIFCollator:
    def __init__(
            self, 
            normalize_images: bool = True, 
            device: str = "cpu"
            ):
        self.normalize_images = normalize_images
        self.device = device

        self.field_specs = {
            "state": torch.float32,
            "state_mask": torch.bool,
            "action": torch.float32,
            "action_mask": torch.bool,
            "images": torch.float32 if normalize_images else torch.uint8,
            'rel_state': torch.float32,
            'progress': torch.float32,
        }

    @staticmethod
    def _np_to_torch(x: np.ndarray, target_dtype=None):
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            if target_dtype:
                t = t.to(target_dtype)
            elif t.dtype == torch.float64:
                t = t.float()
            elif t.dtype == torch.int32:
                t = t.long()
            return t
        return torch.as_tensor(x, dtype=target_dtype) if target_dtype else torch.as_tensor(x)

    def _stack_field(self, batch: List[Dict[str, Any]], key: str, dtype):
        tensors = [self._np_to_torch(sample[key], dtype) for sample in batch]
        return torch.stack(tensors, dim=0).to(self.device, non_blocking=True) 
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for key, tgt_dtype in self.field_specs.items():
            if key in batch[0]:
                out[key] = self._stack_field(batch, key, tgt_dtype)

        if self.normalize_images:
            out["images"] = out["images"] / 255.0  # convert to float32

        out["embodiment_id"] = torch.as_tensor([b["embodiment_id"] for b in batch],
                                               dtype=torch.long, device=self.device)
        out["language"] = [b["language"] for b in batch]
        return out
