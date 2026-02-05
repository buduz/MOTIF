from typing import Any, Dict, Optional, Union
import json
from pathlib import Path
import numpy as np
import torch

from data.dataset import ModalityConfig
from data.schema import DatasetMetadata
from data.data_config import DATA_CONFIG_MAP
from data.embodiment_tags import EMBODIMENT_TAGS

from model.stage2_3 import build_model

class InferenceWrapper:
    def __init__(self, 
                 model_path: Path,
                 denoising_steps: Optional[int] = None,
                 device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
                 ):

        cfg_path = Path(model_path).parent / "config.json"
        with open(cfg_path, "r") as f:
            config = json.load(f)
        
        self.model = build_model(config['models']['stage3'])
        ckpt = torch.load(model_path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(ckpt['model'], strict=False)

        print("[load stage3] missing keys (ignored):")
        for k in missing:
            if not (k.startswith("mkl.dino_encoder.") or k.startswith("mkl.text_encoder.")):
                print("  ", k)
        print("[load stage3] unexpected keys:", unexpected)

        self.model.to(device)
        self.model.eval()
        
        self.cfg_path = Path(model_path).parent
        self.device = device
        self.cur_robot = None
        self.cur_embodiment_tag = None
        self.cur_modality_config = None
        self.cur_transform = None

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def set_robot_uid(self, cur_robot: str):
        """Set the robot uid for the model."""
        if cur_robot not in EMBODIMENT_TAGS:
            raise ValueError(f"Unsupported robot uid: {cur_robot}. Supported robots: {list(EMBODIMENT_TAGS.keys())}")
        
        self.cur_robot = cur_robot
        self.cur_embodiment_tag = EMBODIMENT_TAGS[cur_robot]
        
        if self.cur_embodiment_tag not in DATA_CONFIG_MAP:
             raise ValueError(f"Tag {self.cur_embodiment_tag} not found in DATA_CONFIG_MAP")
             
        data_config_obj = DATA_CONFIG_MAP[self.cur_embodiment_tag]
        
        self.cur_modality_config = data_config_obj.modality_config()
        self.cur_transform = data_config_obj.transform()
        
        self.cur_transform.eval()
        
        print(f'current embodiment:{self.cur_robot} (tag: {self.cur_embodiment_tag})')
        
        # Load transforms
        self._load_metadata(self.cfg_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.cur_embodiment_tag)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.cur_embodiment_tag}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self.cur_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self.cur_modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self.cur_modality_config:
            self._state_delta_indices = np.array(self.cur_modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        if self.cur_transform is None:
            raise RuntimeError("Robot UID not set. Please call set_robot_uid() first.")
        # Ensure correct dimensions before applying transforms
        return self.cur_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        if self.cur_transform is None:
            raise RuntimeError("Robot UID not set. Please call set_robot_uid() first.")
        return self.cur_transform.unapply(action)
    
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        if self.cur_robot is None:
            raise RuntimeError("Robot UID not set. Please call set_robot_uid() first.")
        
        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        # Apply transforms
        normalized_input = self.apply_transforms(observations)
        normalized_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in normalized_input.items()}
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action
    
    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        model_pred = self.model.get_action(normalized_input)
        normalized_action = model_pred["action_pred"]
        return normalized_action
    
    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})
    
    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True
    
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        if self.cur_modality_config is None:
            raise RuntimeError("Robot UID not set. Please call set_robot_uid() first.")
        return self.cur_modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        if self.cur_modality_config is None:
            raise RuntimeError("Robot UID not set. Please call set_robot_uid() first.")
        return self.cur_modality_config

#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data
