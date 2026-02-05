import dataclasses
import logging
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import deque

from experiment.lerobot_datasets.lerobot_dataset import LeRobotDataset
from eval.websocket_policy_server import ExternalRobotInferenceClient

class RowQueue:
    def __init__(self):
        # Use deque as the underlying container, popleft() has O(1) time complexity
        self.queue = deque()

    def put(self, batch_numpy):
        """Store numpy array with shape (16, 7)"""
        # Split (16, 7) into 16 (1, 7) arrays and enqueue them sequentially
        for i in range(batch_numpy.shape[0]):
            self.queue.append(batch_numpy[i:i+1, :])

    def get(self):
        """Retrieve numpy array with shape (1, 7)"""
        if self.queue:
            return self.queue.popleft()
        return None 
    
    def is_empty(self):
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)

def plot_prediction_vs_groundtruth(gt: np.ndarray, pred: np.ndarray, save_path: str):
    """
    Plot the relationship between ground truth and inference values over time and save it.
    """
    assert gt.shape == pred.shape, "gt and pred shapes must be the same"
    n, d = gt.shape

    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create figure
    fig, axes = plt.subplots(d, 1, figsize=(10, 2 * d), sharex=True)

    for i in range(d):
        axes[i].plot(range(n), gt[:, i], label='Ground Truth', color='blue', linestyle='-')
        axes[i].plot(range(n), pred[:, i], label='Prediction', color='red', linestyle='--')
        axes[i].set_ylabel(f'Dim {i+1}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Step')

    plt.suptitle('Ground Truth vs Prediction Over Time (All Dimensions)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Plot saved to {save_path}")

@dataclasses.dataclass
class Args:
    # --- Server Parameters ---
    host: str = "0.0.0.0"
    port: int = 5555
    
    # --- Policy Parameters ---
    replan_steps: int = 5
    robot_uid: str = "arx5_real_rel"
    
    # --- Data & Output Paths ---
    dataset_path: str = "./demo_data_processed/arx5"
    # Only keep the directory path here
    save_dir: str = "./experiment/"

def eval_dp(args: Args) -> None:
    q = RowQueue()
    
    # Load dataset
    logging.info(f"Loading dataset from: {args.dataset_path}")
    dataset = LeRobotDataset(args.dataset_path)
    episode_data_index = dataset.episode_data_index
    from_indices = episode_data_index['from']
    to_indices = episode_data_index['to']
    
    # Initialize inference client
    logging.info(f"Connecting to policy server at {args.host}:{args.port} with robot_uid={args.robot_uid}")
    policy_client = ExternalRobotInferenceClient(host=args.host, port=args.port)
    policy_client.set_robot_uid(args.robot_uid) 

    robot_name = os.path.basename(os.path.normpath(args.dataset_path))
    
    target_episode_idx = 10 
    
    start = from_indices[target_episode_idx]
    end = to_indices[target_episode_idx]
    
    gt_actions = []
    pred_actions = []
    
    # Logic to read task description from data
    task_description = None
    
    logging.info(f"Starting inference for Robot: {robot_name}, Episode Index: {target_episode_idx}")
    for ep_idx in range(start, end):
        data = dataset[ep_idx]
        
        # Extract task description from the first frame/data point encountered
        if task_description is None:
            task_description = data['task']
            logging.info(f"Task description loaded from data: {task_description}")
            
        # Prepare inputs
        image = (data['observation.images.image'].permute(1, 2, 0).unsqueeze(0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        wrist_image = (data['observation.images.wrist_image'].permute(1, 2, 0).unsqueeze(0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        state = data['observation.state'].to('cuda').unsqueeze(0).cpu().numpy()
        gt_action = data['action'][:7]
        
        element = {
            "video.image": image,
            "video.wrist_image": wrist_image,
            "state": state,
            "annotation.human.task_description": [task_description], 
        }
        
        # Inference logic with queue
        if q.is_empty():
            action_chunk = policy_client.get_action(element)
            pred_action = np.concatenate([
                action_chunk['action.position'],
                action_chunk['action.rotation'],
                action_chunk['action.gripper'][:, None]
            ], axis=1) 
            q.put(pred_action[:args.replan_steps])
        
        gt_actions.append(gt_action.cpu().numpy())
        pred_actions.append(q.get().squeeze(0))
        
    gt_actions = np.array(gt_actions)
    pred_actions = np.array(pred_actions)
    
    # Construct final save path: /.../experiment/arx5_episode_10.png
    filename = f"{robot_name}_episode_{target_episode_idx}.png"
    full_save_path = os.path.join(args.save_dir, filename)
    
    plot_prediction_vs_groundtruth(gt_actions, pred_actions, save_path=full_save_path)

def main():
    parser = argparse.ArgumentParser(description="Eval DP Policy")
    default_args = Args()

    # Add arguments corresponding to Args dataclass
    parser.add_argument("--host", type=str, default=default_args.host, help="Server host")
    parser.add_argument("--port", type=int, default=default_args.port, help="Server port")
    parser.add_argument("--replan-steps", type=int, default=default_args.replan_steps, help="Number of steps to plan ahead")
    parser.add_argument("--dataset-path", type=str, default=default_args.dataset_path, help="Path to the dataset")
    parser.add_argument("--robot-uid", type=str, default=default_args.robot_uid, help="Robot UID for the policy")
    parser.add_argument("--save-dir", type=str, default=default_args.save_dir, help="Directory to save the comparison plot")

    parsed = parser.parse_args()
    args = Args(**vars(parsed))
    
    eval_dp(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()