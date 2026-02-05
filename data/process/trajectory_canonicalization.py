import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import pytorch3d.transforms as pt
from tqdm import tqdm 

'''
Kinematic Trajectory Canonicalization.

Usage:
python data/process/trajectory_canonicalization.py \
    --dataset_path path/to/data \
    --save_path path/to/save_path \
    --in_rotation_type quat \
    --out_action_type rel \
    --out_rotation_type quat
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Batch process ManiSkill datasets with configurable features.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path containing sub-datasets")
    parser.add_argument("--save_path", type=str, required=True, help="Root path to save processed sub-datasets")
    
    parser.add_argument("--in_rotation_type", type=str, choices=['quat', 'rpy'], default='rpy',
                    help="INPUT rotation format in state_array (quat: 4 dims, rpy: 3 dims).")

    parser.add_argument("--out_action_type", type=str, choices=['rel', 'delta', 'state'], default='rel', 
                        help="'rel': Relative to start frame. 'delta': Step-wise change. 'state': Raw state.")
    
    parser.add_argument("--out_rotation_type", type=str, choices=['quat', 'rot6d', 'rpy'], default='quat',
                        help="OUTPUT rotation representation.")
    
    return parser.parse_args()

def mat_transform(rot_matrices, rot_type):
    """Converts a batch of rotation matrices to the specified OUTPUT format."""
    if rot_type == 'quat':
        return pt.matrix_to_quaternion(rot_matrices)
    elif rot_type == 'rot6d':
        return pt.matrix_to_rotation_6d(rot_matrices)
    elif rot_type == 'rpy':
        return pt.matrix_to_euler_angles(rot_matrices, convention="XYZ")
    else:
        raise ValueError(f"Unsupported rotation type: {rot_type}")

def calculate_diff(A_xyz, A_R, A_width, B_xyz, B_R, B_width, rot_type):
    """Calculates transformation of B relative to A using rotation matrices."""
    diff_xyz = B_xyz - A_xyz
    diff_width = B_width - A_width

    # Relative Rotation: R_diff = R_B * R_A^T
    diff_R = torch.bmm(B_R, A_R.transpose(1, 2))
    diff_rot_encoded = mat_transform(diff_R, rot_type)
    
    return torch.cat([diff_xyz, diff_rot_encoded, diff_width], dim=1)

def process_trajectory(state_array, action_type, out_rot_type, in_rot_type='quat'):
    """
    Computes trajectory features.
    Input state_array: numpy (N, D)
        if quat: [x, y, z, qw, qx, qy, qz, width]
        if rpy:  [x, y, z, r, p, y, width]
    """
    if action_type == 'state':
        return state_array

    device = torch.device('cpu')
    states = torch.from_numpy(state_array).float().to(device)
    
    # 1. Parse Position
    pos = states[:, :3]
    
    # 2. Parse Rotation and unify to Rotation Matrix
    if in_rot_type == 'quat':
        rot_data = states[:, 3:7]
        width = states[:, 7:8]
        rot_mat = pt.quaternion_to_matrix(rot_data)
    elif in_rot_type == 'rpy':
        rot_data = states[:, 3:6]
        width = states[:, 6:7]
        # Common XYZ convention for ManiSkill/Lerobot
        rot_mat = pt.euler_angles_to_matrix(rot_data, convention="XYZ")
    else:
        raise ValueError(f"Unsupported input rotation type: {in_rot_type}")
    
    # Prepare reference frames (First frame)
    start_pos = pos[0:1].expand_as(pos)
    start_rot_mat = rot_mat[0:1].expand_as(rot_mat)
    start_width = width[0:1].expand_as(width)

    if action_type == 'rel':
        features = calculate_diff(start_pos, start_rot_mat, start_width, 
                                  pos, rot_mat, width, out_rot_type)
        
    elif action_type == 'delta':
        # Step 1: Calculate relative to start frame (Quat used as intermediate)
        rel_feat_raw = calculate_diff(start_pos, start_rot_mat, start_width, 
                                      pos, rot_mat, width, rot_type='quat')
        
        rel_pos = rel_feat_raw[:, :3]
        rel_quat = rel_feat_raw[:, 3:7]
        rel_width = rel_feat_raw[:, 7:8]
        
        # Construct Next Step Sequence (Padding last frame)
        next_pos = torch.cat((rel_pos[1:], rel_pos[-1:]), dim=0)
        next_quat = torch.cat((rel_quat[1:], rel_quat[-1:]), dim=0)
        next_width = torch.cat((rel_width[1:], rel_width[-1:]), dim=0)
        
        # Step 2: Calculate Delta between current and next step
        rel_R = pt.quaternion_to_matrix(rel_quat)
        next_R = pt.quaternion_to_matrix(next_quat)
        
        features = calculate_diff(rel_pos, rel_R, rel_width,
                                  next_pos, next_R, next_width, out_rot_type)
        
    return features.numpy()

def process_single_dataset(src_dir, dst_dir, args):
    """Processes a single sub-dataset and updates metadata."""
    dataset_name = os.path.basename(src_dir)
    print(f"\n[{dataset_name}] Copying files...")
    
    if os.path.exists(dst_dir):
        os.system(f'rm -r "{dst_dir}"')
    os.makedirs(dst_dir, exist_ok=True)
    os.system(f'cp -r "{src_dir}"/* "{dst_dir}"')
    
    parquet_dir = os.path.join(dst_dir, 'data/chunk-000')
    if not os.path.exists(parquet_dir):
        print(f"[{dataset_name}] Skipping: No data/chunk-000 found.")
        return

    files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
    print(f"[{dataset_name}] Processing {len(files)} episodes...")

    orig_action_dim = 0

    for file in tqdm(files, desc=f"Processing {dataset_name}", leave=False):
        file_path = os.path.join(parquet_dir, file)
        df = pd.read_parquet(file_path)
        
        state_array = np.stack(df["observation.state"].to_numpy())
        original_action = np.stack(df["action"].to_numpy())
        
        if orig_action_dim == 0 and original_action.shape[-1] > 0:
            orig_action_dim = original_action.shape[-1]
        
        new_features = process_trajectory(state_array, args.out_action_type, args.out_rotation_type, args.in_rotation_type)
        
        # Calculate Progress (0.0 to 1.0)
        progress = np.array([(i + 1) / len(df) for i in range(len(df))], dtype='float32').reshape(-1, 1)
        
        # Concatenate: [Original Action, New Features, Progress]
        new_action = np.concatenate([original_action, new_features, progress], axis=-1)
        
        df["action"] = list(new_action)
        df.to_parquet(file_path)

    # Update Episode Stats
    stats_path = os.path.join(dst_dir, 'meta/episodes_stats.jsonl')
    if os.path.exists(stats_path):
        updated_stats = []
        with open(stats_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                
                ep_filename = f"episode_{obj['episode_index']:06d}.parquet"
                parquet_path = os.path.join(parquet_dir, ep_filename)
                
                if os.path.exists(parquet_path):
                    df = pd.read_parquet(parquet_path)
                    action_data = np.stack(df["action"].to_numpy())
                    
                    obj['stats']['action']['max'] = np.max(action_data, axis=0).tolist()
                    obj['stats']['action']['min'] = np.min(action_data, axis=0).tolist()
                    obj['stats']['action']['mean'] = np.mean(action_data, axis=0).tolist()
                    obj['stats']['action']['std'] = np.std(action_data, axis=0).tolist()
                
                updated_stats.append(obj)
        
        with open(stats_path, "w", encoding="utf-8") as f:
            for obj in updated_stats:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[{dataset_name}] Stats updated.")

    # Update Info Metadata
    info_path = os.path.join(dst_dir, 'meta/info.json')
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        
        first_file = os.path.join(parquet_dir, files[0])
        df_temp = pd.read_parquet(first_file)
        total_dim = len(df_temp["action"].iloc[0])
        
        info['features']['action'] = {
            'dtype': 'float32', 
            'shape': [total_dim], 
            'names': ['action']
        }
        
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print(f"[{dataset_name}] Info updated (Action Dim: {total_dim}).")

    # Update Modality Metadata (Dynamic)
    modality_path = os.path.join(dst_dir, 'meta/modality.json')
    if os.path.exists(modality_path) and orig_action_dim > 0:
        with open(modality_path, "r", encoding="utf-8") as f:
            modality = json.load(f)
        
        if "action" not in modality:
            modality["action"] = {}

        rot_config = {
            'quat': (4, 'quaternion'),
            'rot6d': (6, 'rotation_6d'),
            'rpy': (3, 'euler_angles_rpy')
        }
        
        if args.out_rotation_type not in rot_config:
            print(f"[Warning] Unknown rotation type {args.out_rotation_type}, skipping modality update.")
        else:
            rot_dim, rot_type_name = rot_config[args.out_rotation_type]
            
            current_idx = orig_action_dim
            
            modality["action"]["rel_state_position"] = {
                "start": current_idx,
                "end": current_idx + 3
            }
            current_idx += 3
            
            modality["action"]["rel_state_rotation"] = {
                "start": current_idx,
                "end": current_idx + rot_dim,
                "rotation_type": rot_type_name
            }
            current_idx += rot_dim
            
            modality["action"]["rel_state_gripper"] = {
                "start": current_idx,
                "end": current_idx + 1
            }
            current_idx += 1
            
            # progress (1 dim)
            modality["action"]["progress"] = {
                "start": current_idx,
                "end": current_idx + 1
            }

            with open(modality_path, "w", encoding="utf-8") as f:
                json.dump(modality, f, indent=4, ensure_ascii=False)
            print(f"[{dataset_name}] Modality updated (Dynamic based on dim={orig_action_dim}, rot={args.out_rotation_type}).")

def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Input Root:   {args.dataset_path}")
    print(f"  Output Root:  {args.save_path}")
    print(f"  Action Type:  {args.out_action_type}")
    print(f"  Out Rot Type: {args.out_rotation_type}")
    print(f"  In Rot Type:  {args.in_rotation_type}")
    print("-" * 50)

    sub_dirs = sorted([
        d for d in os.listdir(args.dataset_path) 
        if os.path.isdir(os.path.join(args.dataset_path, d)) and not d.startswith('.')
    ])

    if not sub_dirs:
        print("No sub-directories found in dataset_path.")
        return

    print(f"Found {len(sub_dirs)} datasets: {sub_dirs}")
    print("-" * 50)

    for sub_dir in sub_dirs:
        src_dir = os.path.join(args.dataset_path, sub_dir)
        dst_dir = os.path.join(args.save_path, sub_dir)
        
        try:
            process_single_dataset(src_dir, dst_dir, args)
        except Exception as e:
            print(f"\nERROR processing {sub_dir}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()